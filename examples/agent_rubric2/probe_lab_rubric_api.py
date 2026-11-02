#!/usr/bin/env python3
"""Bounded lab-gateway probe; stdlib only, no GPU, Ray, or VERL import.

Never retries or discovers capacity by saturation. Stops admitting requests on
the first error. Reported concurrency is a short observed test, not a quota.
"""

import argparse
import concurrent.futures as futures
import datetime as dt
import http.client
import importlib.util
import json
import math
import os
import socket
import ssl
import sys
import time
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse

DEFAULT_URL = "https://api.example.com/v1"
ROOT = Path(__file__).resolve().parents[2]


def load_pool_helpers():
    path = ROOT / "verl/trainer/ppo/rubric_pool.py"
    if not path.is_file():
        raise ValueError("同步 verl/trainer/ppo/rubric_pool.py 后再运行（不需要安装 VERL）。")
    spec = importlib.util.spec_from_file_location("lab_probe_rubric_pool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def endpoint_from_base(base):
    parsed = urlparse(base)
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (parsed.scheme != "https" and not local_http) or not parsed.hostname:
        raise ValueError("base-url 必须使用 HTTPS（仅本机模拟测试允许 HTTP）。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base-url 不得包含凭据、查询参数或 fragment。")
    # Also reject malformed ports before any request is made.
    _ = parsed.port
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward a laboratory credential to a redirected host.
        return None


def request_once(endpoint, key, payload, timeout, ca_file, validator):
    start = time.monotonic()
    result = {"ok": False, "http_status": None, "error": None}
    try:
        context = ssl.create_default_context(cafile=ca_file)
        opener = request.build_opener(NoRedirect(), request.HTTPSHandler(context=context))
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as response:
            result["http_status"] = response.status
            # No streaming; cap response size to avoid an unexpected huge body.
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("oversized response")
        data = json.loads(raw)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            result["error"] = "output_truncated"
        else:
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty response")
            validator(content)
            result["ok"] = True
            result["response_chars"] = len(content)
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            result["usage"] = {
                name: value
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance((value := usage.get(name)), int) and not isinstance(value, bool) and value >= 0
            }
    except error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = {
            400: "bad_request_check_model_or_parameters",
            401: "authentication_failed",
            403: "permission_or_gateway_denied",
            404: "endpoint_or_model_not_found",
            429: "rate_limited",
        }.get(exc.code, "redirect_blocked" if 300 <= exc.code < 400 else "http_error")
        # Preserve only a safe numeric Retry-After, never arbitrary error bodies.
        retry_after = (exc.headers.get("Retry-After", "") if exc.headers else "").strip()
        if retry_after.isdigit() and len(retry_after) <= 8:
            result["retry_after_seconds"] = int(retry_after)
        exc.close()
    except error.URLError as exc:
        reason = exc.reason
        result["error"] = (
            "tls_error"
            if isinstance(reason, ssl.SSLError)
            else "timeout"
            if isinstance(reason, (TimeoutError, socket.timeout))
            else "network_error"
        )
    except ssl.SSLError:
        result["error"] = "tls_error"
    except TimeoutError:
        result["error"] = "timeout"
    except (OSError, http.client.HTTPException):
        result["error"] = "network_error"
    except (ValueError, TypeError, KeyError, IndexError):
        result["error"] = "invalid_response_or_score_schema"
    result["latency_seconds"] = round(time.monotonic() - start, 4)
    return result


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_stage(name, concurrency, planned, records, elapsed, peak, stop_reason):
    good = [r for r in records if r["ok"]]
    latencies = [r["latency_seconds"] for r in good]
    tokens = [r.get("usage", {}).get("total_tokens") for r in good]
    known_tokens = [t for t in tokens if t is not None]
    return {
        "name": name,
        "configured_concurrency": concurrency,
        "observed_peak_inflight": peak,
        "planned_requests": planned,
        "sent_requests": len(records),
        "successful_requests": len(good),
        "success_rate": len(good) / len(records) if records else 0,
        "healthy": len(records) == planned and len(good) == planned,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(elapsed, 4),
        "success_requests_per_second": len(good) / elapsed if elapsed else 0,
        "latency_mean_seconds": sum(latencies) / len(latencies) if latencies else None,
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "p95_is_small_sample": len(good) < 20,
        "total_tokens_mean": sum(known_tokens) / len(known_tokens) if known_tokens else None,
        "usage_available_fraction": len(known_tokens) / len(good) if good else 0,
        "requests": records,
    }


def run_stage(name, concurrency, planned, call, deadline, max_rps, heartbeat_seconds=10, admission_state=None):
    start = time.monotonic()
    admission_state = admission_state if admission_state is not None else {"next_at": start}
    next_admission = admission_state["next_at"]
    last_heartbeat = start
    records, pending = [], {}
    submitted = 0
    peak = 0
    stop_reason = None
    with futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        while pending or (submitted < planned and stop_reason is None):
            now = time.monotonic()
            if now >= deadline and stop_reason is None:
                stop_reason = "time_budget_exhausted"
            # Handle every completed error BEFORE admitting any more work.
            for future in list(pending):
                if not future.done():
                    continue
                index = pending.pop(future)
                record = future.result()
                record["request_index"] = index
                records.append(record)
                if not record["ok"] and stop_reason is None:
                    stop_reason = record["error"] or "request_failed"
            if stop_reason is None and submitted < planned and len(pending) < concurrency and now >= next_admission:
                pending[executor.submit(call, submitted)] = submitted
                submitted += 1
                peak = max(peak, len(pending))
                next_admission = now + 1 / max_rps
                admission_state["next_at"] = next_admission
            if now - last_heartbeat >= heartbeat_seconds:
                print(
                    f"[{name}] 已完成 {len(records)}/{planned}，在途 {len(pending)}，耗时 {now - start:.0f}s",
                    flush=True,
                )
                last_heartbeat = now
            if pending:
                futures.wait(pending, timeout=0.05, return_when=futures.FIRST_COMPLETED)
            elif stop_reason is None and submitted < planned:
                time.sleep(min(0.05, max(0, next_admission - time.monotonic())))
    records.sort(key=lambda record: record["request_index"])
    return summarize_stage(name, concurrency, planned, records, time.monotonic() - start, peak, stop_reason)


def build_profiles(args, pool):
    if args.audit_file:
        audit = json.loads(Path(args.audit_file).read_text(encoding="utf-8"))
        profiles = []
        for group in audit.get("groups", []):
            rubrics = [{"id": item["id"], "text": item["text"]} for item in group.get("items", [])]
            if not rubrics:
                continue
            for transcript in group.get("transcripts", []):
                for offset in range(0, len(rubrics), args.items_per_request):
                    profiles.append(
                        {
                            "task": group["task"],
                            "trajectory": pool.clean_transcript(transcript),
                            "rubrics": rubrics[offset : offset + args.items_per_request],
                        }
                    )
        if not profiles:
            raise ValueError("audit-file 内没有包含 items 和 transcripts 的 rubric pool groups。")
        return profiles, "training_audit"
    criteria = [
        "Search using the requested item type and color.",
        "Inspect the product price before purchase.",
        "Check that the material is cotton.",
        "Select size M before purchase.",
        "Keep the final price below 30 dollars.",
        "Choose the blue color required by the task.",
        "Avoid buying a product with incompatible size.",
        "Read the product description before purchase.",
    ]
    return [
        {
            "task": "Buy a blue cotton shirt, size M, below 30 dollars.",
            "trajectory": (
                "ACTION: search[blue cotton shirt]\nOBSERVATION: [B000TEST01] Blue cotton shirt.\n"
                "ACTION: click[B000TEST01]\nOBSERVATION: Price $25. Material: cotton. Color: blue. Sizes: S, M, L.\n"
                "ACTION: click[M]\nOBSERVATION: Selected size M.\n"
                "ACTION: click[Buy Now]\nOBSERVATION: Purchase completed."
            ),
            "rubrics": [{"id": f"r{i:03d}", "text": text} for i, text in enumerate(criteria)],
        }
    ], "synthetic_short_8_items"


def build_payload(model, messages, max_tokens, send_thinking):
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if send_thinking:
        payload["enable_thinking"] = False
    return payload


def training_estimate(stages, requests_per_step, step_seconds, profile_kind):
    healthy = [s for s in stages if s["name"].startswith("rubric_c") and s["healthy"]]
    if not healthy:
        return {"available": False, "reason": "no_healthy_rubric_stage"}
    # More concurrency need not be faster; choose the measured throughput leader.
    best = max(healthy, key=lambda s: s["success_requests_per_second"])
    rps = best["success_requests_per_second"]
    seconds = requests_per_step / rps
    token_mean = best["total_tokens_mean"]
    return {
        "available": True,
        "estimate_only": True,
        "representative_training_payload": profile_kind == "training_audit",
        "sampled_fastest_concurrency": best["configured_concurrency"],
        "sampled_fastest_observed_peak_inflight": best["observed_peak_inflight"],
        "scoring_requests_per_step": requests_per_step,
        "estimated_scoring_seconds_per_step": seconds,
        "baseline_step_seconds": step_seconds,
        "estimated_added_time_pct": 100 * seconds / step_seconds if step_seconds else None,
        "estimated_api_total_tokens_per_step": requests_per_step * token_mean if token_mean is not None else None,
        "estimated_model_step_seconds": step_seconds + seconds if step_seconds else None,
        "excludes": [
            "local_rubric_generation",
            "extra_teacher_logprob_forward",
            "retries",
            "shared_service_load_changes",
        ],
        "absolute_service_capacity": None,
        "training_ready": "not_established_by_short_probe",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--model", required=True, help="管理员授权的百炼模型 Code；不自动扫描模型")
    parser.add_argument("--api-key-env", default="IAI_API_KEY")
    parser.add_argument("--mode", choices=("smoke", "staged"), default="smoke")
    parser.add_argument("--concurrency", default="1,2,4", help="升序并发档位；默认最高 4")
    parser.add_argument("--requests-per-worker", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=32, help="含连通测试，总请求预算；无自动重试")
    parser.add_argument("--max-rps", type=float, default=1.0, help="全局请求发起速率上限，不是每线程上限")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="停止发新请求的时间预算；在途请求会收尾")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="rubric 评分回复上限，对齐训练默认值")
    parser.add_argument("--omit-enable-thinking", action="store_true", help="网关不支持 enable_thinking 时手动重跑")
    parser.add_argument("--ca-file", help="可选受信任 CA PEM；始终验证 TLS，不提供跳过验证选项")
    parser.add_argument("--audit-file", help="可选 rubric_pool_audit/step_*.json，将其中任务与轨迹发给实验室 API")
    parser.add_argument("--items-per-request", type=int, default=64)
    parser.add_argument(
        "--training-requests-per-step", type=int, default=64, help="默认 8 prompts × 8 轨迹的满负载估计"
    )
    parser.add_argument("--training-step-seconds", type=float, help="不含新增云评分时的实测单步耗时")
    parser.add_argument("--output", help="JSON 文件名；已有文件不会覆盖")
    args = parser.parse_args(argv)
    try:
        args.levels = [int(x.strip()) for x in args.concurrency.split(",")]
        if not args.levels or args.levels != sorted(set(args.levels)) or args.levels[0] != 1:
            raise ValueError("并发档位必须从 1 开始、严格递增，如 1,2,4。")
        if args.levels[-1] > 32:
            raise ValueError("共享接口自检最多支持 32 并发；更高压力测试请交由管理员。")
        for name in (
            "requests_per_worker",
            "max_requests",
            "max_rps",
            "timeout",
            "max_seconds",
            "max_tokens",
            "items_per_request",
            "training_requests_per_step",
        ):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数。")
        if args.max_requests > 256 or args.max_rps > 8:
            raise ValueError("自检请求预算最多 256、发起速率最多 8 RPS；默认分别为 32 和 1。")
        if not math.isfinite(args.cooldown) or args.cooldown < 0:
            raise ValueError("cooldown 必须为有限非负数。")
        if args.training_step_seconds is not None and (
            not math.isfinite(args.training_step_seconds) or args.training_step_seconds <= 0
        ):
            raise ValueError("training-step-seconds 必须为有限正数。")
        endpoint_from_base(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        print(f"未调用 API：请先设置环境变量 {args.api_key_env}，不要把密钥写入脚本。", file=sys.stderr)
        return 2
    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "experiment_outputs/lab_api_probe"
        / (dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}.json")
    )
    if output.exists():
        print("未调用 API：output 已存在，请换文件名。", file=sys.stderr)
        return 2
    pool = load_pool_helpers()
    profiles, profile_kind = build_profiles(args, pool)
    report = {
        "schema_version": 1,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": endpoint_from_base(args.base_url),
        "model": args.model,
        "mode": args.mode,
        "profile_kind": profile_kind,
        "profile_count": len(profiles),
        "profile_trajectory_chars": [len(p["trajectory"]) for p in profiles],
        "profile_rubric_counts": [len(p["rubrics"]) for p in profiles],
        "limits": {"max_requests": args.max_requests, "max_rps": args.max_rps, "max_seconds": args.max_seconds},
        "enable_thinking_sent": not args.omit_enable_thinking,
        "judge_max_tokens": args.max_tokens,
        "stages": [],
        "absolute_max_concurrency": None,
        "notes": [
            "不包含密钥、原始请求/响应文本；HTTP 错误正文不记录。",
            "短时测试不是长期稳定性或最大并发证明；共享配额可能随其他用户负载变化。",
            "连通/JSON 格式通过不等于 rubric 评分质量通过；默认合成短请求不能代表长轨迹。",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a private report before calls. Subsequent writes replace only our own report.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    deadline = time.monotonic() + args.max_seconds
    admission_state = {"next_at": time.monotonic()}
    print(f"模型={args.model}；模式={args.mode}；画像={profile_kind}；JSON={output}", flush=True)
    print("无自动重试；首次错误停止发新请求。在途请求仍需等待收尾。", flush=True)
    if args.audit_file:
        print("将发送 audit 中的任务、动作/观测和 rubric 文本；请确认这些内容允许上传实验室服务。", flush=True)

    def call(payload, validator):
        return request_once(report["endpoint"], key, payload, args.timeout, args.ca_file, validator)

    def check_smoke(text):
        if pool.strip_reasoning(text).strip() != "API_OK":
            raise ValueError("smoke response did not follow instruction")

    try:
        smoke = build_payload(
            args.model,
            [{"role": "user", "content": "Reply with exactly API_OK. No other text."}],
            32,
            not args.omit_enable_thinking,
        )
        stage = run_stage(
            "connectivity",
            1,
            1,
            lambda _: call(smoke, check_smoke),
            deadline,
            args.max_rps,
            admission_state=admission_state,
        )
        report["stages"].append(stage)
        save()
        print(f"[connectivity] {'PASS' if stage['healthy'] else 'FAIL'}", flush=True)
        if not stage["healthy"]:
            for record in stage["requests"]:
                print(f"HTTP={record['http_status']}；原因={record['error']}", flush=True)
        used = stage["sent_requests"]
        if args.mode == "staged" and stage["healthy"]:
            for concurrency in args.levels:
                planned = concurrency * args.requests_per_worker
                if used + planned > args.max_requests or time.monotonic() + args.cooldown >= deadline:
                    report["stop_reason"] = "insufficient_budget_for_next_complete_stage"
                    break
                time.sleep(args.cooldown)

                def judge_call(index):
                    profile = profiles[index % len(profiles)]
                    messages = [
                        {"role": "system", "content": pool.JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": json.dumps(profile, ensure_ascii=False),
                        },
                    ]
                    payload = build_payload(args.model, messages, args.max_tokens, not args.omit_enable_thinking)
                    expected = [item["id"] for item in profile["rubrics"]]
                    return call(payload, lambda text: pool.parse_judge_response(text, expected))

                print(f"[rubric_c{concurrency}] 开始：{planned} 个请求，全局发起速率 ≤ {args.max_rps:g}/s", flush=True)
                stage = run_stage(
                    f"rubric_c{concurrency}",
                    concurrency,
                    planned,
                    judge_call,
                    deadline,
                    args.max_rps,
                    admission_state=admission_state,
                )
                report["stages"].append(stage)
                used += stage["sent_requests"]
                p95 = stage["latency_p95_seconds"]
                print(
                    f"[rubric_c{concurrency}] {stage['successful_requests']}/{stage['sent_requests']} 成功；"
                    f"实测在途峰值 {stage['observed_peak_inflight']}；p95={p95}；停止原因={stage['stop_reason']}",
                    flush=True,
                )
                save()
                if not stage["healthy"]:
                    report["stop_reason"] = stage["stop_reason"] or "incomplete_stage"
                    break
    except KeyboardInterrupt:
        report["stop_reason"] = "user_interrupted"
    finally:
        stages = report["stages"]
        healthy = [s for s in stages if s["name"].startswith("rubric_c") and s["healthy"]]
        report["highest_healthy_configured_concurrency"] = max(
            (s["configured_concurrency"] for s in healthy), default=0
        )
        report["highest_healthy_observed_inflight"] = max((s["observed_peak_inflight"] for s in healthy), default=0)
        report["training_estimate"] = training_estimate(
            stages,
            args.training_requests_per_step,
            args.training_step_seconds,
            profile_kind,
        )
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    if args.mode == "staged":
        print(
            f"本次通过的最高 rubric 并发配置：{report['highest_healthy_configured_concurrency']}；"
            f"观察到的在途峰值：{report['highest_healthy_observed_inflight']}。不是服务并发上限。",
            flush=True,
        )
    else:
        print("本次仅测连通性，未测试 rubric 评分格式和并发。", flush=True)
    estimate = report["training_estimate"]
    if estimate["available"]:
        print(
            f"按 {args.training_requests_per_step} 次评分/step 粗估：新增云评分耗时 "
            f"{estimate['estimated_scoring_seconds_per_step']:.1f}s/step（不含本地生成和额外 forward）。",
            flush=True,
        )
    print(f"报告已保存：{output}", flush=True)
    return 0 if all(s["healthy"] for s in report["stages"]) and not report.get("stop_reason") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError):
        # Never echo exception text: parsers/proxies can include credential-bearing data.
        print("自检未完成：请检查 audit JSON、输出目录、CA 文件及参数。未打印原始错误以保护凭据。", file=sys.stderr)
        sys.exit(2)
