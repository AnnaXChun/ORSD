"""Offline mock/loopback tests; never call a provider or require a real API key."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib import error

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("lab_api_probe", ROOT / "examples/agent_rubric2/probe_lab_rubric_api.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class Response(io.BytesIO):
    status = 200


def fake_chat(content="API_OK", finish="stop"):
    return Response(
        json.dumps(
            {
                "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        ).encode()
    )


class ProbeTests(unittest.TestCase):
    def test_gateway_url_and_unsafe_urls(self):
        self.assertEqual(p.endpoint_from_base(p.DEFAULT_URL), p.DEFAULT_URL + "/chat/completions")
        self.assertEqual(p.endpoint_from_base(p.DEFAULT_URL + "/chat/completions"), p.DEFAULT_URL + "/chat/completions")
        for url in ("http://public.example/v1", "https://secret@example.org/v1", "https://x/v1?key=secret"):
            with self.assertRaises(ValueError):
                p.endpoint_from_base(url)

    def test_request_flag_and_schema_reuse(self):
        pool = p.load_pool_helpers()
        content = '{"scores":[{"id":"r000","score":1,"evidence":"Visible action"}]}'
        opener = SimpleNamespace(open=lambda *a, **k: fake_chat(content))
        payload = p.build_payload("mock", [], 4096, True)
        self.assertIs(payload["enable_thinking"], False)
        self.assertNotIn("enable_thinking", p.build_payload("mock", [], 4096, False))
        with patch.object(p.request, "build_opener", return_value=opener):
            result = p.request_once(
                "https://example.invalid/v1",
                "dummy",
                payload,
                5,
                None,
                lambda text: pool.parse_judge_response(text, ["r000"]),
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["usage"]["total_tokens"], 120)

    def test_truncated_output_and_malformed_matrix_fail(self):
        pool = p.load_pool_helpers()
        for content, finish, expected in (
            ("{}", "length", "output_truncated"),
            ("{}", "stop", "invalid_response_or_score_schema"),
        ):
            opener = SimpleNamespace(open=lambda *a, content=content, finish=finish, **k: fake_chat(content, finish))
            with patch.object(p.request, "build_opener", return_value=opener):
                result = p.request_once(
                    "https://example.invalid/v1",
                    "dummy",
                    {},
                    5,
                    None,
                    lambda text: pool.parse_judge_response(text, ["r000"]),
                )
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], expected)

    def test_rate_limit_no_retry_and_secret_not_echoed(self):
        secret = "dummy-secret-not-a-real-key"
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise error.HTTPError(
                "https://example.invalid", 429, secret, {"Retry-After": "60"}, io.BytesIO(secret.encode())
            )

        with patch.object(p.request, "build_opener", return_value=SimpleNamespace(open=fail)):
            result = p.request_once("https://example.invalid/v1", secret, {}, 5, None, lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["error"], "rate_limited")
        self.assertEqual(result["retry_after_seconds"], 60)
        self.assertNotIn(secret, json.dumps(result))

    def test_stop_after_first_error_without_flooding(self):
        seen = []

        def call(index):
            seen.append(index)
            return {"ok": False, "error": "rate_limited", "latency_seconds": 0.01}

        result = p.run_stage("rubric_c4", 4, 20, call, time.monotonic() + 5, 100)
        self.assertEqual(len(seen), 1)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["stop_reason"], "rate_limited")

    def test_request_count_concurrency_and_global_pacing(self):
        active, peak = 0, 0
        lock = threading.Lock()
        starts = []

        def call(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                starts.append(time.monotonic())
            time.sleep(0.12)
            with lock:
                active -= 1
            return {"ok": True, "error": None, "latency_seconds": 0.12}

        state = {"next_at": time.monotonic()}
        result = p.run_stage("rubric_c2", 2, 4, call, time.monotonic() + 5, 20, admission_state=state)
        self.assertTrue(result["healthy"])
        self.assertEqual(result["sent_requests"], 4)
        self.assertEqual(peak, 2)
        self.assertTrue(all(b - a >= 0.045 for a, b in zip(starts, starts[1:], strict=False)))
        self.assertGreater(state["next_at"], starts[-1])

    def test_deadline_prevents_requests(self):
        result = p.run_stage("rubric_c1", 1, 2, lambda _: self.fail("must not call"), time.monotonic() - 1, 1)
        self.assertEqual(result["sent_requests"], 0)
        self.assertEqual(result["stop_reason"], "time_budget_exhausted")

    def test_report_training_estimate_not_capacity_claim(self):
        stage = p.summarize_stage(
            "rubric_c2",
            2,
            2,
            [
                {"ok": True, "latency_seconds": 2, "usage": {"total_tokens": 500}},
                {"ok": True, "latency_seconds": 2, "usage": {"total_tokens": 500}},
            ],
            2,
            2,
            None,
        )
        result = p.training_estimate([stage], 64, 144, "synthetic_short_8_items")
        self.assertEqual(result["estimated_scoring_seconds_per_step"], 64)
        self.assertFalse(result["representative_training_payload"])
        self.assertIsNone(result["absolute_service_capacity"])
        self.assertTrue(stage["p95_is_small_sample"])

    def test_audit_profiles_omit_labels_sources_and_chunk(self):
        pool = p.load_pool_helpers()
        audit = {
            "groups": [
                {
                    "task": "buy",
                    "outcomes": [0, 1],
                    "transcripts": ["ACTION: search[x]\nofficial_reward: 1.0"],
                    "items": [
                        {"id": "r0", "text": "inspect", "sources": [1]},
                        {"id": "r1", "text": "buy", "sources": [0]},
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.json"
            path.write_text(json.dumps(audit))
            profiles, kind = p.build_profiles(SimpleNamespace(audit_file=str(path), items_per_request=1), pool)
        self.assertEqual(kind, "training_audit")
        self.assertEqual(len(profiles), 2)
        self.assertNotIn("sources", json.dumps(profiles))
        self.assertNotIn("outcomes", json.dumps(profiles))
        self.assertNotIn("official_reward", profiles[0]["trajectory"])

    def test_no_key_makes_zero_calls(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(p, "request_once") as mock:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(p.main(["--model", "mock"]), 2)
            mock.assert_not_called()

    def test_request_budget_does_not_run_partial_next_stage(self):
        def call(*args, **kwargs):
            payload, validator = args[2], args[-1]
            if payload["max_tokens"] == 32:
                validator("API_OK")
            return {"ok": True, "error": None, "latency_seconds": 0.01}

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"IAI_API_KEY": "dummy"}):
            path = Path(temp) / "report.json"
            with patch.object(p, "request_once", side_effect=call) as mock, contextlib.redirect_stdout(io.StringIO()):
                code = p.main(
                    [
                        "--model",
                        "mock",
                        "--mode",
                        "staged",
                        "--concurrency",
                        "1,2",
                        "--max-requests",
                        "3",
                        "--max-rps",
                        "8",
                        "--cooldown",
                        "0",
                        "--output",
                        str(path),
                    ]
                )
            report = json.loads(path.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(mock.call_count, 3)
            self.assertEqual(report["stop_reason"], "insufficient_budget_for_next_complete_stage")
            self.assertNotIn("dummy", path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_loopback_http_end_to_end_and_redirect_blocking(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append(self.path)
                if self.path == "/redirect/chat/completions":
                    self.send_response(302)
                    self.send_header("Location", "/leak")
                    self.end_headers()
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if payload["max_tokens"] == 32:
                    text = "API_OK"
                else:
                    profile = json.loads(payload["messages"][-1]["content"])
                    text = json.dumps(
                        {
                            "scores": [
                                {"id": item["id"], "score": 1, "evidence": "mock evidence"}
                                for item in profile["rubrics"]
                            ]
                        }
                    )
                raw = fake_chat(text).getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"IAI_API_KEY": "dummy"}):
                path = Path(temp) / "report.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    code = p.main(
                        [
                            "--model",
                            "mock",
                            "--base-url",
                            base + "/v1",
                            "--mode",
                            "staged",
                            "--concurrency",
                            "1,2",
                            "--requests-per-worker",
                            "1",
                            "--max-rps",
                            "8",
                            "--cooldown",
                            "0",
                            "--output",
                            str(path),
                        ]
                    )
                report = json.loads(path.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(received), 4)
            self.assertEqual(report["highest_healthy_configured_concurrency"], 2)
            result = p.request_once(base + "/redirect/chat/completions", "dummy", {}, 2, None, lambda _: None)
            self.assertEqual(result["error"], "redirect_blocked")
            self.assertNotIn("/leak", received)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
