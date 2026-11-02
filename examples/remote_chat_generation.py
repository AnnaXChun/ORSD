#!/usr/bin/env python3
"""Small OpenAI-compatible chat generation adapter for standalone experiments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class RemoteSample:
    text: str
    token_count: int


@dataclass
class RemoteRequestOutput:
    outputs: list[RemoteSample]


class RemoteChatGenerator:
    """Generate independent samples through an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        concurrency: int,
        timeout: float,
        max_retries: int = 3,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("Remote generation concurrency must be positive")
        if max_retries < 0:
            raise ValueError("Remote generation max retries cannot be negative")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The remote backend requires the `openai` package in this environment."
            ) from exc
        self._client_cls = AsyncOpenAI
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries

    async def _generate_one(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
        enable_thinking: bool,
        semaphore: asyncio.Semaphore,
        client: Any,
    ) -> RemoteSample:
        extra_body: dict[str, Any] = {"enable_thinking": enable_thinking}
        if top_k >= 0:
            extra_body["top_k"] = top_k

        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    stream = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=seed,
                        n=1,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body=extra_body,
                    )
                    reasoning_parts: list[str] = []
                    content_parts: list[str] = []
                    token_count = 0
                    async for chunk in stream:
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            token_count = int(
                                getattr(usage, "completion_tokens", 0) or token_count
                            )
                        for choice in getattr(chunk, "choices", []) or []:
                            delta = getattr(choice, "delta", None)
                            if delta is None:
                                continue
                            reasoning_delta = getattr(
                                delta, "reasoning_content", None
                            )
                            content_delta = getattr(delta, "content", None)
                            if reasoning_delta:
                                reasoning_parts.append(str(reasoning_delta))
                            if content_delta:
                                content_parts.append(str(content_delta))
                    reasoning = "".join(reasoning_parts)
                    content = "".join(content_parts)
                    if reasoning:
                        text = f"<think>\n{reasoning}\n</think>\n{content}"
                    else:
                        text = content
                    if token_count <= 0:
                        # Only used for the descriptive length statistic when an
                        # OpenAI-compatible endpoint omits usage from stream chunks.
                        token_count = max(1, len(text) // 4)
                    return RemoteSample(text=text, token_count=token_count)
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(2**attempt, 8))
            assert last_error is not None
            raise RuntimeError(
                f"Remote generation failed after {self.max_retries + 1} attempts"
            ) from last_error

    async def _generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        n: int,
        seed: int,
        enable_thinking: bool,
    ) -> list[RemoteRequestOutput]:
        semaphore = asyncio.Semaphore(self.concurrency)
        client = self._client_cls(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        total = len(messages_batch) * n

        async def indexed_sample(index: int, coroutine: Any) -> tuple[int, RemoteSample]:
            return index, await coroutine

        tasks = [
            asyncio.create_task(
                indexed_sample(
                    prompt_index * n + sample_index,
                    self._generate_one(
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=max_tokens,
                        seed=seed + prompt_index * n + sample_index,
                        enable_thinking=enable_thinking,
                        semaphore=semaphore,
                        client=client,
                    ),
                )
            )
            for prompt_index, messages in enumerate(messages_batch)
            for sample_index in range(n)
        ]
        flat_samples: list[RemoteSample | None] = [None] * total
        completed = 0
        report_every = max(1, min(self.concurrency, total))
        try:
            for future in asyncio.as_completed(tasks):
                index, sample = await future
                flat_samples[index] = sample
                completed += 1
                if completed % report_every == 0 or completed == total:
                    print(
                        f"[DashScope] completed {completed}/{total} generations",
                        flush=True,
                    )
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            await client.close()
        ordered_samples = [sample for sample in flat_samples if sample is not None]
        if len(ordered_samples) != total:
            raise RuntimeError(
                f"Remote generation returned {len(ordered_samples)}/{total} samples"
            )
        return [
            RemoteRequestOutput(outputs=ordered_samples[index * n : (index + 1) * n])
            for index in range(len(messages_batch))
        ]

    def generate(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        n: int,
        seed: int,
        enable_thinking: bool,
    ) -> list[RemoteRequestOutput]:
        if n <= 0:
            raise ValueError("Remote generation sample count must be positive")
        return asyncio.run(
            self._generate_batch(
                messages_batch,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                enable_thinking=enable_thinking,
            )
        )


def completion_token_count(output: Any) -> int:
    """Read a completion length from either vLLM or the remote adapter."""
    if hasattr(output, "token_count"):
        return int(output.token_count)
    return len(output.token_ids)
