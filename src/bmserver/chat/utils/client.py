import asyncio
import gc
import json
import random
import time
from collections.abc import Awaitable
from contextlib import suppress

import numpy as np
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from tqdm.asyncio import tqdm

from bmserver.chat.schema import (
    DEFAULT_SERVED_MODEL_NAME,
    SERVER_HTTP_TIMEOUT,
    SERVER_START_TIMEOUT,
    WORDS,
    BenchmarkMetric,
    BenchmarkParam,
    RequestMetric,
    RequestParam,
)


async def check_health(*, session: ClientSession) -> bool:
    with suppress(Exception):
        async with session.get(url="/v1/models", raise_for_status=True):
            return True
    return False


async def wait_for_health(*, base_url: str) -> bool:
    async with ClientSession(
        base_url=base_url,
        headers={"Content-Type": "application/json"},
        timeout=ClientTimeout(total=SERVER_HTTP_TIMEOUT),
    ) as session:
        start_time: float = time.perf_counter()
        while time.perf_counter() - start_time < SERVER_START_TIMEOUT:
            if await check_health(session=session):
                return True
            await asyncio.sleep(delay=1)
        return False


async def detect_text_tokens(*, session: ClientSession, messages: list[dict]) -> int:
    async with session.post(
        url="/tokenize",
        json={"messages": messages, "add_generation_prompt": True},
        raise_for_status=True,
    ) as response:
        payload: dict = await response.json()
        return payload["count"]


async def generate_request_param(
    *,
    session: ClientSession,
    benchmark_param: BenchmarkParam,
    progress_bar: tqdm | None,
) -> RequestParam:
    text_content: str = "".join(
        random.choices(population=WORDS, k=benchmark_param.num_prompt_tokens)
    )
    message_content: list[dict] = [{"type": "text", "text": text_content}]
    messages: list[dict] = [{"role": "user", "content": message_content}]
    while True:
        num_tokens: int = await detect_text_tokens(session=session, messages=messages)
        delta_tokens: int = benchmark_param.num_prompt_tokens - num_tokens
        text_content = messages[0]["content"][0]["text"]
        if delta_tokens < -len(text_content):
            raise RuntimeError("Not enough tokens to generate request messages.")
        if delta_tokens == 0:
            break
        if delta_tokens > 0:
            text_content += "".join(random.choices(population=WORDS, k=delta_tokens))
            messages[0]["content"][0]["text"] = text_content
            continue
        if delta_tokens < 0:
            messages[0]["content"][0]["text"] = text_content[:delta_tokens]
            continue
    if progress_bar is not None:
        progress_bar.update()
    return RequestParam(
        messages=messages, num_completion_tokens=benchmark_param.num_completion_tokens
    )


async def generate_request_params(
    *, base_url: str, benchmark_param: BenchmarkParam
) -> list[RequestParam]:
    async with ClientSession(
        base_url=base_url,
        connector=TCPConnector(limit=benchmark_param.num_requests),
        headers={"Content-Type": "application/json"},
        timeout=ClientTimeout(total=SERVER_HTTP_TIMEOUT),
    ) as session:
        with tqdm(
            desc="Generating request params", total=benchmark_param.num_requests
        ) as progress_bar:
            coros: list[Awaitable[RequestParam]] = [
                generate_request_param(
                    session=session,
                    benchmark_param=benchmark_param,
                    progress_bar=progress_bar,
                )
                for _ in range(benchmark_param.num_requests)
            ]
            return await asyncio.gather(*coros)


async def send_request(
    *, session: ClientSession, request_param: RequestParam, progress_bar: tqdm | None
) -> RequestMetric:
    ttft: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    start_time: float = time.perf_counter()
    async with session.post(
        url="/v1/chat/completions",
        json={
            "model": DEFAULT_SERVED_MODEL_NAME,
            "messages": request_param.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": request_param.num_completion_tokens,
            "ignore_eos": True,
        },
        raise_for_status=True,
    ) as response:
        async for chunk_bytes in response.content:
            chunk_bytes: bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk: str = chunk_bytes.decode(encoding="utf-8")
            chunk = chunk.removeprefix("data: ")
            if chunk == "[DONE]":
                continue
            if ttft == 0.0:
                ttft = time.perf_counter() - start_time
            usage: dict | None = json.loads(s=chunk).get("usage")
            if usage is not None:
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
        latency: float = time.perf_counter() - start_time
        ttft_ms: float = ttft * 1000
        tpot_ms: float = (latency - ttft) * 1000 / request_param.num_completion_tokens
        if progress_bar is not None:
            progress_bar.update()
        return RequestMetric(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
        )


async def send_requests(
    base_url: str, request_params: list[RequestParam]
) -> BenchmarkMetric:
    gc.collect()
    gc.freeze()
    async with ClientSession(
        base_url=base_url,
        connector=TCPConnector(limit=len(request_params)),
        headers={"Content-Type": "application/json"},
        timeout=ClientTimeout(total=SERVER_HTTP_TIMEOUT),
    ) as session:
        with tqdm(desc="Sending requests", total=len(request_params)) as progress_bar:
            tasks: list[Awaitable[RequestMetric]] = []
            start_time: float = time.perf_counter()
            for request_param in request_params:
                task: Awaitable[RequestMetric] = send_request(
                    session=session,
                    request_param=request_param,
                    progress_bar=progress_bar,
                )
                tasks.append(task)
            request_metrics: list[RequestMetric] = await asyncio.gather(*tasks)
            end_time: float = time.perf_counter()
    duration: float = end_time - start_time
    prompt_tokens: int = sum(metric.prompt_tokens for metric in request_metrics)
    completion_tokens: int = sum(metric.completion_tokens for metric in request_metrics)
    latency_values: list[float] = [metric.latency for metric in request_metrics]
    ttft_ms_values: list[float] = [metric.ttft_ms for metric in request_metrics]
    tpot_ms_values: list[float] = [metric.tpot_ms for metric in request_metrics]
    return BenchmarkMetric(
        request_throughput_min=len(request_metrics) / duration * 60,
        completion_token_throughput=completion_tokens / duration,
        total_token_throughput=(prompt_tokens + completion_tokens) / duration,
        mean_latency=float(np.mean(latency_values)),
        mean_ttft_ms=float(np.mean(ttft_ms_values)),
        mean_tpot_ms=float(np.mean(tpot_ms_values)),
    )
