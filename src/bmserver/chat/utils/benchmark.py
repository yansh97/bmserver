import asyncio
import multiprocessing as mp
import multiprocessing.context as mpc
import os
import shlex
from argparse import Namespace
from collections.abc import Awaitable
from typing import Any

from bmhub.backend import HubBackend
from vllm import LLM, EngineArgs
from vllm.config import CacheConfig
from vllm.utils import FlexibleArgumentParser

from bmserver.chat.database import ORMBenchmarkResult, database_session
from bmserver.chat.schema import (
    MAX_CONCURRENT_REQUESTS,
    BenchmarkMetric,
    BenchmarkParam,
    BenchmarkResult,
    Model,
    RequestParam,
    ServeParam,
    ServeStatus,
)
from bmserver.chat.utils.client import generate_request_params, send_requests
from bmserver.chat.utils.server import vllm_base_command
from bmserver.schema import Environment


def detect_serve_status_worker(
    *,
    environment: Environment,
    model: Model,
    serve_param: ServeParam,
    queue: mp.SimpleQueue,
) -> None:
    # TODO: Hack for "vllm_config"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    command: str = vllm_base_command(
        environment=environment, model=model, serve_param=serve_param
    )
    parser: FlexibleArgumentParser = EngineArgs.add_cli_args(
        parser=FlexibleArgumentParser()
    )
    args: Namespace = parser.parse_args(args=shlex.split(s=command))
    llm = LLM(**vars(args))
    cache_config: CacheConfig = (
        llm.llm_engine.engine_core.engine_core.scheduler.cache_config  # Hack  # type: ignore  # noqa: E501
    )
    assert cache_config.num_gpu_blocks is not None
    max_cache_tokens: int = cache_config.block_size * cache_config.num_gpu_blocks
    max_requests: int = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    max_requests = min(max_requests, MAX_CONCURRENT_REQUESTS)
    status = ServeStatus(max_cache_tokens=max_cache_tokens, max_requests=max_requests)
    queue.put(obj=status)


def detect_serve_status_in_process(
    *, environment: Environment, model: Model, serve_param: ServeParam
) -> ServeStatus:
    ctx: mpc.SpawnContext = mp.get_context(method="spawn")
    queue: mp.SimpleQueue = ctx.SimpleQueue()
    worker_kwargs: dict[str, Any] = {
        "environment": environment,
        "model": model,
        "serve_param": serve_param,
        "queue": queue,
    }
    worker_process = ctx.Process(
        target=detect_serve_status_worker, kwargs=worker_kwargs
    )
    try:
        worker_process.start()
        worker_process.join()
    finally:
        worker_process.terminate()
        worker_process.join()
    if queue.empty():
        raise RuntimeError("Failed to detect vLLM config.")
    return queue.get()


def expand_num_requests(max_requests: int) -> list[int]:
    values: list[int] = []
    tmp = 1
    while tmp <= max_requests:
        values.append(tmp)
        tmp *= 2
    if values[-1] != max_requests:
        values.append(max_requests)
    return values


def generate_benchmark_params(
    *, prompt_tokens: int, completion_tokens: int, serve_status: ServeStatus
) -> list[BenchmarkParam]:
    num_request_tokens: int = prompt_tokens + completion_tokens
    max_requests: int = serve_status.max_cache_tokens // num_request_tokens
    max_requests = min(max_requests, serve_status.max_requests)
    if max_requests == 0:
        return []
    params: list[BenchmarkParam] = []
    for num_requests in expand_num_requests(max_requests=max_requests):
        param = BenchmarkParam(
            num_prompt_tokens=prompt_tokens,
            num_completion_tokens=completion_tokens,
            max_requests=max_requests,
            num_requests=num_requests,
        )
        params.append(param)
    return params


def benchmark_server(*, benchmark_param: BenchmarkParam, port: int) -> BenchmarkMetric:
    warmup_request_params: list[RequestParam] = asyncio.run(
        main=generate_request_params(
            base_url=f"http://127.0.0.1:{port}",
            benchmark_param=benchmark_param.model_copy(
                update={"num_requests": 1, "num_completion_tokens": 1}
            ),
        )
    )
    request_params: list[RequestParam] = asyncio.run(
        main=generate_request_params(
            base_url=f"http://127.0.0.1:{port}", benchmark_param=benchmark_param
        )
    )

    warmup_coro: Awaitable[BenchmarkMetric] = send_requests(
        base_url=f"http://127.0.0.1:{port}", request_params=warmup_request_params
    )
    asyncio.run(main=warmup_coro)

    benchmark_coro: Awaitable[BenchmarkMetric] = send_requests(
        base_url=f"http://127.0.0.1:{port}", request_params=request_params
    )
    return asyncio.run(main=benchmark_coro)


def upload_benchmark_result(result: BenchmarkResult) -> None:
    with database_session() as session:
        orm_result: ORMBenchmarkResult | None = session.get(
            entity=ORMBenchmarkResult,
            ident=(
                result.environment.nvidia_device_name,
                result.environment.nvidia_device_count,
                result.environment.bmserver_version,
                result.model_spec.name,
                result.benchmark_param.num_prompt_tokens,
                result.benchmark_param.num_completion_tokens,
                result.benchmark_param.num_requests,
            ),
        )
        if orm_result is None:
            orm_result = ORMBenchmarkResult(
                nvidia_device_name=result.environment.nvidia_device_name,
                nvidia_device_count=result.environment.nvidia_device_count,
                bmserver_version=result.environment.bmserver_version,
                model_name=result.model_spec.name,
                num_prompt_tokens=result.benchmark_param.num_prompt_tokens,
                num_completion_tokens=result.benchmark_param.num_completion_tokens,
                num_requests=result.benchmark_param.num_requests,
                nvidia_driver_version=result.environment.nvidia_driver_version,
                torch_version=result.environment.torch_version,
                transformers_version=result.environment.transformers_version,
                vllm_version=result.environment.vllm_version,
                model_modal=result.model_spec.modal,
                model_series=result.model_spec.series,
                model_size=result.model_spec.size,
                model_quant=result.model_spec.quant.value,
                model_tool_call=result.model_spec.tool_call.value,
                model_reasoning=result.model_spec.reasoning.value,
                model_repo_id=result.model_spec.repo_id[HubBackend.HUGGING_FACE],
                vllm_seed=result.serve_param.seed,
                vllm_extra_params=result.serve_param.extra_params,
                vllm_max_cache_tokens=result.serve_status.max_cache_tokens,
                vllm_max_requests=result.serve_status.max_requests,
                max_requests=result.benchmark_param.max_requests,
                request_throughput_min=result.benchmark_metric.request_throughput_min,
                completion_token_throughput=result.benchmark_metric.completion_token_throughput,
                total_token_throughput=result.benchmark_metric.total_token_throughput,
                mean_latency=result.benchmark_metric.mean_latency,
                mean_ttft_ms=result.benchmark_metric.mean_ttft_ms,
                mean_tpot_ms=result.benchmark_metric.mean_tpot_ms,
            )
            session.add(instance=orm_result)
        else:
            orm_result.nvidia_driver_version = result.environment.nvidia_driver_version
            orm_result.torch_version = result.environment.torch_version
            orm_result.transformers_version = result.environment.transformers_version
            orm_result.vllm_version = result.environment.vllm_version
            orm_result.model_modal = result.model_spec.modal
            orm_result.model_series = result.model_spec.series
            orm_result.model_size = result.model_spec.size
            orm_result.model_quant = result.model_spec.quant.value
            orm_result.model_tool_call = result.model_spec.tool_call.value
            orm_result.model_reasoning = result.model_spec.reasoning.value
            orm_result.model_repo_id = result.model_spec.repo_id[
                HubBackend.HUGGING_FACE
            ]
            orm_result.vllm_seed = result.serve_param.seed
            orm_result.vllm_extra_params = result.serve_param.extra_params
            orm_result.vllm_max_cache_tokens = result.serve_status.max_cache_tokens
            orm_result.vllm_max_requests = result.serve_status.max_requests
            orm_result.max_requests = result.benchmark_param.max_requests
            orm_result.request_throughput_min = (
                result.benchmark_metric.request_throughput_min
            )
            orm_result.completion_token_throughput = (
                result.benchmark_metric.completion_token_throughput
            )
            orm_result.total_token_throughput = (
                result.benchmark_metric.total_token_throughput
            )
            orm_result.mean_latency = result.benchmark_metric.mean_latency
            orm_result.mean_ttft_ms = result.benchmark_metric.mean_ttft_ms
            orm_result.mean_tpot_ms = result.benchmark_metric.mean_tpot_ms
        session.commit()
