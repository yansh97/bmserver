import asyncio
import multiprocessing as mp
import multiprocessing.context as mpc
import shlex
import socket
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any

from bmhub.backend import HubBackend
from vllm.entrypoints.cli.main import main as vllm_main

from bmserver.chat.schema import (
    DEFAULT_SERVED_MODEL_NAME,
    Model,
    QuantFormat,
    ReasoningFormat,
    ServeParam,
    ToolCallFormat,
)
from bmserver.chat.utils.client import wait_for_health
from bmserver.schema import Environment

QUANT_CLI_ARGS: dict[QuantFormat, str] = {
    QuantFormat.NONE: "",
    QuantFormat.GPTQ8: " --quantization gptq_marlin",
    QuantFormat.GPTQ4: " --quantization gptq_marlin",
    QuantFormat.AWQ: " --quantization awq_marlin",
}
TOOL_CALL_CLI_ARGS: dict[ToolCallFormat, str] = {
    ToolCallFormat.NONE: "",
    ToolCallFormat.HERMES: " --enable-auto-tool-choice --tool-call-parser hermes",
}
REASONING_CLI_ARGS: dict[ReasoningFormat, str] = {
    ReasoningFormat.NONE: "",
    ReasoningFormat.DEEPSEEK_R1: " --enable-reasoning --reasoning-parser deepseek_r1",
}


def vllm_base_command(
    *, environment: Environment, model: Model, serve_param: ServeParam
) -> str:
    assert model.info is not None
    command: str = f"--model {model.info.path}"
    command += QUANT_CLI_ARGS[model.spec.quant]
    command += " --model-impl vllm"
    command += " --device cuda"
    command += f" --tensor-parallel-size {environment.nvidia_device_count}"
    command += " --task generate"
    command += " --generation-config auto"
    command += " --enable-prefix-caching"
    command += " --enable-chunked-prefill"
    command += f" --seed {serve_param.seed}"
    command += " --compilation-config 3"
    command += f" {serve_param.extra_params}" if serve_param.extra_params else ""
    return command


def vllm_server_command(
    *,
    environment: Environment,
    hub: HubBackend,
    model: Model,
    serve_param: ServeParam,
    host: str,
    port: int,
    api_key: str | None,
    disable_log_requests: bool,
) -> str:
    command: str = vllm_base_command(
        environment=environment, model=model, serve_param=serve_param
    )
    command += TOOL_CALL_CLI_ARGS[model.spec.tool_call]
    command += REASONING_CLI_ARGS[model.spec.reasoning]
    command += f" --host {host} --port {port}"
    command += (
        f" --served-model-name {model.spec.repo_id[hub]} {DEFAULT_SERVED_MODEL_NAME}"
    )
    command += f" --api-key {api_key}" if api_key else ""
    if disable_log_requests:
        command += " --disable-log-requests"
        command += " --disable-uvicorn-access-log"
    command = command.removeprefix("--model ")
    command = f"vllm serve {command}"
    return command


def start_vllm_server(*, command: str) -> None:
    sys.argv = shlex.split(s=command)
    vllm_main()


def find_available_port(*, start_port=8080, end_port=10000) -> int:
    for port in range(start_port, end_port):
        with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as s:
            with suppress(Exception):
                s.bind(("localhost", port))
                return port
    raise RuntimeError("No available port found.")


def wait_for_vllm_server(*, port: int) -> None:
    base_url: str = f"http://127.0.0.1:{port}"
    if asyncio.run(main=wait_for_health(base_url=base_url)):
        return
    raise RuntimeError("Timeout waiting for VLLM server to start.")


@contextmanager
def start_vllm_server_in_process(
    *,
    environment: Environment,
    hub: HubBackend,
    model: Model,
    serve_param: ServeParam,
    disable_log_requests: bool,
) -> Generator[int, None, None]:
    port: int = find_available_port()
    command: str = vllm_server_command(
        environment=environment,
        hub=hub,
        model=model,
        serve_param=serve_param,
        host="127.0.0.1",
        port=port,
        api_key=None,
        disable_log_requests=disable_log_requests,
    )
    ctx: mpc.SpawnContext = mp.get_context(method="spawn")
    worker_kwargs: dict[str, Any] = {
        "command": command,
    }
    worker_process = ctx.Process(target=start_vllm_server, kwargs=worker_kwargs)
    try:
        worker_process.start()
        wait_for_vllm_server(port=port)
        yield port
    finally:
        worker_process.terminate()
        worker_process.join()
