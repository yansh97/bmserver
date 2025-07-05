from itertools import product
from pathlib import Path
from typing import Annotated, Any

import typer
from bmhub.backend import HubBackend
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from bmserver.chat.schema import (
    BenchmarkMetric,
    BenchmarkParam,
    BenchmarkResult,
    Model,
    ServeParam,
    ServeStatus,
)
from bmserver.chat.utils.benchmark import (
    benchmark_server,
    detect_serve_status_in_process,
    generate_benchmark_params,
    upload_benchmark_result,
)
from bmserver.chat.utils.hub import find_model, search_models
from bmserver.chat.utils.server import (
    start_vllm_server,
    start_vllm_server_in_process,
    vllm_server_command,
)
from bmserver.schema import Environment
from bmserver.settings import settings
from bmserver.utils import get_console

app = typer.Typer(no_args_is_help=True, help="Toolkit for chat models")

HUB_ARGUMENT: Any = typer.Argument(help="Model hub name", show_default=False)
NAME_ARGUMENT: Any = typer.Argument(help="Model name", show_default=False)
NAME_PATTERN_ARGUMENT: Any = typer.Argument(
    help="Glob pattern for model names", show_default=False
)
LOCAL_DIR_OPTION: Any = typer.Option(
    "--local-dir",
    help="Local model storage directory",
    exists=True,
    file_okay=False,
    dir_okay=True,
    resolve_path=True,
)
YES_OPTION: Any = typer.Option(
    "--yes", "-y", help="Confirm the action without prompting"
)

SEED_OPTION: Any = typer.Option("--seed", help="Random seed for reproducibility")
EXTRA_SERVE_PARAMS_OPTION: Any = typer.Option(
    "--extra-params", help="Extra parameters for vLLM serve command"
)
HOST_OPTION: Any = typer.Option(
    "--host", help="Host of the OpenAI compatible API service"
)
PORT_OPTION: Any = typer.Option(
    "--port", help="Port of the OpenAI compatible API service"
)
API_KEY_OPTION: Any = typer.Option(
    "--api-key", help="API key for OpenAI compatible API service"
)
CHECK_OPTION: Any = typer.Option(
    "--check", help="Output vLLM serve command without running it"
)

NUM_PROMPT_TOKENS_OPTION: Any = typer.Option(
    "--num-prompt-tokens", help="Number of prompt tokens for each request", min=1
)
NUM_COMPLETION_TOKENS_OPTION: Any = typer.Option(
    "--num-completion-tokens",
    help="Number of completion tokens for each request",
    min=1,
)
UPLOAD_OPTION: Any = typer.Option(
    "--upload", help="Upload the benchmark results to the PostgreSQL database"
)


def console_print_models(
    console: Console, models: list[Model], hub: HubBackend
) -> None:
    table = Table(box=box.SIMPLE)
    table.add_column(header="NAME", no_wrap=True)
    table.add_column(header="MODAL", no_wrap=True)
    table.add_column(header="SERIES", no_wrap=True)
    table.add_column(header="SIZE", justify="right", no_wrap=True)
    table.add_column(header="QUANT", no_wrap=True)
    table.add_column(header="TOOL CALL", no_wrap=True)
    table.add_column(header="REASONING", no_wrap=True)
    table.add_column(header="MODEL ID", no_wrap=True)
    table.add_column(header="SIZE ON DISK", justify="right", no_wrap=True)
    for model in models:
        table.add_row(
            model.spec.name,
            model.spec.modal,
            model.spec.series,
            f"{model.spec.size}B",
            model.spec.quant.value,
            model.spec.tool_call.value,
            model.spec.reasoning.value,
            model.spec.repo_id[hub],
            model.info.size_on_disk_str if model.info is not None else "",
        )
    console.print(table)


def console_print_result(console: Console, benchmark_result: BenchmarkResult) -> None:
    console.print("{s:{c}^{n}}".format(s=" Benchmark Result ", n=50, c="="))
    console.print(benchmark_result)
    console.print("=" * 50)


@app.command()
def search(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    pattern: Annotated[str | None, NAME_PATTERN_ARGUMENT] = None,
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
) -> None:
    """Search supported chat models."""
    models: list[Model] = search_models(hub=hub, pattern=pattern, local_dir=local_dir)
    console: Console = get_console()
    console_print_models(console=console, models=models, hub=hub)


@app.command(name="list")
def list_(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    pattern: Annotated[str | None, NAME_PATTERN_ARGUMENT] = None,
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
) -> None:
    """List downloaded chat models."""
    models: list[Model] = search_models(hub=hub, pattern=pattern, local_dir=local_dir)
    models = [model for model in models if model.info is not None]
    console: Console = get_console()
    console_print_models(console=console, models=models, hub=hub)


@app.command()
def download(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    name: Annotated[str, NAME_ARGUMENT],
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
) -> None:
    """Download a chat model from the model hub."""
    model: Model = find_model(hub=hub, name=name, local_dir=local_dir)
    model_id: str = model.spec.repo_id[hub]
    model_path: Path | None = None if local_dir is None else local_dir / model_id
    model_path_str: str = "cache" if model_path is None else f"<{model_path}>"
    console: Console = get_console()
    console.print(f"Downloading <{model_id}> to {model_path_str}...")
    hub.download_model(model_id=model_id, model_path=model_path)


@app.command()
def update(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    pattern: Annotated[str | None, NAME_PATTERN_ARGUMENT] = None,
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
    yes: Annotated[bool, YES_OPTION] = False,
) -> None:
    """Update downloaded chat models."""
    models: list[Model] = search_models(hub=hub, pattern=pattern, local_dir=local_dir)
    models = [model for model in models if model.info is not None]
    count: int = len(models)
    console: Console = get_console()
    console.print(f"The following {count} chat models will be updated:")
    console_print_models(console=console, models=models, hub=hub)
    console.print(
        escape(markup=f"Do you want to update the {count} chat models? [y/N] "), end=""
    )
    if not yes and not console.input().lower() == "y":
        return
    for i, m in enumerate(iterable=models):
        assert m.info is not None
        id_: str = m.spec.repo_id[hub]
        path: Path = m.info.path
        console.print(
            escape(markup=f"[{i + 1}/{count}] Updating <{id_}> in <{path}>...")
        )
        hub.update_model(model=m.info)


@app.command()
def remove(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    pattern: Annotated[str | None, NAME_PATTERN_ARGUMENT] = None,
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
    yes: Annotated[bool, YES_OPTION] = False,
) -> None:
    """Remove downloaded chat models."""
    models: list[Model] = search_models(hub=hub, pattern=pattern, local_dir=local_dir)
    models = [model for model in models if model.info is not None]
    count: int = len(models)
    console: Console = get_console()
    console.print(f"The following {count} chat models will be removed:")
    console_print_models(console=console, models=models, hub=hub)
    console.print(
        escape(markup=f"Do you want to remove the {count} chat models? [y/N] "), end=""
    )
    if not yes and not console.input().lower() == "y":
        return
    for i, m in enumerate(iterable=models):
        assert m.info is not None
        id_: str = m.spec.repo_id[hub]
        path: Path = m.info.path
        console.print(
            escape(markup=f"[{i + 1}/{count}] Removing <{id_}> in <{path}>...")
        )
        hub.remove_model(model=m.info)


@app.command()
def serve(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    name: Annotated[str, NAME_ARGUMENT],
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
    seed: Annotated[int, SEED_OPTION] = 2025,
    extra_params: Annotated[str | None, EXTRA_SERVE_PARAMS_OPTION] = None,
    host: Annotated[str, HOST_OPTION] = "127.0.0.1",
    port: Annotated[int, PORT_OPTION] = 8080,
    api_key: Annotated[str | None, API_KEY_OPTION] = None,
    check: Annotated[bool, CHECK_OPTION] = False,
) -> None:
    """Serve a chat model and provide OpenAI compatible API service."""
    environment: Environment = Environment.detect()
    model: Model = find_model(hub=hub, name=name, local_dir=local_dir)
    serve_param = ServeParam(seed=seed, extra_params=extra_params)
    model_id: str = model.spec.repo_id[hub]
    if model.info is None:
        raise RuntimeError(f"<{model_id}> is not downloaded.")
    command: str = vllm_server_command(
        environment=environment,
        hub=hub,
        model=model,
        serve_param=serve_param,
        host=host,
        port=port,
        api_key=api_key,
        disable_log_requests=False,
    )
    console: Console = get_console()
    console.print(f"vLLM serve command: {command}")
    if check:
        return
    console.print(f"Serving <{model_id}> on <{host}:{port}>...")
    start_vllm_server(command=command)


@app.command()
def benchmark(
    hub: Annotated[HubBackend, HUB_ARGUMENT],
    name: Annotated[str, NAME_ARGUMENT],
    *,
    local_dir: Annotated[Path | None, LOCAL_DIR_OPTION] = None,
    seed: Annotated[int, SEED_OPTION] = 2025,
    extra_params: Annotated[str | None, EXTRA_SERVE_PARAMS_OPTION] = None,
    num_prompt_tokens: Annotated[list[int], NUM_PROMPT_TOKENS_OPTION] = [500],
    num_completion_tokens: Annotated[list[int], NUM_COMPLETION_TOKENS_OPTION] = [500],
    upload: Annotated[bool, UPLOAD_OPTION] = False,
) -> None:
    """Benchmark a chat model for inference performance."""
    if upload and settings.postgres_url is None:
        raise RuntimeError("Environment variable BMSERVER_POSTGRES_URL is not set.")
    environment: Environment = Environment.detect()
    model: Model = find_model(hub=hub, name=name, local_dir=local_dir)
    serve_param = ServeParam(seed=seed, extra_params=extra_params)
    model_id: str = model.spec.repo_id[hub]
    if model.info is None:
        raise RuntimeError(f"<{model_id}> is not downloaded.")
    console: Console = get_console()
    serve_status: ServeStatus = detect_serve_status_in_process(
        environment=environment, model=model, serve_param=serve_param
    )
    console.print(f"Serving <{model_id}> on <127.0.0.1>...")
    with start_vllm_server_in_process(
        environment=environment,
        hub=hub,
        model=model,
        serve_param=serve_param,
        disable_log_requests=True,
    ) as server_port:
        for prompt_tokens, completion_tokens in product(
            num_prompt_tokens, num_completion_tokens
        ):
            benchmark_params: list[BenchmarkParam] = generate_benchmark_params(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                serve_status=serve_status,
            )
            for benchmark_param in benchmark_params:
                console.print(
                    f"Benchmarking <{model_id}> with {benchmark_param.to_str()}..."
                )
                benchmark_metric: BenchmarkMetric = benchmark_server(
                    benchmark_param=benchmark_param, port=server_port
                )
                benchmark_result: BenchmarkResult = BenchmarkResult(
                    environment=environment,
                    model_spec=model.spec,
                    serve_param=serve_param,
                    serve_status=serve_status,
                    benchmark_param=benchmark_param,
                    benchmark_metric=benchmark_metric,
                )
                console_print_result(console=console, benchmark_result=benchmark_result)
                if upload:
                    upload_benchmark_result(result=benchmark_result)
