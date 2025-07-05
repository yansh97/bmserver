from enum import Enum
from pathlib import Path

from bmhub.backend import HubBackend
from bmhub.schema import ModelInfo
from pydantic import BaseModel

from bmserver.schema import Environment

MODELS_JSONL: Path = Path(__file__).parent / "models.jsonl"

DEFAULT_SERVED_MODEL_NAME = "DEFAULT"

MAX_CONCURRENT_REQUESTS: int = 256

SERVER_START_TIMEOUT: int = 10 * 60
SERVER_HTTP_TIMEOUT: int = 6 * 60 * 60

# 常用100汉字
WORDS: str = (
    "的一了是我不在人们有来他这上着个地到大里说就去子得"
    "也和那要下看天时过出小么起你都把好还多没为又可家学"
    "只以主会样年想能生同老中十从自面前头道它后然走很像"
    "见两用她国动进成回什边作对开而己些现山民候经发工向"
)


class QuantFormat(Enum):
    NONE = "none"
    GPTQ8 = "gptq8"
    GPTQ4 = "gptq4"
    AWQ = "awq"


class ToolCallFormat(Enum):
    NONE = "none"
    HERMES = "hermes"


class ReasoningFormat(Enum):
    NONE = "none"
    DEEPSEEK_R1 = "deepseek-r1"


class ModelSpec(BaseModel):
    name: str
    modal: str
    series: str
    size: float
    quant: QuantFormat
    tool_call: ToolCallFormat
    reasoning: ReasoningFormat
    repo_id: dict[HubBackend, str]


class Model(BaseModel):
    spec: ModelSpec
    info: ModelInfo | None


class ServeParam(BaseModel):
    seed: int
    extra_params: str | None


class ServeStatus(BaseModel):
    max_cache_tokens: int
    max_requests: int


class BenchmarkParam(BaseModel):
    num_prompt_tokens: int
    num_completion_tokens: int
    max_requests: int
    num_requests: int

    def to_str(self) -> str:
        return f"{self.num_requests} requests ({self.num_prompt_tokens} tokens -> {self.num_completion_tokens} tokens)"  # noqa: E501


class RequestParam(BaseModel):
    messages: list[dict]
    num_completion_tokens: int


class RequestMetric(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    latency: float
    ttft_ms: float
    tpot_ms: float


class BenchmarkMetric(BaseModel):
    request_throughput_min: float
    completion_token_throughput: float
    total_token_throughput: float
    mean_latency: float
    mean_ttft_ms: float
    mean_tpot_ms: float


class BenchmarkResult(BaseModel):
    environment: Environment
    model_spec: ModelSpec
    serve_param: ServeParam
    serve_status: ServeStatus
    benchmark_param: BenchmarkParam
    benchmark_metric: BenchmarkMetric
