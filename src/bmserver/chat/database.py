from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from bmserver.settings import settings


class ORMBase(DeclarativeBase):
    pass


class ORMBenchmarkResult(ORMBase):
    __tablename__: str = "chat_model_benchmark_result"

    # primary key
    nvidia_device_name: Mapped[str] = mapped_column(
        String(length=64), nullable=False, primary_key=True
    )
    nvidia_device_count: Mapped[int] = mapped_column(
        Integer(), nullable=False, primary_key=True
    )
    bmserver_version: Mapped[str] = mapped_column(
        String(length=16), nullable=False, primary_key=True
    )
    model_name: Mapped[str] = mapped_column(
        String(length=64), nullable=False, primary_key=True
    )
    num_prompt_tokens: Mapped[int] = mapped_column(
        Integer(), nullable=False, primary_key=True
    )
    num_completion_tokens: Mapped[int] = mapped_column(
        Integer(), nullable=False, primary_key=True
    )
    num_requests: Mapped[int] = mapped_column(
        Integer(), nullable=False, primary_key=True
    )

    # environment
    nvidia_driver_version: Mapped[str] = mapped_column(
        String(length=16), nullable=False
    )
    torch_version: Mapped[str] = mapped_column(String(length=16), nullable=False)
    transformers_version: Mapped[str] = mapped_column(String(length=16), nullable=False)
    vllm_version: Mapped[str] = mapped_column(String(length=16), nullable=False)

    # model
    model_modal: Mapped[str] = mapped_column(String(length=16), nullable=False)
    model_series: Mapped[str] = mapped_column(String(length=32), nullable=False)
    model_size: Mapped[float] = mapped_column(Float(), nullable=False)
    model_quant: Mapped[str] = mapped_column(String(length=16), nullable=False)
    model_tool_call: Mapped[str] = mapped_column(String(length=16), nullable=False)
    model_reasoning: Mapped[str] = mapped_column(String(length=16), nullable=False)
    model_repo_id: Mapped[str] = mapped_column(String(length=64), nullable=False)

    # serve param
    vllm_seed: Mapped[int] = mapped_column(Integer(), nullable=False)
    vllm_extra_params: Mapped[str | None] = mapped_column(
        String(length=256), nullable=True
    )

    # serve status
    vllm_max_cache_tokens: Mapped[int] = mapped_column(Integer(), nullable=False)
    vllm_max_requests: Mapped[int] = mapped_column(Integer(), nullable=False)

    # benchmark param
    max_requests: Mapped[int] = mapped_column(Integer(), nullable=False)

    # benchmark metric
    request_throughput_min: Mapped[float] = mapped_column(Float(), nullable=False)
    completion_token_throughput: Mapped[float] = mapped_column(Float(), nullable=False)
    total_token_throughput: Mapped[float] = mapped_column(Float(), nullable=False)
    mean_latency: Mapped[float] = mapped_column(Float(), nullable=False)
    mean_ttft_ms: Mapped[float] = mapped_column(Float(), nullable=False)
    mean_tpot_ms: Mapped[float] = mapped_column(Float(), nullable=False)


engine: Engine | None = None
if settings.postgres_url is not None:
    engine = create_engine(url=settings.postgres_url)
    ORMBase.metadata.create_all(bind=engine)


@contextmanager
def database_session() -> Iterator[Session]:
    assert engine is not None
    with Session(bind=engine) as session:
        yield session
