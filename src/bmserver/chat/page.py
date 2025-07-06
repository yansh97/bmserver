import math

import pandas as pd
import streamlit as st
from sqlalchemy import ColumnElement, Select, func, or_, select
from streamlit.delta_generator import DeltaGenerator

from bmserver.chat.database import ORMBenchmarkResult, database_session
from bmserver.settings import settings

if settings.postgres_url is None:
    st.error(body="请先设置数据库连接环境变量：BMSERVER_POSTGRES_URL")
    exit()


def get_query_where_var() -> list[ColumnElement]:
    if "query_where" not in st.session_state:
        st.session_state.query_where = []
    return st.session_state.query_where


def append_query_where_var(where: ColumnElement) -> None:
    st.session_state.query_where.append(where)


def check_num_requests(num_requests: int) -> bool:
    while True:
        if num_requests == 1:
            return True
        if num_requests % 2 != 0:
            return False
        num_requests = num_requests // 2


def get_bmserver_version_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.bmserver_version.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.bmserver_version)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_modal_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.model_modal.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_modal)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_tool_call_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.model_tool_call.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_tool_call)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_reasoning_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.model_reasoning.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_reasoning)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_series_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.model_series.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_series)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_size_options(where: list[ColumnElement]) -> list[float]:
    with database_session() as session:
        stmt: Select[tuple[float]] = (
            select(ORMBenchmarkResult.model_size.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_size)
        )
        return list(session.execute(statement=stmt).scalars())


def get_model_quant_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.model_quant.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.model_quant)
        )
        return list(session.execute(statement=stmt).scalars())


def get_nvidia_device_name_options(where: list[ColumnElement]) -> list[str]:
    with database_session() as session:
        stmt: Select[tuple[str]] = (
            select(ORMBenchmarkResult.nvidia_device_name.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.nvidia_device_name)
        )
        return list(session.execute(statement=stmt).scalars())


def get_nvidia_device_count_options(where: list[ColumnElement]) -> list[int]:
    with database_session() as session:
        stmt: Select[tuple[int]] = (
            select(ORMBenchmarkResult.nvidia_device_count.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.nvidia_device_count)
        )
        return list(session.execute(statement=stmt).scalars())


def get_num_prompt_tokens_options(where: list[ColumnElement]) -> list[int]:
    with database_session() as session:
        stmt: Select[tuple[int]] = (
            select(ORMBenchmarkResult.num_prompt_tokens.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.num_prompt_tokens)
        )
        return list(session.execute(statement=stmt).scalars())


def get_num_completion_tokens_options(where: list[ColumnElement]) -> list[int]:
    with database_session() as session:
        stmt: Select[tuple[int]] = (
            select(ORMBenchmarkResult.num_completion_tokens.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.num_completion_tokens)
        )
        return list(session.execute(statement=stmt).scalars())


def get_num_requests_options(where: list[ColumnElement]) -> list[int | str]:
    with database_session() as session:
        stmt: Select[tuple[int]] = (
            select(ORMBenchmarkResult.num_requests.distinct())
            .where(*where)
            .order_by(ORMBenchmarkResult.num_requests)
        )
        result: list[int | str] = [
            num_requests
            for num_requests in session.execute(statement=stmt).scalars()
            if check_num_requests(num_requests=num_requests)
        ]
        return ["MAX"] + result


def count_benchmark_results(where: list[ColumnElement]) -> int:
    with database_session() as session:
        stmt: Select[tuple[int]] = select(
            func.count(ORMBenchmarkResult.bmserver_version)
        ).where(*where)
        return session.execute(statement=stmt).scalar_one()


def get_benchmark_results(
    where: list[ColumnElement], offset: int, limit: int
) -> pd.DataFrame:
    with database_session() as session:
        benchmark_results_stmt: Select[tuple[ORMBenchmarkResult]] = (
            select(ORMBenchmarkResult)
            .where(*where)
            .offset(offset=offset)
            .limit(limit=limit)
            .order_by(
                ORMBenchmarkResult.bmserver_version,
                ORMBenchmarkResult.model_modal,
                ORMBenchmarkResult.model_tool_call,
                ORMBenchmarkResult.model_reasoning,
                ORMBenchmarkResult.model_series,
                ORMBenchmarkResult.model_size,
                ORMBenchmarkResult.model_quant,
                ORMBenchmarkResult.nvidia_device_name,
                ORMBenchmarkResult.nvidia_device_count,
                ORMBenchmarkResult.num_prompt_tokens,
                ORMBenchmarkResult.num_completion_tokens,
                ORMBenchmarkResult.num_requests,
            )
        )
        benchmark_results: list[ORMBenchmarkResult] = list(
            session.execute(statement=benchmark_results_stmt).scalars()
        )
        columns: list[str] = [
            column.name for column in ORMBenchmarkResult.__table__.columns
        ]
        records: list[dict] = [
            {column: getattr(benchmark_result, column) for column in columns}
            for benchmark_result in benchmark_results
        ]
        frame: pd.DataFrame = pd.DataFrame.from_records(data=records)
        return frame.set_index(
            keys=[
                "bmserver_version",
                "model_name",
                "nvidia_device_name",
                "nvidia_device_count",
                "num_prompt_tokens",
                "num_completion_tokens",
                "num_requests",
            ]
        )


column1: DeltaGenerator
column2: DeltaGenerator
column3: DeltaGenerator

st.title(body="对话（Chat）大模型推理性能测试数据库")

st.divider()

select_bmserver_version_values: list[str] = st.segmented_control(
    label="选择 BMServer 版本",
    key="select_bmserver_version_values",
    options=get_bmserver_version_options(where=get_query_where_var()),
    selection_mode="multi",
)
if select_bmserver_version_values:
    append_query_where_var(
        where=ORMBenchmarkResult.bmserver_version.in_(
            other=select_bmserver_version_values
        )
    )

column1, column2, column3 = st.columns(spec=3)

with column1:
    select_model_modal_values: list[str] = st.segmented_control(
        label="选择模态",
        key="select_model_modal_values",
        options=get_model_modal_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_model_modal_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_modal.in_(other=select_model_modal_values)
        )

with column2:
    select_model_tool_call_values: list[str] = st.segmented_control(
        label="选择工具调用模式",
        key="select_model_tool_call_values",
        options=get_model_tool_call_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_model_tool_call_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_tool_call.in_(
                other=select_model_tool_call_values
            )
        )

with column3:
    select_model_reasoning_values: list[str] = st.segmented_control(
        label="选择推理模式",
        key="select_model_reasoning_values",
        options=get_model_reasoning_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_model_reasoning_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_reasoning.in_(
                other=select_model_reasoning_values
            )
        )

column1, column2, column3 = st.columns(spec=3)

with column1:
    select_model_series_values: list[str] = st.segmented_control(
        label="选择模型系列",
        key="select_model_series_values",
        options=get_model_series_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_model_series_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_series.in_(other=select_model_series_values)
        )

with column2:
    select_model_size_values: list[float] = st.segmented_control(
        label="选择模型大小",
        key="select_model_size_values",
        options=get_model_size_options(where=get_query_where_var()),
        format_func=lambda x: f"{int(x) if x.is_integer() else x}B",
        selection_mode="multi",
    )
    if select_model_size_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_size.in_(other=select_model_size_values)
        )

with column3:
    select_model_quant_values: list[str] = st.segmented_control(
        label="选择模型量化",
        key="select_model_quant_values",
        options=get_model_quant_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_model_quant_values:
        append_query_where_var(
            where=ORMBenchmarkResult.model_quant.in_(other=select_model_quant_values)
        )

column1, column2 = st.columns(spec=2)

with column1:
    select_device_name_values: list[str] = st.segmented_control(
        label="选择 NVIDIA 显卡名称",
        key="select_device_name_values",
        options=get_nvidia_device_name_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_device_name_values:
        append_query_where_var(
            where=ORMBenchmarkResult.nvidia_device_name.in_(
                other=select_device_name_values
            )
        )

with column2:
    select_device_count_values: list[int] = st.segmented_control(
        label="选择 NVIDIA 显卡数量",
        key="select_device_count_values",
        options=get_nvidia_device_count_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_device_count_values:
        append_query_where_var(
            where=ORMBenchmarkResult.nvidia_device_count.in_(
                other=select_device_count_values
            )
        )

column1, column2 = st.columns(spec=2)

with column1:
    select_num_prompt_tokens_values: list[int] = st.segmented_control(
        label="选择输入 Token 数",
        key="select_num_prompt_tokens_values",
        options=get_num_prompt_tokens_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_num_prompt_tokens_values:
        append_query_where_var(
            where=ORMBenchmarkResult.num_prompt_tokens.in_(
                other=select_num_prompt_tokens_values
            )
        )

with column2:
    select_num_completion_tokens_values: list[int] = st.segmented_control(
        label="选择输出 Token 数",
        key="select_num_completion_tokens_values",
        options=get_num_completion_tokens_options(where=get_query_where_var()),
        selection_mode="multi",
    )
    if select_num_completion_tokens_values:
        append_query_where_var(
            where=ORMBenchmarkResult.num_completion_tokens.in_(
                other=select_num_completion_tokens_values
            )
        )

select_num_requests_values: list[int | str] = st.segmented_control(
    label="选择并发数",
    key="select_num_requests_values",
    options=get_num_requests_options(where=get_query_where_var()),
    format_func=lambda x: "最大并发" if x == "MAX" else f"{x} 并发",
    selection_mode="multi",
)
query_num_requests_where: list[ColumnElement] = []
if "MAX" in select_num_requests_values:
    query_num_requests_where.append(
        ORMBenchmarkResult.num_requests == ORMBenchmarkResult.max_requests
    )
select_num_requests_values = [
    num_requests
    for num_requests in select_num_requests_values
    if isinstance(num_requests, int)
]
if select_num_requests_values:
    query_num_requests_where.append(
        ORMBenchmarkResult.num_requests.in_(other=select_num_requests_values)
    )
if query_num_requests_where:
    append_query_where_var(where=or_(*query_num_requests_where))

st.divider()

num_results: int = count_benchmark_results(where=get_query_where_var())

if num_results == 0:
    st.warning(body="没有查询到符合条件的结果")
    st.stop()

page_size: int | None = st.segmented_control(
    label="选择每页显示条数",
    key="page_size",
    options=[20, 50, 100],
    default=20,
    selection_mode="single",
)
if page_size is None:
    st.stop()

num_pages = int(math.ceil(num_results / page_size))
st.markdown(body=f"共查询到 {num_results} 条结果，分 {num_pages} 页显示")

if num_pages > 1:
    query_page_num: int = st.number_input(
        label="选择页码",
        key="query_page_num",
        min_value=1,
        max_value=num_pages,
        value=1,
    )
    offset: int = (query_page_num - 1) * page_size
else:
    offset = 0
benchmark_results: pd.DataFrame = get_benchmark_results(
    where=get_query_where_var(), offset=offset, limit=page_size
)
st.dataframe(
    data=benchmark_results,
    height=35 * len(benchmark_results) + 38,
    use_container_width=True,
    hide_index=False,
    column_config={
        "bmserver_version": "BMServer 版本",
        "model_name": "模型",
        "nvidia_device_name": "显卡名称",
        "nvidia_device_count": "显卡数",
        "num_prompt_tokens": "输入 Token 数",
        "num_completion_tokens": "输出 Token 数",
        "num_requests": "并发数",
        "nvidia_driver_version": "驱动版本",
        "torch_version": "PyTorch 版本",
        "transformers_version": "Transformers 版本",
        "vllm_version": "vLLM 版本",
        "model_modal": "模态",
        "model_series": "系列",
        "model_size": "参数量(B)",
        "model_quant": "量化",
        "model_tool_call": "工具调用",
        "model_reasoning": "推理模式",
        "model_repo_id": "HF Repo",
        "vllm_seed": "vLLM 种子",
        "vllm_extra_params": "vLLM 参数",
        "vllm_max_cache_tokens": "Token 容量",
        "vllm_max_requests": None,
        "max_requests": "最大并发数",
        "request_throughput_min": "请求吞吐量(min)",
        "completion_token_throughput": "输出吞吐量(s)",
        "total_token_throughput": "总吞吐量(s)",
        "mean_latency": "请求延迟(s)",
        "mean_ttft_ms": "请求TTFT(ms)",
        "mean_tpot_ms": "请求TPOT(ms)",
    },
)
