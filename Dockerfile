FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV PYTHONPATH=/app/src

COPY ./.python-version /app/.python-version
COPY ./uv.lock /app/uv.lock
COPY ./pyproject.toml /app/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --only-group web

COPY ./src /app/src
COPY ./web.py /app/web.py

ENTRYPOINT []

CMD ["uv", "run", "--only-group", "web", "streamlit", "run", "web.py"]
