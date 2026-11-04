FROM python:3.11-slim-bookworm

# XGBoost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"

# Dependencies first (cached layer): exactly the versions in uv.lock, without the dev group
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY predict.py pipeline_v1.bin ./

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
