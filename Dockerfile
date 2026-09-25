# Stage 1 — builder: resolve and build the venv from the lockfile only.
# Only pyproject.toml/uv.lock are copied here so this layer is cached
# and skipped on rebuilds unless a dependency actually changes.
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The project itself (src/) is copied and installed in a second, cheap step:
# this layer only reruns when source changes, not the dependency-heavy one above.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Stage 2 — runtime: slim image with just the venv + source, no build tools.
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
