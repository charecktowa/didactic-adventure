# Development and training environment: the same Linux, Python and dependencies everywhere.
FROM python:3.13-slim

# uv, pinned to the version the lock file was produced with.
COPY --from=astral/uv:0.8.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first, in their own layer: code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group mlops --no-install-project

# Then the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group mlops

# Run as an unprivileged user. ZenML's config folder exists up front so the named volume
# mounted there inherits this user as owner instead of root.
RUN useradd --create-home app \
    && mkdir -p /home/app/.config/zenml \
    && chown -R app /app /home/app
USER app

CMD ["python", "-m", "pipelines.training", "--model", "logistic_regression"]
