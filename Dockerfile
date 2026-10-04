# ============================================================
# Stage 1: "builder" - install all the Python libraries
# ============================================================
FROM python:3.10-bookworm AS builder

# Bring the uv tool into this stage (copied from uv's official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Work inside the /app folder
WORKDIR /app

# uv settings:
# - compile Python files ahead of time, so the app starts faster
# - copy files instead of linking them (needed inside Docker)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy ONLY the dependency files first (see Question 4)
COPY pyproject.toml uv.lock ./

# Create the virtual environment in /app/.venv with all the libraries
#   --frozen               use exactly the versions in uv.lock, never change them
#   --no-dev               skip development-only libraries
#   --no-install-project   install the libraries only, not our own project package
RUN uv sync --frozen --no-dev --no-install-project


# ============================================================
# Stage 2: "runtime" - a small image that only runs the API
# ============================================================
FROM python:3.10-slim-bookworm AS runtime

WORKDIR /app

# Take ONLY the finished virtual environment from the builder stage
# (uv and the download caches stay behind in stage 1)
COPY --from=builder /app/.venv /app/.venv

# Use the Python and programs from that virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"

# Copy our source code (this changes often, so it comes last)
COPY src/ ./src/

# Document that the API listens on port 8000
EXPOSE 8000

# The command that runs when a container starts from this image
ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]