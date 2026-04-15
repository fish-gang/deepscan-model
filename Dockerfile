FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y python3 python3-venv && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync

COPY . .

ENTRYPOINT ["uv", "run", "python", "main.py"]