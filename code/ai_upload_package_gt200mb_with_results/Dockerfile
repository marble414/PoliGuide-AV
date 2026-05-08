FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-optional.txt pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
COPY checkpoints ./checkpoints

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

CMD ["bash"]
