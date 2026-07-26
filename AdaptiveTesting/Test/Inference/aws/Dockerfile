# Inference sweep image: CUDA + vLLM + Prometheus judge.
# Build:  docker build -f aws/Dockerfile -t adaptive-inference:latest .
# (build context = AdaptiveTesting/Test/Inference)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip git ca-certificates awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install -r /app/requirements.txt

# Ship the sweep code (weights are pulled at runtime, never baked in).
COPY . /app/Inference

# Model weights cache lives on the instance NVMe (mounted at /cache).
VOLUME ["/cache"]

ENTRYPOINT ["/app/Inference/aws/entrypoint.sh"]
