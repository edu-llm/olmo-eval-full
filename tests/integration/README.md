# Integration Tests

This directory contains integration tests that require external services like Docker, GPUs, or network access.

## Prerequisites

### For vLLM Provider Tests

1. **Docker with GPU support** (recommended):
   ```bash
   # Install nvidia-container-toolkit
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

   # Verify GPU access in Docker
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

2. **OR vLLM installed locally** with GPU:
   ```bash
   pip install vllm
   ```

## Running Integration Tests

`--gpu` is the flag that enables these tests, and it is not optional. Every test in
`test_vllm_provider.py` is marked `gpu`, and `tests/conftest.py` skips gpu-marked tests
unless `--gpu` is passed — so without it the file reports 19 skips and no failures, which
is easy to misread as a pass.

There is no `--integration` flag. `integration` is a marker name in `pyproject.toml`, and
passing it as an option makes pytest exit with `unrecognized arguments: --integration`.
Integration tests that need Docker services run by default; `--no-docker` turns them off.

### Quick Start (with Docker)

```bash
# Run all vLLM integration tests
pytest tests/integration/test_vllm_provider.py -v --gpu

# The test harness will automatically:
# 1. Start a vLLM Docker container with a small model (Qwen2-0.5B)
# 2. Wait for the model to load (~2-5 minutes)
# 3. Run the tests
# 4. Stop the container
```

### Using a Pre-running vLLM Instance

If you already have vLLM running (locally or in Docker):

```bash
# Skip Docker management
pytest tests/integration/test_vllm_provider.py -v --gpu --no-docker
```

### Using a Different Model

```bash
# Use a different model (must be compatible with vLLM)
pytest tests/integration/test_vllm_provider.py -v --gpu \
    --vllm-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Manual Docker Setup

If you prefer to manage Docker manually:

```bash
# Start vLLM container
docker compose -f tests/integration/docker-compose.vllm.yml up -d vllm

# Wait for health check to pass
docker compose -f tests/integration/docker-compose.vllm.yml ps

# Run tests (skip Docker management)
pytest tests/integration/test_vllm_provider.py -v --gpu --no-docker

# Stop container when done
docker compose -f tests/integration/docker-compose.vllm.yml down
```

### CPU-Only Testing (Experimental)

For environments without GPU:

```bash
# Start CPU-only vLLM (very slow, for basic validation only)
docker compose -f tests/integration/docker-compose.vllm.yml --profile cpu up -d vllm-cpu

# Note: CPU inference is extremely slow and may timeout
```

## Test Structure

```
tests/integration/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures and configuration
├── docker-compose.vllm.yml  # Docker Compose for vLLM
├── README.md                # This file
└── test_vllm_provider.py     # vLLM provider integration tests
```

## Test Categories

### `TestVLLMProviderGenerate`
Tests for text generation:
- Single/multiple prompts
- Sampling parameters (temperature, top_p, etc.)
- Stop sequences
- Multiple samples
- Logprobs during generation
- Deterministic generation

### `TestVLLMProviderLogprobs`
Tests for multiple-choice scoring via logprobs:
- Single/multiple requests
- Continuation scoring
- Correct answer detection

### `TestVLLMProviderEdgeCases`
Edge case handling:
- Empty prompts
- Long prompts
- Special characters
- Empty continuations

### `TestVLLMProviderWithTasks`
End-to-end task integration:
- ARC-style multiple choice
- Batch processing

## Troubleshooting

### Container fails to start
```bash
# Check logs
docker logs olmo-eval-vllm-test

# Common issues:
# - Not enough GPU memory: Try a smaller model
# - Missing CUDA drivers: Install nvidia-container-toolkit
```

### Tests timeout
```bash
# Increase timeout (default 5 minutes for model loading)
# Edit VLLM_STARTUP_TIMEOUT in conftest.py
```

### Out of GPU memory
```bash
# Use a smaller model
pytest tests/integration/ -v --gpu --vllm-model "Qwen/Qwen2-0.5B"

# Or reduce GPU memory utilization in docker-compose.yml
```

## CI does not run these, and cannot

This section used to carry a GitHub Actions example on `runs-on: [self-hosted, gpu]`, and
two jobs in `.github/workflows/ci.yml` were copied from it. Both have been deleted. edu-llm
has no self-hosted runners and no access to GitHub's larger-runner product, so a job asking
for those labels queues as a pending check for 24 hours and is then cancelled — which is
worse than no check, because a pending dot reads as one about to pass.

Run the tests above yourself on a GPU box. For GPU work that belongs in the account rather
than on your laptop, submit it to AWS Batch with `edullm submit`, which runs against the
image `.edullm/Dockerfile` builds. `AGENTS.md` has that path.
