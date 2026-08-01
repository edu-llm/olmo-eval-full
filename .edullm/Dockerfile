# Research image for the eduLLM platform.
#
# The root Dockerfile is not this file and cannot be. It opens with a defaulted
# FROM and copies a binary out of ghcr.io/astral-sh/uv:latest, a mutable tag, so
# the platform refuses it as unregistered_stage_reference. Everything a published
# layer contains has to be reachable from the pinned base or from a version the
# lockfile names, which is what this file does instead.
#
# The base is nvidia/cuda:12.8.1-runtime-ubuntu24.04, pinned by digest in
# config/repositories.yaml and passed in by the build. It is the platform's second
# registered base and the only reason this repository needs one: the CUDA userspace
# libraries have to match the host driver, so they come from the base rather than
# from anything installed below.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# git is not a convenience here. The ifbench dependency is a git URL, so uv cannot
# resolve the lockfile without it.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3.12 \
        python3.12-venv \
        python3-pip; \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# The same uv the platform's build workflow pins, so the resolver that reads
# uv.lock here is the one the repository's own CI reads it with.
RUN set -eux; \
    python3.12 -m venv /opt/venv; \
    pip install --no-cache-dir "uv==0.11.32"

WORKDIR /opt/olmo-eval

# Dependencies before source, so editing an eval task does not re-resolve the tree.
#
# --no-default-groups drops dev and vllm. The four extras that stay are the ones
# olmo_eval.cli imports unconditionally or the platform needs to get results out:
# beaker for the cli.beaker group, hf for the tokenizer, s3 for writing results,
# clients for the OpenAI-compatible backend. None of them pulls torch.
COPY pyproject.toml uv.lock README.md ./
RUN set -eux; \
    uv sync --frozen --no-default-groups --no-install-project \
        --extra beaker --extra hf --extra s3 --extra clients

COPY . .
RUN set -eux; \
    uv pip install --no-deps .

# What the image claims. Each line is a claim that was cheaper to check here than
# to discover in a Batch job.
#
# The CUDA check is the load-bearing one: it is the whole argument for registering
# a second base, so a build that quietly landed on the python base should fail here
# rather than fail later against a GPU.
RUN set -eux; \
    ldconfig -p | grep -q libcudart; \
    cd /; \
    python -c "from importlib.metadata import version; print(version('olmo-eval'))"; \
    python -c "import importlib.resources as r; f = r.files('olmo_eval.evals.tasks'); names = [e.name for e in f.iterdir() if e.name.endswith('.jsonl')]; assert names, 'task jsonl data did not travel in the wheel; the evals that read it would fail at load'; print(len(names), 'task data files')"; \
    olmo-eval --help > /dev/null

# torch and vllm are not installed. They are a default group in pyproject.toml that
# this image drops, because a torch built against this base is several gigabytes and
# which backend a run needs depends on the run. The repository's own image treats
# them the same way. uv is on PATH inside the venv, so a workload adds one without
# rebuilding, and the CUDA libraries such a wheel links against are already here.
CMD ["olmo-eval", "--help"]
