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

# THE CHECKSUM A CHECKPOINT WRITE ASKS FOR IS NOT IN botocore, AND THE s3 EXTRA DOES
# NOT BRING IT. The eduLLM platform commits a run's output by putting the payload
# with `ChecksumAlgorithm="CRC32C"` and then a `_SUCCESS` object carrying the digest
# S3 attested back; its reader compares the two and calls what it cannot verify
# corrupt. botocore has no pure-Python CRC-32C -- `botocore.httpchecksum` adds the
# algorithm to its supported set only when `awscrt` imports -- and boto3 depends on
# botocore without extras, so nothing here ever brought it in. OLMo-core's image
# found this at the end rather than the start: run run_019fcf53 on 2026-08-05
# trained twenty steps on a T4, logged every one, and died on the first line after
# the loop with `MissingDependencyException: Using CRC32C requires an additional
# dependency`. Every image the platform builds writes through the same protocol.
#
# NOT DECLARED IN pyproject.toml's s3 EXTRA, WHICH IS WHERE IT BELONGS AND WHERE IT
# COSTS TOO MUCH. Adding `botocore[crt]` beside `boto3` there and re-locking with the
# uv this file pins resolves cleanly and adds exactly one package -- and it also
# moves this image's own dependencies, because a new edge changes how uv derives the
# conflicting-extra markers the vllm/openhands groups need. Measured against the four
# extras synced above on 2026-08-05: numpy 2.4.4 -> 2.2.6, rich 14.3.4 -> 13.9.4,
# textual 8.2.4 -> 6.2.1, and pandas split by Python version. Downgrading the harness
# to install a checksum is the wrong trade, so the lockfile is left alone and the
# extra is added over the top of the resolution it already made.
#
# THE PIN IS DERIVED FROM THE INSTALLED botocore RATHER THAN WRITTEN DOWN, which is
# the whole reason this is one line and not two. Naming a version here would be a
# second copy of what uv.lock already pins, able to go stale against it in silence;
# asking for the extra of exactly the botocore that `uv sync` put here cannot. It
# also cannot move botocore, which is the failure a separate install usually brings
# -- verified on 2026-08-05 against botocore 1.42.95, where it installed awscrt
# 0.31.2 and nothing else, the same version the lockfile route resolved to.
#
# TWO ASSERTIONS, BECAUSE THERE ARE TWO WAYS TO GET THIS WRONG. The install
# succeeding proves a distribution is on disk; it does not prove botocore reached it.
# botocore imports awscrt inside a try/except and sets HAS_CRT False on any failure
# -- an incompatible version, a wheel built for another libc -- so an image can carry
# awscrt and refuse CRC32C anyway, silently, until a run has already been paid for.
# The first is the exact call that died, answered by Stubber so it needs no network,
# no credentials and no bucket, while botocore still resolves the checksum before the
# stub is consulted -- which is precisely the frame the failed run's traceback names.
# The second is awscrt's own answer to CRC-32C of b"123456789", fixed by the standard
# at 0xE3069283, which is the difference between "botocore will use this" and "and
# the number it gets is right". Both were run against botocore 1.42.95 with and
# without the CRT on 2026-08-05: they raise the production exception without it and
# pass with it, so neither is vacuous.
RUN set -eux; \
    uv pip install "botocore[crt]==$(python -c 'import botocore; print(botocore.__version__)')"; \
    cd /; \
    python -c "import boto3; from botocore import UNSIGNED; from botocore.config import Config; from botocore.stub import Stubber; client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED)); stub = Stubber(client); stub.add_response('put_object', {}); stub.activate(); client.put_object(Bucket='b', Key='k', Body=b'123456789', ChecksumAlgorithm='CRC32C')"; \
    python -c "from awscrt.checksums import crc32c; assert crc32c(b'123456789') == 0xE3069283, 'awscrt computes something other than CRC-32C'"

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
