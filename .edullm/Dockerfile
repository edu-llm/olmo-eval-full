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

# Whether this image can hold a model itself, which is one line and is somebody's
# decision rather than a default. 0 is what every build has produced so far: no torch,
# so `mock` and `litellm` run and the three providers that load weights do not. 1 adds
# the `vllm` group, which is what `vllm`, `vllm_server` and `huggingface` need.
#
# The platform's build workflow passes BASE_IMAGE and nothing else, so this cannot be
# set per build. Flipping the default here is the whole change, and it is deliberately
# a one-line diff on the default branch rather than a flag somebody has to find.
#
# WHAT THE 1 COSTS, measured against this lockfile on 2026-08-06 for cp312 linux
# x86_64 rather than estimated. 99 packages and 138 MB of wheels become 235 packages
# and 4.54 GiB. Two thirds of that is six files: torch at 873 MiB and the CUDA
# libraries torch links, cudnn 674, cublas 567, nccl 307, cusparse 275 and cusparselt
# 274. None of it is reducible while a model runs in this process, because that is the
# CUDA userspace the wheel links against.
#
# THE ALTERNATIVE IS NOT A SMALLER IMAGE, IT IS A DIFFERENT PLACE TO PAY. uv is on
# PATH inside the venv, so a workload can install the group at the top of a run
# instead. That moves the same 4.54 GiB off the image and onto every single run, on a
# GPU that is being billed while it downloads, and it puts a resolution nothing
# reviewed inside a job that is meant to be reproducible. Paying once per image is the
# cheaper half of that trade for anything run more than a handful of times.
#
# --extra vllm RATHER THAN --group vllm, AND THE DIFFERENCE IS NOT COSMETIC. The group
# is a one-line mirror of the extra, `vllm = ["olmo-eval[vllm]"]`, so it reaches vLLM
# only through the project referring to itself. This layer passes --no-install-project,
# the project is therefore not in the environment, and that self-reference resolves to
# nothing. uv also carries `conflicts` for this extra, so the vllm edge in uv.lock is
# guarded by `extra == 'extra-9-olmo-eval-vllm'`, which only an --extra turns on.
#
# ASKING FOR THE GROUP HERE FAILS IN THE WORST AVAILABLE WAY, WHICH IS QUIETLY AND
# EXPENSIVELY. Measured on run 31078390731: it installed 24 packages, among them torch
# and every CUDA library, because `torch-c-dlpack-ext` and `opencv-python-headless` are
# plain members of the extra and pull them. vLLM itself was not among the 24. That is a
# multi-gigabyte image that looks like it can self-host and cannot, and nothing but the
# import check below would have said so.
#
# DEFAULTS TO 0, WHICH REPRODUCES THE IMAGE THAT IS REGISTERED TODAY. Whether this
# account self-hosts vLLM or evaluates against a served endpoint is not a question a
# Dockerfile should answer, so this leaves it open and makes the answer one character.
# Set it to 1 and the image can serve a model itself. Built at 1 on run 31078822781,
# which published sha256:efc0556f38845d0198a79ab33362c3a6f6e1b372f820e9e65464a7336dba51b0
# with torch 2.10.0+cu128 and vLLM importable, so the 1 path is a thing that has run
# rather than a thing that should work.
#
# SET TO 1 ON THE SteeringVectors BRANCH. The steering-vector job needs torch, and
# with the default 0 it materialized ~4.5 GiB of torch + CUDA wheels at the top of
# every run, on the billed GPU. Baking it here pays that once per image instead. The
# run command still runs `uv sync --extra vllm ...` so it must keep that extra: uv
# sync is exact and would otherwise remove the torch/vLLM this now provides.
ARG INSTALL_TORCH_AND_VLLM=1
RUN set -eux; \
    if [ "${INSTALL_TORCH_AND_VLLM}" != "1" ]; then exit 0; fi; \
    uv sync --frozen --no-default-groups --no-install-project --no-cache \
        --extra beaker --extra hf --extra s3 --extra clients \
        --extra vllm; \
    uv cache clean

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

# Which providers this image can actually run, asserted rather than described, and
# asserted the same way whichever value INSTALL_TORCH_AND_VLLM took. A comment saying
# what an image carries is a claim nothing checks, and this file has already been
# read once as evidence that torch was here.
#
# `mock` and `litellm` are always constructible, so they are checked unconditionally
# and the build at 1 is held to the same floor the build at 0 is.
#
# WHY THE 1 CHECKS AN IMPORT AND NOT A DISTRIBUTION. `importlib.metadata.version`
# answers from a `.dist-info` directory and proves only that an installer wrote one.
# torch is a C extension linked against the CUDA userspace in the base, and the way
# it fails is a wheel that unpacks and then cannot resolve a symbol at load, so the
# import is the first thing that would notice. `torch.version.cuda` separates the
# CUDA build from the CPU-only build of the same version number, which is otherwise
# a difference nothing here would see until a run asked for a device.
#
# vLLM is imported rather than merely counted for the same reason, and it is the
# import a GPU-less builder is most likely to refuse. It does not refuse: the
# platform layer resolves to a CPU platform with a warning, and `LLM` is bound.
# Nothing below constructs one, because that needs a device this runner has not got.
RUN set -eux; \
    cd /; \
    python -c "from olmo_eval.inference import create_provider; print('constructed', type(create_provider('mock', 'image-smoke')).__name__)"; \
    python -c "from olmo_eval.inference.providers import LiteLLMProvider; print('importable', LiteLLMProvider.__name__)"; \
    if [ "${INSTALL_TORCH_AND_VLLM}" != "1" ]; then exit 0; fi; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda, 'this is the CPU build of torch, so nothing here can reach a card'"; \
    python -c "from vllm import LLM; print('vllm imports, LLM bound:', LLM.__name__)"; \
    python -c "from olmo_eval.inference.providers import HuggingFaceProvider, VLLMProvider, VLLMServerProvider; print('importable', HuggingFaceProvider.__name__, VLLMProvider.__name__, VLLMServerProvider.__name__)"

# What runs in this image, and it depends on how it was built. At
# INSTALL_TORCH_AND_VLLM=0 there is no torch, so `mock` and `litellm` run and the
# three providers that load weights onto a card do not. At 1 all six run. Read the
# ARG above to see which one this image was built as.
CMD ["olmo-eval", "--help"]
