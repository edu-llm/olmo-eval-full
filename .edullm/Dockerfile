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

# Which backend this image carries, which is one line and is somebody's decision rather
# than a default. `none` is what every published build has produced so far: no torch, so
# `mock` and `litellm` run and the four providers that load weights do not. `vllm` is what
# `vllm` and `vllm_server` need. `olmo-core` is what `olmo_core` needs, and it is the only
# position of the three that can read a sharded checkpoint OLMo-core wrote. `hf` needs
# torch and transformers and nothing else, so it comes with either loaded position.
#
# The platform's build workflow passes BASE_IMAGE and nothing else, so this cannot be set
# per build. Changing the value here is the whole change, and it is deliberately a one-line
# diff on the branch being built rather than a flag somebody has to find.
#
# THE VALUE IS THE NAME OF THE EXTRA, WHICH IS WHY NOTHING HERE MAPS ONE TO THE OTHER. A
# table would be a second copy of pyproject.toml's extra names, free to go stale against
# them in silence; passing the value straight to --extra cannot. `none` is the only value
# that is not an extra, and it is the one that installs nothing.
#
# AN UNRECOGNISED VALUE FAILS THE BUILD RATHER THAN FALLING BACK, which is the one thing
# this step does that the `!= 1` test it replaces could not. That test read every value
# except exactly `1` as "install nothing", so `INSTALL_TORCH_AND_VLLM=true` produced a
# torch-less image that looked like it had been asked for torch, and nothing would have
# said so until a provider failed to import inside a job that had already been paid for.
#
# WHAT EACH POSITION COSTS, measured against this lockfile for cp312 linux x86_64 rather
# than estimated, by summing the size PyPI reports for the wheel a cp312 manylinux x86_64
# interpreter selects, over the entries whose markers hold there. The three commands that
# produce these sets are at the foot of this comment.
#
#     none          99 packages      139 MiB
#     vllm         235 packages     4.92 GiB
#     olmo-core    138 packages     3.98 GiB
#
# Most of both loaded positions is the same six files, and none of it is reducible while a
# model runs in this process, because that is the CUDA userspace the torch wheel links
# against: torch at 873 MiB, cudnn 674, cublas 567, nccl 307, cusparse 275 and cusparselt
# 274. What `olmo-core` saves against `vllm` is 97 packages and 0.94 GiB, all of it vLLM's
# own engine and the kernels it ships, vllm at 413 MiB and flashinfer-cubin at 255 the
# largest two. So the position that brings a checkpoint loader is also the cheaper one.
#
# THE 4.54 GiB THIS PARAGRAPH USED TO RECORD FOR THE LOADED POSITION DOES NOT REPRODUCE,
# and the correction is here rather than in a commit of its own because leaving it would
# put two numbers measured two ways in one comment. uv.lock has not moved since 2026-07-20
# and the package count and all six per-file figures above are the ones that measurement
# recorded, so it read the same set. The 0.38 GiB between the two figures is accounted for
# by flashinfer-cubin at 255.3 MiB and nvidia-nvshmem-cu12 at 132.7, 388.0 together, whose
# markers both hold on linux x86_64 and which are therefore in the image.
#
# THE ALTERNATIVE IS NOT A SMALLER IMAGE, IT IS A DIFFERENT PLACE TO PAY. uv is on
# PATH inside the venv, so a workload can install the extra at the top of a run
# instead. That moves the same several gigabytes off the image and onto every single
# run, on a GPU that is being billed while it downloads, and it puts a resolution
# nothing reviewed inside a job that is meant to be reproducible. Paying once per image
# is the cheaper half of that trade for anything run more than a handful of times, and
# it is sharper at `olmo-core` than at `vllm`: an adaptive session is minutes of
# compute, so a 3.98 GiB install in front of it costs more than the work it enables,
# and it is paid again for every checkpoint in a training run.
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
# NEITHER POSITION NEEDS A RE-LOCK, WHICH IS THE PART THAT MAKES A THIRD ONE CHEAP.
# Both extras are already declared in pyproject.toml and both already carry their
# conflict markers in uv.lock, so turning one on moves nothing else in the resolution.
# The paragraph below on botocore[crt] is what the other kind of change costs here: one
# new edge in the s3 extra re-derives the conflicting-extra markers and drags numpy,
# rich and textual backwards. A `--frozen` sync of an extra the lockfile already
# describes cannot do that, and `--frozen` is what refuses to find out.
#
# DEFAULTS TO none, WHICH REPRODUCES THE IMAGE THAT IS REGISTERED TODAY. Which backend
# this account evaluates against is not a question a Dockerfile should answer, so this
# leaves it open and makes the answer one word. Built at what is now spelled `vllm` on
# run 31078822781, which published
# sha256:efc0556f38845d0198a79ab33362c3a6f6e1b372f820e9e65464a7336dba51b0 with torch
# 2.10.0+cu128 and vLLM importable, so that position is a thing that has run rather than
# a thing that should work. `olmo-core` has been resolved against this lockfile and not
# yet built, and the assertions below are where that stops being true or fails trying.
#
# The sets above are these, and reading one back costs no network and cannot rewrite the
# lockfile, because --frozen:
#
#     uv export --frozen --no-default-groups --no-emit-project \
#         --extra beaker --extra hf --extra s3 --extra clients [--extra vllm|olmo-core]
ARG MODEL_BACKEND=olmo-core
RUN set -eux; \
    case "${MODEL_BACKEND}" in \
        none) exit 0 ;; \
        vllm|olmo-core) ;; \
        *) echo "MODEL_BACKEND is '${MODEL_BACKEND}'; it must be none, vllm or olmo-core" >&2; exit 1 ;; \
    esac; \
    uv sync --frozen --no-default-groups --no-install-project --no-cache \
        --extra beaker --extra hf --extra s3 --extra clients \
        --extra "${MODEL_BACKEND}"; \
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
# asserted the same way whichever value MODEL_BACKEND took. A comment saying what an
# image carries is a claim nothing checks, and this file has already been read once as
# evidence that torch was here.
#
# `mock` and `litellm` are always constructible, so they are checked unconditionally
# and a loaded build is held to the same floor `none` is.
#
# WHY A LOADED POSITION CHECKS AN IMPORT AND NOT A DISTRIBUTION.
# `importlib.metadata.version` answers from a `.dist-info` directory and proves only
# that an installer wrote one. torch is a C extension linked against the CUDA userspace
# in the base, and the way it fails is a wheel that unpacks and then cannot resolve a
# symbol at load, so the import is the first thing that would notice.
# `torch.version.cuda` separates the CUDA build from the CPU-only build of the same
# version number, which is otherwise a difference nothing here would see until a run
# asked for a device.
#
# vLLM is imported rather than merely counted for the same reason, and it is the
# import a GPU-less builder is most likely to refuse. It does not refuse: the
# platform layer resolves to a CPU platform with a warning, and `LLM` is bound.
# Nothing below constructs one, because that needs a device this runner has not got.
#
# THE PROVIDER CLASSES ARE NO LONGER IMPORTED HERE, BECAUSE DOING SO PROVED NOTHING.
# This line used to import HuggingFaceProvider, VLLMProvider and VLLMServerProvider and
# read as the check that the loaded position worked. Every heavy import in those modules
# is under `if TYPE_CHECKING` or inside a method -- `providers/__init__.py` is a
# `__getattr__` shim over lazy submodules -- so all three import on an image with no
# torch and no vLLM in it. The check passed at `none` as readily as at `1`, which is the
# definition of vacuous, and the imports above and below it are the ones that bite.
#
# WHAT olmo-core CHECKS IS THE PROVIDER'S OWN IMPORT LIST, CALLED RATHER THAN COPIED.
# `_import_olmo_core` is the single function OlmoCoreProvider calls before it touches a
# checkpoint, and it reaches torch, cached_path, olmo_core.data,
# olmo_core.distributed.checkpoint, olmo_core.generate.generation_module,
# olmo_core.nn.attention and transformers behind one try/except. Calling it here means
# the build checks whatever the provider currently needs instead of a copy of that list
# that can drift from it; renaming it fails this build loudly, which is the direction
# this should fail in. Its underscore is deliberate and is the reason this sentence
# exists.
#
# get_checkpoint_metadata inside that list is the whole point of the position. A
# HuggingFace directory carries its architecture with it and a sharded DCP checkpoint
# does not, so rebuilding the model from the run config is the only way to read one, and
# ai2-olmo-core is the only thing here that can. cached_path arrives with it and with
# nothing else: it is absent from both other positions.
RUN set -eux; \
    cd /; \
    python -c "from olmo_eval.inference import create_provider; print('constructed', type(create_provider('mock', 'image-smoke')).__name__)"; \
    python -c "from olmo_eval.inference.providers import LiteLLMProvider; print('importable', LiteLLMProvider.__name__)"; \
    if [ "${MODEL_BACKEND}" = "none" ]; then exit 0; fi; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda, 'this is the CPU build of torch, so nothing here can reach a card'"; \
    if [ "${MODEL_BACKEND}" = "vllm" ]; then \
        python -c "from vllm import LLM; print('vllm imports, LLM bound:', LLM.__name__)"; \
    fi; \
    if [ "${MODEL_BACKEND}" = "olmo-core" ]; then \
        python -c "from dataclasses import fields; from olmo_eval.inference.providers.olmo_core_utils import _import_olmo_core; resolved = _import_olmo_core(); print('olmo_core loader resolved:', ', '.join(f.name for f in fields(resolved)))"; \
    fi

# What runs in this image, and it depends on how it was built. At MODEL_BACKEND=none
# there is no torch, so `mock` and `litellm` run and the four providers that load
# weights onto a card do not. At `vllm` those two plus `vllm`, `vllm_server` and `hf`
# run. At `olmo-core` those two plus `olmo_core` and `hf` run, and `vllm` and
# `vllm_server` do not. Read the ARG above to see which one this image was built as.
CMD ["olmo-eval", "--help"]
