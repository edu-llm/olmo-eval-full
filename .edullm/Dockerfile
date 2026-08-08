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
ARG INSTALL_TORCH_AND_VLLM=0
RUN set -eux; \
    if [ "${INSTALL_TORCH_AND_VLLM}" != "1" ]; then exit 0; fi; \
    uv sync --frozen --no-default-groups --no-install-project --no-cache \
        --extra beaker --extra hf --extra s3 --extra clients \
        --extra vllm; \
    uv cache clean

# Whether this image can read a raw OLMo-core checkpoint. Separate from the ARG above
# rather than a third value on it, because the two are orthogonal: this branch wants
# torch without vLLM, and the question "can it serve a model" is not the question "can
# it read the checkpoints we train".
#
# THE ARG ABOVE DOES NOT COVER THIS, WHICH IS THE WHOLE REASON THIS EXISTS. Resolved
# against uv.lock on 2026-08-07: --extra vllm brings torch 2.10.0 and vLLM 0.19.1 across
# 242 packages and does NOT contain ai2-olmo-core, because that lives in a separate
# `olmo_core` extra neither position syncs. So INSTALL_TORCH_AND_VLLM=1 would cost the
# full 4.54 GiB and still leave the image unable to load a step directory -- an image
# that looks like it can evaluate our own checkpoints and cannot.
#
# WHAT IT COSTS, AND IT IS THE CHEAPER OF THE TWO. --extra olmo-core resolves to 141
# packages: torch 2.10.0, ai2-olmo-core 2.4.0, torchao 0.17.0 and the CUDA runtime, with
# no vLLM. That is 101 packages fewer than the vllm path. ai2-olmo-core itself is a
# 582 KB pure-Python wheel, so essentially all of the weight is torch and CUDA, which is
# why a hypothetical torch-only extra would save nothing worth having.
#
# WHY ai2-olmo-core AND NOT A CONVERSION. A checkpoint written by OLMo-core is sharded
# DCP -- config.json plus model_and_optim/*.distcp -- and carries no architecture. This
# library is what rebuilds one via TransformerConfig.from_dict. A HuggingFace directory
# carries its architecture with it; a step directory does not, so there is no reading it
# without this. Converting instead means unsharding gigabytes per checkpoint against a
# CAT session that administers 13-40 items.
#
# DEFAULTS TO 1 ON THIS BRANCH, WHICH IS A DEPARTURE FROM THE ARG ABOVE AND DELIBERATE.
# That one defaults to 0 to reproduce the registered image. This branch exists to run
# adaptive diagnostics against raw training checkpoints, so an image without the loader
# is an image with no purpose. The workflow passes BASE_IMAGE and nothing else, so a
# default is the only way to express it. On a merge to main this would become everyone's
# default and is worth raising then; on edullm/** it affects this branch alone.
#
# THE LOCKFILE'S ai2-olmo-core CANNOT READ THE CHECKPOINTS THIS BRANCH EXISTS TO READ,
# WHICH IS WHY THE SYNC IS NOT THE END OF THIS LAYER. --extra olmo-core resolves PyPI
# ai2-olmo-core 2.4.0, and 2.4.0's schema has no `sequence_mixer` on
# TransformerBlockConfig and no `partial_rotary_factor` or `no_global_rope` on RoPEConfig
# -- read out of allenai/OLMo-core at tag v2.4.0 rather than inferred. Every checkpoint
# this branch targets writes all three, so TransformerConfig.from_dict raises on
# config.json before a single weight is read. An image carrying 2.4.0 is an image that
# can import olmo_core and cannot load anything we train.
#
# THE FORK IS INSTALLED HERE RATHER THAN AT THE TOP OF THE RUN, AND THAT IS THE CHANGE.
# It used to be a `uv pip install --no-deps` line in .edullm/run.yaml, which is a repair
# performed on a GPU that is already being billed, against a network that may or may not
# answer, in a command nothing reviews and nothing tests. Pinning it here makes the image
# either correct or unbuilt.
#
# --no-deps, SO THE GIT INSTALL CANNOT MOVE THE RESOLUTION. The fork declares torch and
# the rest for itself; without this flag it would re-resolve them against PyPI over the
# top of the set `uv sync --frozen` just installed from uv.lock, and the image would no
# longer be the lockfile's. It goes after the sync rather than instead of it because the
# extra is what brings torch, torchao and the surrounding dependency set -- the fork is
# a 582 KB pure-Python tree and brings none of that.
#
# PINNED BY COMMIT, THOUGH THE BRANCH IS WHAT PEOPLE SAY. edullm/memory-split-135m is a
# branch and a branch moves, so an image built from it is not reproducible and the
# platform's provenance record would name a base digest beside a ref that can change
# underneath it. 4f694f72 is what that branch pointed at on 2026-08-08. Moving the fork
# means moving this line, which is the point.
#
# git+https:// RESOLVES HERE BECAUSE THIS FILE ALREADY INSTALLS git, a few layers up --
# the ifbench dependency is a git URL, so uv could not read uv.lock without it. OLMo-core's
# own .edullm/Dockerfile reaches for GitHub tarball URLs instead, and that is not a style
# difference: its registered base is a bare python image with no git, so `pip install
# git+https://…` there would fail. A tarball would work here too and names the same forty
# hex digits; git+https is used because uv reports the resolved commit in direct_url.json,
# which is what the provenance assertion below reads.
#
# THE ASSERTIONS ARE THE POINT OF THIS LAYER, NOT THE INSTALL, and `import olmo_core` is
# not one of them. The top-level package imports lazily, so it passes on 2.4.0, on the
# fork, and on anything else called ai2-olmo-core -- OLMo-core's image learned that the
# expensive way, on a GPU run that died four minutes in against an image whose shallow
# import check had passed. What is asserted instead is the schema, three ways:
#
#   1. RoPEConfig(no_global_rope=..., partial_rotary_factor=...) constructs. On 2.4.0
#      that is a TypeError on an unexpected keyword, because neither field exists.
#   2. TransformerBlockConfig declares a `sequence_mixer` field. 2.4.0 calls it
#      `attention`, and the fork keeps `attention` as a deprecated InitVar alias -- so
#      reading dataclasses.fields() rather than trying an attribute is what separates
#      the two.
#   3. The whole round trip the loader actually performs: build this checkpoint's
#      architecture with TransformerConfig.smollm2_135M -- a builder 2.4.0 does not have
#      either -- serialize it to the on-disk config schema, and parse it back with
#      TransformerConfig.from_dict, which is the exact call that fails on 2.4.0. The
#      partial_rotary_factor is set to a non-default value first so the round trip has
#      to carry it rather than reproduce a default.
#
# The provenance check is separate and answers a different question. The three above
# prove the code in this image has the schema; direct_url.json proves it came from the
# commit named on the line above rather than from some other build that happens to.
ARG INSTALL_OLMO_CORE=1
ARG OLMO_CORE_FORK_COMMIT=4f694f721b72065caaf117591bd9387f23711a5b
RUN set -eux; \
    if [ "${INSTALL_OLMO_CORE}" != "1" ]; then exit 0; fi; \
    uv sync --frozen --no-default-groups --no-install-project --no-cache \
        --extra beaker --extra hf --extra s3 --extra clients \
        --extra olmo-core; \
    uv pip install --no-cache --no-deps \
        "ai2-olmo-core @ git+https://github.com/edu-llm/OLMo-core@${OLMO_CORE_FORK_COMMIT}"; \
    uv cache clean; \
    cd /; \
    python -c "from olmo_core.nn.rope import RoPEConfig; c = RoPEConfig(no_global_rope=True, partial_rotary_factor=0.25); assert c.partial_rotary_factor == 0.25 and c.no_global_rope is True"; \
    python -c "import dataclasses; from olmo_core.nn.transformer.config import TransformerBlockConfig; names = {f.name for f in dataclasses.fields(TransformerBlockConfig)}; assert 'sequence_mixer' in names, f'TransformerBlockConfig has no sequence_mixer field, so this is PyPI ai2-olmo-core rather than the fork: {sorted(names)}'"; \
    python -c "from olmo_core.nn.transformer.config import TransformerConfig; cfg = TransformerConfig.smollm2_135M(vocab_size=49152); cfg.block.sequence_mixer.rope.partial_rotary_factor = 0.25; data = cfg.as_config_dict(); rope = data['block']['sequence_mixer']['rope']; assert rope['partial_rotary_factor'] == 0.25, f'partial_rotary_factor did not survive serialization: {rope}'; back = TransformerConfig.from_dict(data); assert back.block.sequence_mixer.rope.partial_rotary_factor == 0.25; print('TransformerConfig.from_dict parsed sequence_mixer + partial_rotary_factor,', f'{back.num_params:,d} params')"; \
    python -c "import importlib.metadata as md; text = md.distribution('ai2-olmo-core').read_text('direct_url.json') or ''; assert 'edu-llm/OLMo-core' in text and '${OLMO_CORE_FORK_COMMIT}' in text, f'ai2-olmo-core did not come from the pinned fork commit; direct_url.json says {text!r}'; print('ai2-olmo-core', md.version('ai2-olmo-core'), 'from', text)"

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

# Whether this image can read a raw OLMo-core checkpoint, asserted for the same reason
# everything above it is: a comment is a claim nothing checks.
#
# A SEPARATE BLOCK RATHER THAN THREE MORE LINES ABOVE, AND NOT FOR TIDINESS. That block
# exits 0 when INSTALL_TORCH_AND_VLLM is not 1, which is this branch's value, so an
# assertion appended to it would never run in exactly the image it was written to check.
#
# A DIFFERENT QUESTION FROM THE SCHEMA ASSERTIONS IN THE INSTALL LAYER, WHICH IS WHY BOTH
# ARE HERE. Those ask whether the olmo_core in this image can parse our checkpoints'
# config; this asks whether the five symbols *this repository's* loader binds are still
# where it expects them, which a fork is exactly the sort of thing to move.
#
# IT ASSERTS core_utils._import_olmo_core() RATHER THAN `import olmo_core`, because that
# is the precise call the loader makes and it is stricter than the sum of its parts. The
# helper imports TokenizerConfig, get_checkpoint_metadata, GenerationConfig,
# TransformerGenerationModule and AttentionBackendName inside one try, and converts any
# failure into an ImportError naming the extra. A bare `import olmo_core` would pass on a
# version that moved one of those five symbols; this cannot. If it returns, every name
# the checkpoint path resolves is bound.
#
# torch is checked for its CUDA build here too. The block above only does that at
# INSTALL_TORCH_AND_VLLM=1, so at 0-with-olmo-core -- this branch -- nothing else would
# notice a CPU-only wheel until a run asked for a device.
ARG INSTALL_OLMO_CORE
RUN set -eux; \
    if [ "${INSTALL_OLMO_CORE}" != "1" ]; then exit 0; fi; \
    cd /; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda, 'this is the CPU build of torch, so nothing here can reach a card'"; \
    python -c "import olmo_core; print('ai2-olmo-core', getattr(olmo_core, '__version__', 'unknown'))"; \
    python -c "from olmo_eval.inference.providers import olmo_core_utils as u; i = u._import_olmo_core(); print('olmo_core loader ready:', i.TransformerGenerationModule.__name__, i.TokenizerConfig.__name__, i.AttentionBackendName.__name__)"

# What runs in this image, and it depends on how it was built. At
# INSTALL_TORCH_AND_VLLM=0 there is no torch, so `mock` and `litellm` run and the
# three providers that load weights onto a card do not. At 1 all six run. Read the
# ARG above to see which one this image was built as.
CMD ["olmo-eval", "--help"]
