"""Curate the 200-model roster: English-first, general-purpose, vLLM-clean.

`build_models_200.py` optimised for size/org spread over a pool that had already
lost most mainstream English families to the vLLM blocklist. The bins therefore
filled with language-targeted checkpoints (Polish, German, Japanese, Korean,
Russian, ...), mirror re-uploads, and domain specialists — none of which suit an
English tutoring instrument.

This selector rebuilds the roster from a wider candidate pool (the builder's RAW
pool plus the ATLAS/OpenLM banks) under explicit, auditable rules:

  * hard exclusions   — vLLM-broken, language-targeted, mirrors, merges,
                        quantised, roleplay, thinking-trace, out-of-range
  * tiering           — general-purpose English first, multilingual-including-
                        English second, domain specialists only as filler
  * one per identity  — (family, size, tuning); base vs instruct at the same
                        size is a *wanted* pair, not a duplicate
  * forced anchors    — the pilot-intersection models that link this run back to
                        the frozen 82-model calibration

Run: uv run python AdaptiveTesting/Experiments/models_200/curate_models_200.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

RUNTIME_YAML = REPO / "AdaptiveTesting" / "Inputs" / "Models" / "models_200.yaml"
EDULLM_YAML = REPO / "eduLLM-Evals" / "models.yaml"
TUTOR_OUT = REPO / "eduLLM-Evals" / "models_200.yaml"
OPENLM_CSV = REPO / "AdaptiveTesting" / "Inputs" / "OpenLM" / "models_selected.csv"
ATLAS_CAL_CSV = (
    REPO / "AdaptiveTesting" / "Inputs" / "ATLAS" / "arc_0p5_7b" / "cal_models_0p5_7b.csv"
)
ATLAS_SIZES_CSV = (
    REPO
    / "AdaptiveTesting"
    / "Experiments"
    / "atlas_recalibrate_0p5_7b"
    / "results"
    / "atlas_model_sizes_heuristic.csv"
)
REPORT_OUT = HERE / "results" / "models_200_curation_report.md"

PARAM_MIN, PARAM_MAX = 0.2, 7.0
TARGET_N = 200

# Set by --allow-vllm-broken. The vLLM exclusions bind only on the vLLM path;
# the CPU sweep runs `--backend hf` (see the 200-Model Run pre-registration
# §2.4), where these families load fine and supply the low-ability floor.
ALLOW_VLLM_BROKEN = False


# --------------------------------------------------------------------------- #
# Exclusion rules
# --------------------------------------------------------------------------- #

# Families / id patterns that failed under vLLM on the P6 sweep (engine load,
# ModelConfig ValidationError, hang, or forced hf_fallback). Copied from
# build_models_200.py so the two selectors cannot drift apart, then extended.
VLLM_BLOCK_SUBSTR = (
    "mamba",
    "openelm",
    "gemma-3",
    "recurrentgemma",
    "gemma-2-2b-it",
    "gemma-2b-it",
    "gemma-1.1-2b-it",
    "gemma-1.1-7b-it",
    "gemma-7b-it",
    "gemma-2-9b-it",
    "tiiuae/falcon-",
    "cerebras",
    "mosaicml/mpt",
    "mpt-7b",
    "internlm",
    "h2o-danube3",
    "pythia",
    "gpt-neo-",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "bloomz",
    "llm360/amber",
    "tinyllama",
    "microsoft/phi-2",
    "redpajama",
    "rwkv",
    "zamba",
    "hazyresearch/based",
    "olmoe",
    # extensions: same failure classes, reachable from the wider ATLAS pool
    "stablelm-base-alpha",
    "gpt-j-6b",
    "gpt-neox",
    "gptj",
    "persimmon",
    "jamba",
    "griffin",
    "hymba",
    "plamo",
    "xgen-7b",
    # DCLM ships a custom `open_lm` architecture vLLM cannot load
    "dclm",
)

# Not decoder-only causal LMs. The harness scores next-token loglikelihood and
# generates continuations, so encoder-decoder / seq2seq checkpoints are simply
# the wrong model class — and vLLM will not serve most of them.
NON_CAUSAL_MARKERS = (
    "t5",
    "tfine",
    "-bart",
    "pegasus",
    "switch-base",
    "switch-large",
    "ul2",
    "bert",
    "roberta",
    "electra",
    "deberta",
    "-nli",
    "sentence-",
    "-embed",
    "reranker",
)

# Sparse mixtures: routed experts change the memory profile and several failed
# the sweep. `a400m` / `a800m` style suffixes denote active-parameter counts.
MOE_MARKERS = (
    "mixtral",
    "-moe",
    "moe-",
    "olmoe",
    "a400m",
    "a800m",
    "a13b",
    "a3b",
    "a2b",
    "granitemoe",
    "deepseekmoe",
)

# `8x7b` is a 47B Mixtral, not a 7B model — name-based size tables read it as 7.
_MOE_SHAPE_RE = re.compile(r"\d+\s*x\s*\d+\.?\d*\s*b\b")

# Dialogue-response models (Reddit-style single turns) rather than instruction
# followers: they score zero on every tutoring rubric, and an all-fail row gets
# dropped by the variance filter anyway.
CHITCHAT_MARKERS = ("dialogpt", "blenderbot", "-persona", "convai")

VLLM_BLOCK_EXACT = {
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "LLM360/Amber",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "togethercomputer/RedPajama-INCITE-7B-Base",
    "apple/OpenELM-1_1B",
    "apple/OpenELM-3B",
    "cerebras/Cerebras-GPT-1.3B",
    "cerebras/Cerebras-GPT-2.7B",
    "cerebras/Cerebras-GPT-6.7B",
    "mosaicml/mpt-7b",
    "mosaicml/mpt-7b-instruct",
    "h2oai/h2o-danube3-4b-base",
    "internlm/internlm2-1_8b",
    "internlm/internlm2-7b",
    "internlm/internlm2_5-7b",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "KoboldAI/GPT-J-6B-Janeway",
    "NbAiLab/nb-gpt-j-6B",
    "togethercomputer/GPT-JT-6B-v1",
    "kakaobrain/kogpt",
    "stanford-crfm/BioMedLM",
    "Writer/camel-5b-hf",
    "replit/replit-code-v1-3b",
    "replit/replit-code-v1_5-3b",
    "bigcode/santacoder",
    "AlekseyKorshuk/vicuna-7b",
}

# Language-targeted checkpoints. An English tutoring instrument gets no signal
# from a Polish- or Japanese-only decoder, and their responses degrade the judge
# pass. Keyed by the substring that triggers the drop so the report can explain
# every removal. Multilingual models that genuinely include English are NOT here
# — see MULTILINGUAL_OK.
LANG_MARKERS: dict[str, str] = {
    "gpt-sw3": "Swedish",
    "ai-sweden": "Swedish",
    "rugpt": "Russian",
    "vikhr": "Russian",
    "ruadapt": "Russian",
    "saiga": "Russian",
    "gerpt": "German",
    "german": "German",
    "wechsel-german": "German",
    "sauerkraut": "German",
    "leo-hessianai": "German",
    "leolm": "German",
    "bubbliiiing": "German",
    "occiglot": "German",
    "polish": "Polish",
    "bielik": "Polish",
    "polka": "Polish",
    "trurl": "Polish",
    "curie-7b": "Polish",
    "japanese": "Japanese",
    "open-calm": "Japanese",
    "llm-jp": "Japanese",
    "rinna": "Japanese",
    "swallow": "Japanese",
    "elyza": "Japanese",
    "youri": "Japanese",
    "stockmark": "Japanese",
    "weblab": "Japanese",
    "calm2": "Japanese",
    "sarashina": "Japanese",
    "korean": "Korean",
    "polyglot-ko": "Korean",
    "kogpt": "Korean",
    "koalpaca": "Korean",
    "kullm": "Korean",
    "eeve": "Korean",
    "kanana": "Korean",
    "gemma-ko": "Korean",
    "llama-ko": "Korean",
    "komt": "Korean",
    "synatra": "Korean",
    "hyperclovax": "Korean",
    "-bne": "Spanish",
    "neurona": "Spanish",
    "spanish": "Spanish",
    "flor-": "Catalan",
    "aguila": "Catalan",
    "finnish": "Finnish",
    "turkunlp": "Finnish",
    "minerva-": "Italian",
    "italian": "Italian",
    "cerbero": "Italian",
    "modello-italia": "Italian",
    "fr-boris": "French",
    "croissant": "French",
    "french": "French",
    "vigogne": "French",
    "lyso": "French",
    "czech": "Czech",
    "csmpt": "Czech",
    "nb-gpt": "Norwegian",
    "norwegian": "Norwegian",
    "norallm": "Norwegian",
    "danish": "Danish",
    "munin": "Danish",
    "swedish": "Swedish",
    "viking-7b": "Nordic",
    "turkish": "Turkish",
    "kanarya": "Turkish",
    "trendyol": "Turkish",
    "cosmosgpt": "Turkish",
    "arabic": "Arabic",
    "jais": "Arabic",
    "acegpt": "Arabic",
    "hindi": "Hindi",
    "airavata": "Hindi",
    "openhathi": "Hindi",
    "sarvam": "Hindi",
    "indic": "Indic",
    "indonesia": "Indonesian",
    "cendol": "Indonesian",
    "komodo": "Indonesian",
    "sea-lion": "SEA",
    "sailor": "SEA",
    "vietnamese": "Vietnamese",
    "vinallama": "Vietnamese",
    "phogpt": "Vietnamese",
    "thai": "Thai",
    "typhoon": "Thai",
    "wangchan": "Thai",
    "hebrew": "Hebrew",
    "dictalm": "Hebrew",
    "greek": "Greek",
    "meltemi": "Greek",
    "hungarian": "Hungarian",
    "puli-": "Hungarian",
    "romanian": "Romanian",
    "ukrainian": "Ukrainian",
    "persian": "Persian",
    "maral-": "Persian",
    "dorna": "Persian",
    "portuguese": "Portuguese",
    "sabia": "Portuguese",
    "gervasio": "Portuguese",
    "cabrita": "Portuguese",
    "bode-": "Portuguese",
    "dutch": "Dutch",
    "geitje": "Dutch",
    "fietje": "Dutch",
    "chinese": "Chinese",
    "chatglm": "Chinese",
    "baichuan": "Chinese",
    "yi-ko": "Chinese",
    "xverse": "Chinese",
    "skywork": "Chinese",
    "telechat": "Chinese",
    "orion-14b": "Chinese",
    "tigerbot": "Chinese",
    "ziya": "Chinese",
    "firefly": "Chinese",
    "bluelm": "Chinese",
    "openbuddy": "Chinese",
    "sus-chat": "Chinese",
    "hunyuan": "Chinese",
    "ernie": "Chinese",
    "cpm-bee": "Chinese",
    "aquila": "Chinese",
}

# Genuinely multilingual, English-competent checkpoints. Kept as Tier 2 — they
# would otherwise be caught by LANG_MARKERS or look language-targeted by org.
MULTILINGUAL_OK = {
    "utter-project/EuroLLM-1.7B",
    "utter-project/EuroLLM-1.7B-Instruct",
    "BSC-LT/salamandra-2b",
    "BSC-LT/salamandra-2b-instruct",
    "ai-forever/mGPT",
    "facebook/xglm-1.7B",
    "facebook/xglm-2.9B",
    "facebook/xglm-4.5B",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "OpenBMB/MiniCPM4-0.5B",
    "CohereForAI/aya-expanse-8b",
}

# Mirror / re-upload orgs: byte-identical weights would enter the IRT matrix as
# duplicate persons, faking independent observations.
MIRROR_ORGS = {
    "unsloth",
    "sberbank-ai",
    "TheBloke",
    "mlx-community",
    "ModelCloud",
    "neuralmagic",
    "RedHatAI",
    "AlekseyKorshuk",
    "Felladrin",
    "premai-io",
}
# NousResearch publishes both originals (Hermes) and ungated re-uploads of other
# labs' weights; only the re-uploads are mirrors.
MIRROR_EXACT = {
    "sberbank-ai/rugpt3large_based_on_gpt2",  # dup of ai-forever/…
    "NousResearch/Llama-2-7b-hf",
    "NousResearch/Llama-2-7b-chat-hf",
    "NousResearch/Meta-Llama-3-8B",
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "NousResearch/Llama-3.2-1B",
    "NousResearch/CodeLlama-7b-hf",
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Qwen2.5-1.5B",
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/SmolLM2-1.7B",
    "unsloth/SmolLM2-1.7B-Instruct",
    "unsloth/gemma-2-2b",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3.5-mini-instruct",
}

MERGE_MARKERS = (
    "slerp",
    "dare",
    "-ties",
    "linear-merge",
    "passthrough",
    "mergekit",
    "lazymerge",
    "frankenstein",
    "franken",
    "-merge",
    "merged",
    "-mix-",
    "mixtures",
    "task-arith",
    "della",
    "breadcrumbs",
    "model_stock",
    "-moe",
    "clown",
    "chimera",
)

QUANT_MARKERS = (
    "gptq",
    "awq",
    "gguf",
    "-4bit",
    "-8bit",
    "4bit",
    "8bit",
    "bnb",
    "int4",
    "int8",
    "-lora",
    "lora-",
    "adapter",
    "qlora",
    "exl2",
    "w4a16",
    "w8a8",
    "fp8",
    "autoround",
    "smoothquant",
    "-hqq",
)

ROLEPLAY_MARKERS = (
    "roleplay",
    "-rp-",
    "-erp",
    "nsfw",
    "uncensored",
    "abliterated",
    "waifu",
    "toxic",
    "dolphin",
    "pygmalion",
    "mythomax",
    "mytho",
    "noromaid",
    "kunoichi",
    "silicon-maid",
    "lewd",
    "hentai",
    "smut",
    "janeway",
    "shygmalion",
    "erebus",
    "nymph",
    "lemonade",
    "cinder",
    "bianca",
    "eros",
    "psyfighter",
    "tiefighter",
    "estopia",
    "fimbulvetr",
    "westlake",
    "sanguine",
    "unholy",
    "cinematika",
    "storywriter",
    "novel",
)

# Reasoning models that emit <think> traces. Qwen3 / SmolLM3 stay in because the
# manifest disables thinking explicitly; these have no such switch and would
# spend the whole generation budget on the trace.
THINKING_MARKERS = (
    "r1-distill",
    "deepseek-r1",
    "qwq",
    "-thinking",
    "openthinker",
    "sky-t1",
    "marco-o1",
    "-reasoning",
    "reasoner",
    "o1-",
    "-cot-",
    "s1-32b",
    "still-",
)

JUNK_MARKERS = (
    "random",
    "untrained",
    "dummy",
    "debug",
    "sanity",
    "test-",
    "-test",
    "tmp",
    "scratch",
    "delete",
    "broken",
    "wip",
    "placeholder",
    "experiment",
    "-exp-",
    "trial",
    "checkpoint-",
    "-ckpt",
    "step-",
    "-sft-v",
    "my-",
    "-demo",
)

VISION_MARKERS = ("-vl", "vision", "llava", "idefics", "-mm-", "multimodal", "moondream", "florence")

CODE_MARKERS = (
    "coder",
    "codegen",
    "starcoder",
    "santacoder",
    "replit",
    "magicoder",
    "codellama",
    "code-",
    "-code",
    "stablecode",
    "wizardcoder",
    "deepseek-coder",
    "codegemma",
    "codeqwen",
)
MATH_MARKERS = ("math", "numina", "metamath", "mathstral", "-gsm", "abel-")
BIO_MARKERS = (
    "biogpt",
    "biomed",
    "clinical",
    "pubmed",
    "meditron",
    "medalpaca",
    "-med-",
    "medllama",
    "legal",
    "finance",
    "finbert",
    "fingpt",
    "chemistry",
    "chem-",
)

# Orgs whose general-purpose releases are trustworthy roster material. Used to
# gate the wider ATLAS/OpenLM pools, which are otherwise dominated by anonymous
# community finetunes.
REPUTABLE_ORGS = {
    "qwen",
    "meta-llama",
    "mistralai",
    "google",
    "microsoft",
    "allenai",
    "huggingfacetb",
    "huggingfaceh4",
    "stabilityai",
    "tiiuae",
    "ibm-granite",
    "ibm",
    "openbmb",
    "01-ai",
    "deepseek-ai",
    "eleutherai",
    "bigscience",
    "facebook",
    "nvidia",
    "salesforce",
    "bigcode",
    "databricks",
    "togethercomputer",
    "openlm-research",
    "princeton-nlp",
    "llm360",
    "openai-community",
    "h2oai",
    "lmsys",
    "writer",
    "cohereforai",
    "coherelabs",
    "utter-project",
    "bsc-lt",
    "lgai-exaone",
    "ai-forever",
    "mbzuai",
    "sapienzanlp",
    "arcee-ai",
    "upstage",
    "openchat",
    "berkeley-nest",
    "teknium",
    "intel",
    "argilla",
    "jetbrains",
    "zyphra",
    "lightonai",
    "state-spaces",
    "apple",
    "amd",
    "motif-technologies",
    "pleias",
    "kyutai",
    "liquidai",
    "moonshotai",
    "nousresearch",
    "open-orca",
    "garage-baind",
    "jondurbin",
    "migtissera",
    "pankajmathur",
    "nexusflow",
    "wizardlmteam",
    "snorkelai",
    "deci",
    "bee-spoke-data",
}

# First-party labs: the org that trained and released the weights. When two
# repos resolve to the same (family, size, tuning) identity, the first-party
# release wins — otherwise a third-party experiment like
# `HuggingFaceH4/mistral-7b-grok` displaces `mistralai/Mistral-7B-v0.1`.
FIRST_PARTY_ORGS = {
    "qwen",
    "meta-llama",
    "mistralai",
    "google",
    "microsoft",
    "allenai",
    "huggingfacetb",
    "stabilityai",
    "tiiuae",
    "ibm-granite",
    "openbmb",
    "01-ai",
    "deepseek-ai",
    "eleutherai",
    "bigscience",
    "facebook",
    "nvidia",
    "salesforce",
    "bigcode",
    "databricks",
    "openlm-research",
    "princeton-nlp",
    "openai-community",
    "h2oai",
    "apple",
    "cohereforai",
    "coherelabs",
    "lgai-exaone",
    "utter-project",
    "bsc-lt",
    "mbzuai",
    "writer",
    "llm360",
    "state-spaces",
    "ai-forever",
    "sapienzanlp",
}

# The org that owns each family name. Without this, a third-party derivative can
# win its identity contest on a diversity tiebreak — `togethercomputer/LLaMA-2-7B-32K`
# displacing `meta-llama/Llama-2-7b-hf`.
FAMILY_HOME_ORG = {
    "Llama2": "meta-llama",
    "Llama3": "meta-llama",
    "Llama3.1": "meta-llama",
    "Llama3.2": "meta-llama",
    "Mistral": "mistralai",
    "Qwen1.5": "qwen",
    "Qwen2": "qwen",
    "Qwen2.5": "qwen",
    "Qwen2.5-Coder": "qwen",
    "Qwen2.5-Math": "qwen",
    "Qwen2-Math": "qwen",
    "Qwen3": "qwen",
    "Gemma1": "google",
    "Gemma2": "google",
    "Phi1": "microsoft",
    "Phi3": "microsoft",
    "Phi3.5": "microsoft",
    "Phi4": "microsoft",
    "Orca2": "microsoft",
    "OLMo": "allenai",
    "OLMo2": "allenai",
    "Tulu": "allenai",
    "SmolLM": "huggingfacetb",
    "SmolLM2": "huggingfacetb",
    "SmolLM3": "huggingfacetb",
    "Granite": "ibm-granite",
    "Granite3": "ibm-granite",
    "Falcon3": "tiiuae",
    "StableLM": "stabilityai",
    "StableLM2": "stabilityai",
    "StableCode": "stabilityai",
    "Yi": "01-ai",
    "Yi1.5": "01-ai",
    "MiniCPM": "openbmb",
    "MiniCPM3": "openbmb",
    "MiniCPM4": "openbmb",
    "DeepSeek-LLM": "deepseek-ai",
    "DeepSeek-Coder": "deepseek-ai",
    "DeepSeek-Math": "deepseek-ai",
    "BLOOM": "bigscience",
    "XGLM": "facebook",
    "MobileLLM": "facebook",
    "GPT-2": "openai-community",
    "OpenLLaMA": "openlm-research",
    "Sheared-LLaMA": "princeton-nlp",
    "Minitron": "nvidia",
    "Nemotron": "nvidia",
    "Danube": "h2oai",
    "CodeGen": "salesforce",
    "StarCoder": "bigcode",
    "StarCoder2": "bigcode",
    "Dolly": "databricks",
    "Salamandra": "bsc-lt",
    "EuroLLM": "utter-project",
    "EXAONE": "lgai-exaone",
    "mGPT": "ai-forever",
    "LaMini": "mbzuai",
    "Palmyra": "writer",
    "DCLM": "apple",
    "Command-R": "cohereforai",
    "Zephyr": "huggingfaceh4",
    "OpenChat": "openchat",
    "NeuralChat": "intel",
    "Notus": "argilla",
    "Starling": "nexusflow",
    "Vicuna": "lmsys",
    "OpenHermes": "teknium",
    "Hermes": "nousresearch",
    "Nous-Hermes": "nousresearch",
    "WizardLM": "wizardlmteam",
    "Platypus": "garage-baind",
    "OpenOrca": "open-orca",
    "Airoboros": "jondurbin",
    "DeciLM": "deci",
}


def org_rank(org: str) -> int:
    return 0 if org.lower() in FIRST_PARTY_ORGS else 1


def is_family_home(c: "Candidate") -> bool:
    home = FAMILY_HOME_ORG.get(c.family)
    return home is None or home == c.org.lower()


@dataclass
class Candidate:
    mid: str
    params_b: float
    family: str
    arch: str = "transformer"
    sources: set[str] = field(default_factory=set)
    openlm_avg: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def org(self) -> str:
        return self.mid.split("/")[0] if "/" in self.mid else ""

    @property
    def low(self) -> str:
        return self.mid.lower()

    @property
    def name(self) -> str:
        """Repo name only. Marker matching that could collide with an org name
        (e.g. `pankajmathur` contains "math") must use this, not `low`."""
        return self.mid.split("/")[-1].lower()


def _hit(low: str, markers) -> str | None:
    for m in markers:
        if m in low:
            return m
    return None


# `1_6b` (stablelm-2-1_6b) is a size; `v3_70b` (orca_mini_v3_70b) is a 70B model
# whose underscore belongs to the version, so only a single digit may follow one.
_SIZE_RE = re.compile(r"(?:(\d+)_(\d)|(\d+(?:\.\d+)?))\s*([bm])\b")


def declared_params(mid: str) -> float | None:
    """Parameter count as stated in the repo name, in billions.

    Guards against heuristic size tables that read a version string as a size —
    `Tess-10.7B-v1.5b` is a 10.7B model, not a 1.5B one.
    """
    m = _SIZE_RE.search(mid.split("/")[-1].lower())
    if not m:
        return None
    whole, frac, plain, unit = m.groups()
    try:
        val = float(f"{whole}.{frac}") if plain is None else float(plain)
    except ValueError:
        return None
    return val / 1000.0 if unit == "m" else val


def is_vllm_blocked(mid: str) -> str | None:
    if mid in VLLM_BLOCK_EXACT:
        return "exact-skiplist"
    low = mid.lower()
    hit = _hit(low, VLLM_BLOCK_SUBSTR)
    if hit:
        return hit
    if "gemma" in low and ("-it" in low or "_it" in low):
        return "gemma-it"
    return None


def language_of(mid: str) -> str | None:
    if mid in MULTILINGUAL_OK:
        return None
    low = mid.lower()
    for marker, lang in LANG_MARKERS.items():
        if marker in low:
            return lang
    return None


def classify(c: Candidate) -> tuple[str | None, str]:
    """Return (exclusion_reason, detail). None reason == keep."""
    low = c.low
    if not (PARAM_MIN <= c.params_b <= PARAM_MAX):
        return "out-of-range", f"{c.params_b}B"
    declared = declared_params(c.mid)
    if declared is not None and declared > PARAM_MAX:
        return "out-of-range", f"name declares {declared}B (table said {c.params_b}B)"
    if c.arch != "transformer":
        return "arch", c.arch
    hit = _hit(c.name, NON_CAUSAL_MARKERS)
    if hit:
        return "not-causal-lm", hit
    hit = _hit(low, MOE_MARKERS)
    if hit:
        return "mixture-of-experts", hit
    m = _MOE_SHAPE_RE.search(c.name)
    if m:
        return "mixture-of-experts", m.group(0)
    hit = _hit(c.name, CHITCHAT_MARKERS)
    if hit:
        return "chitchat", hit
    reason = is_vllm_blocked(c.mid)
    if reason and not ALLOW_VLLM_BROKEN:
        return "vllm-broken", reason
    if c.mid in MIRROR_EXACT:
        return "mirror", "duplicate weights"
    if c.org in MIRROR_ORGS:
        return "mirror", f"re-upload org {c.org}"
    lang = language_of(c.mid)
    if lang:
        return "language-targeted", lang
    for label, markers in (
        ("quantised", QUANT_MARKERS),
        ("merge", MERGE_MARKERS),
        ("roleplay", ROLEPLAY_MARKERS),
        ("thinking-trace", THINKING_MARKERS),
        ("junk", JUNK_MARKERS),
        ("vision", VISION_MARKERS),
    ):
        hit = _hit(low, markers)
        if hit:
            return label, hit
    return None, ""


def specialism(c: Candidate) -> str | None:
    # name-scoped: the org can contain a marker by coincidence
    name = c.name
    if _hit(name, BIO_MARKERS):
        return "biomedical"
    if _hit(name, CODE_MARKERS):
        return "code"
    if _hit(name, MATH_MARKERS):
        return "math"
    return None


INSTRUCT_MARKERS = (
    "instruct",
    "-it",
    "-chat",
    "chat-",
    "-sft",
    "sft-",
    "-dpo",
    "dpo-",
    "zephyr",
    "hermes",
    "vicuna",
    "tulu",
    "openchat",
    "-rlhf",
    "-ppo",
    "-orpo",
    "-kto",
    "alpaca",
    "wizard",
    "starling",
    "neural-chat",
    "notus",
    "notux",
    "-aligned",
    "-instructed",
    "openorca",
    "orca-2",
    "orca_mini",
    "collectivecognition",
    "bagel",
    "platypus",
    "synthia",
    "synthia",
    "beagle",
    "merlinite",
    "airoboros",
    "acein",
    "helpful",
    "oasst",
)


def tuning_of(mid: str) -> str:
    low = mid.lower()
    if _hit(low, INSTRUCT_MARKERS):
        return "instruct"
    return "base"


def tier_of(c: Candidate) -> int:
    """Selection priority.

    1 = general-purpose English from a known lab   (the core roster)
    2 = multilingual but English-competent          (kept, lower priority)
    3 = domain specialist                           (filler only, added last)
    4 = anonymous community finetune                (never selected)
    """
    if c.org.lower() not in REPUTABLE_ORGS:
        return 4
    if specialism(c):
        return 3
    if c.mid in MULTILINGUAL_OK:
        return 2
    return 1


# Provenance that predates this selector. `models_200` is deliberately excluded:
# it is this script's own output, so trusting it would make confidence
# self-referential once the roster has been written once.
CURATED_SOURCES = {"builder-raw", "edullm_100"}


def confidence_of(c: Candidate) -> tuple[str, str]:
    """(level, why) — surfaces entries worth a human look before launch."""
    if c.sources & CURATED_SOURCES:
        return "high", "hand-maintained pool (builder RAW / eduLLM 100)"
    if not c.sources - {"atlas_sizes"}:
        return "low", "params_b is a name-heuristic guess; size unverified"
    tier = tier_of(c)
    if tier == 3:
        return "medium", f"{specialism(c)} specialist, added as filler"
    if tier == 2:
        return "medium", "multilingual rather than English-first"
    return "medium", "new from ATLAS/OpenLM bank; not previously vetted"


# --------------------------------------------------------------------------- #
# Pool loading
# --------------------------------------------------------------------------- #


def _canon_family(mid: str, fallback: str = "") -> str:
    """Collapse an id to a family label so size/tuning caps bite correctly."""
    low = mid.lower()
    name = mid.split("/")[-1]
    table = [
        ("qwen3", "Qwen3"),
        ("qwen2.5-coder", "Qwen2.5-Coder"),
        ("qwen2.5-math", "Qwen2.5-Math"),
        ("qwen2.5", "Qwen2.5"),
        ("qwen2-math", "Qwen2-Math"),
        ("qwen2", "Qwen2"),
        ("qwen1.5", "Qwen1.5"),
        ("llama-3.2", "Llama3.2"),
        ("llama-3.1", "Llama3.1"),
        ("llama-3", "Llama3"),
        ("open_llama", "OpenLLaMA"),
        ("openllama", "OpenLLaMA"),
        ("sheared-llama", "Sheared-LLaMA"),
        ("llama-2", "Llama2"),
        # instruction-tuned lineages: distinct alignment recipes, so they earn
        # their own family slot, but every version of one recipe collapses here
        ("openhermes", "OpenHermes"),
        ("hermes", "Hermes"),
        ("bagel", "Bagel"),
        ("notus", "Notus"),
        ("notux", "Notus"),
        ("openorca", "OpenOrca"),
        ("openchat", "OpenChat"),
        ("neural-chat", "NeuralChat"),
        ("notus", "Notus"),
        ("starling", "Starling"),
        ("vicuna", "Vicuna"),
        ("orca-2", "Orca2"),
        ("orca_mini", "OrcaMini"),
        ("wizardlm", "WizardLM"),
        ("capybara", "Capybara"),
        ("beagle", "Beagle"),
        ("distilabel", "Distilabel"),
        ("airoboros", "Airoboros"),
        ("platypus", "Platypus"),
        ("tess-", "Tess"),
        ("synthia", "Synthia"),
        ("merlinite", "Merlinite"),
        ("llemma", "Llemma"),
        ("dclm", "DCLM"),
        ("command-r", "CommandR"),
        ("h2ogpt", "H2OGPT"),
        ("collectivecognition", "CollectiveCognition"),
        ("tulu", "Tulu"),
        ("decilm", "DeciLM"),
        ("mistral", "Mistral"),
        ("olmo-2", "OLMo2"),
        ("olmo", "OLMo"),
        ("gemma-2", "Gemma2"),
        ("gemma", "Gemma1"),
        ("phi-4", "Phi4"),
        ("phi-3.5", "Phi3.5"),
        ("phi-3", "Phi3"),
        ("phi-1", "Phi1"),
        ("smollm3", "SmolLM3"),
        ("smollm2", "SmolLM2"),
        ("smollm", "SmolLM"),
        ("granite-3", "Granite3"),
        ("granite", "Granite"),
        ("minicpm4", "MiniCPM4"),
        ("minicpm3", "MiniCPM3"),
        ("minicpm", "MiniCPM"),
        ("falcon3", "Falcon3"),
        ("stablelm-2", "StableLM2"),
        ("stablelm", "StableLM"),
        ("yi-1.5", "Yi1.5"),
        ("yi-", "Yi"),
        ("deepseek-coder", "DeepSeek-Coder"),
        ("deepseek-math", "DeepSeek-Math"),
        ("deepseek-llm", "DeepSeek-LLM"),
        ("starcoder2", "StarCoder2"),
        ("starcoder", "StarCoder"),
        ("codegen", "CodeGen"),
        ("bloom", "BLOOM"),
        ("xglm", "XGLM"),
        ("gpt2", "GPT-2"),
        ("mgpt", "mGPT"),
        ("eurollm", "EuroLLM"),
        ("salamandra", "Salamandra"),
        ("exaone", "EXAONE"),
        ("minitron", "Minitron"),
        ("nemotron", "Nemotron"),
        ("danube", "Danube"),
        ("zephyr", "Zephyr"),
        ("lamini", "LaMini"),
        ("palmyra", "Palmyra"),
        ("mobilellm", "MobileLLM"),
        ("dolly", "Dolly"),
        ("aquila", "Aquila"),
        ("hermes", "Hermes"),
        ("amber", "Amber"),
        ("pythia", "Pythia"),
    ]
    for key, fam in table:
        # A key ending in a digit must not run into a size number: `llama-2`
        # should match `Llama-2-7b` but not `smol_llama-220M`.
        pat = re.escape(key) + (r"(?!\d)" if key[-1].isdigit() else "")
        if re.search(pat, low):
            return fam
    if fallback:
        return fallback
    return re.sub(r"[-_.]?\d+\.?\d*[bm]\b.*$", "", name, flags=re.I) or name


def load_builder_raw() -> list[Candidate]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("bm200", HERE / "build_models_200.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    out = []
    for mid, p, fam, arch in mod.RAW:
        out.append(Candidate(mid, float(p), fam, arch, {"builder-raw"}))
    return out


def load_yaml_roster(path: Path, source: str) -> list[Candidate]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for entry in data.get("models", []):
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        mid = entry["id"]
        p = float(entry.get("params_b") or 0.0)
        extra = {k: v for k, v in entry.items() if k not in {"id", "params_b", "family", "arch"}}
        out.append(
            Candidate(
                mid,
                p,
                entry.get("family") or _canon_family(mid),
                entry.get("arch", "transformer"),
                {source},
                extra=extra,
            )
        )
    return out


def load_csv_pool(path: Path, id_col: str, source: str, avg_col: str | None = None) -> list[Candidate]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = (row.get(id_col) or "").strip()
            if not mid or "/" not in mid:
                continue
            try:
                p = float(row.get("params_b") or 0.0)
            except ValueError:
                continue
            avg = None
            if avg_col:
                try:
                    avg = float(row[avg_col])
                except (KeyError, TypeError, ValueError):
                    avg = None
            out.append(Candidate(mid, p, _canon_family(mid), "transformer", {source}, openlm_avg=avg))
    return out


HEURISTIC_SOURCES = {"atlas_sizes", "atlas_cal"}


def _resolve_params(c: Candidate) -> None:
    """Prefer the size stated in the repo name when the size we have came only
    from a name-heuristic table, so `params_b` is never a version-string artefact."""
    if c.sources - HEURISTIC_SOURCES:
        return  # a hand-curated roster or the leaderboard supplied it
    declared = declared_params(c.mid)
    if declared is not None and PARAM_MIN <= declared <= PARAM_MAX:
        c.params_b = declared


def _name_key(mid: str) -> str:
    """Same checkpoint re-published under a differently-spelled org/name:
    `ibm/granite-7b-base` vs `ibm-granite/granite-7b-base`,
    `StabilityAI/stablecode-3b` vs `stabilityai/stable-code-3b`."""
    return re.sub(r"[-_.]", "", mid.split("/")[-1].lower())


def build_pool() -> dict[str, Candidate]:
    pool: dict[str, Candidate] = {}
    order = [
        load_yaml_roster(RUNTIME_YAML, "models_200"),
        load_yaml_roster(EDULLM_YAML, "edullm_100"),
        load_builder_raw(),
        load_csv_pool(OPENLM_CSV, "fullname", "openlm", avg_col="average"),
        load_csv_pool(ATLAS_CAL_CSV, "model", "atlas_cal"),
        load_csv_pool(ATLAS_SIZES_CSV, "model", "atlas_sizes"),
    ]
    for group in order:
        for c in group:
            key = c.mid.lower()
            if key in pool:
                cur = pool[key]
                cur.sources |= c.sources
                if cur.openlm_avg is None:
                    cur.openlm_avg = c.openlm_avg
                if not cur.params_b:
                    cur.params_b = c.params_b
                cur.extra.update(c.extra)
            else:
                pool[key] = c

    for c in pool.values():
        _resolve_params(c)
        # Recompute rather than trusting a stored label: `models_200.yaml` is this
        # script's own output, so a stale family would otherwise persist forever.
        c.family = _canon_family(c.mid, fallback=c.family if "builder-raw" in c.sources else "")

    # Collapse cross-org republications, keeping the first-party spelling.
    best: dict[str, Candidate] = {}
    for c in pool.values():
        nk = _name_key(c.mid)
        cur = best.get(nk)
        if cur is None:
            best[nk] = c
            continue
        # lowercase the id for the tiebreak so a mis-cased republication
        # (`StabilityAI/stablecode-3b`) cannot outrank the canonical repo
        def rank(x: Candidate) -> tuple:
            return (
                org_rank(x.org),
                0 if x.sources & {"models_200", "edullm_100", "builder-raw"} else 1,
                0 if x.org.islower() else 1,
                x.mid.lower(),
            )

        keep, drop = (cur, c) if rank(cur) <= rank(c) else (c, cur)
        keep.sources |= drop.sources
        if keep.openlm_avg is None:
            keep.openlm_avg = drop.openlm_avg
        best[nk] = keep
    return {c.mid.lower(): c for c in best.values()}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

# Locked pre-registration anchor pool (200-Model Run §2.3): the pilot ∩ 200
# carryover that links this run's latent scale to the frozen 82-model baseline.
# ≥25 must survive, so these are force-taken ahead of every cap.
ANCHORS = [
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-1124-7B",
    "google/gemma-2-2b",
    "ibm-granite/granite-3.0-2b-base",
    "ibm-granite/granite-3.1-2b-base",
    "ibm-granite/granite-3.1-2b-instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stablelm-zephyr-3b",
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-3B-Base",
    "openbmb/MiniCPM-2B-sft-bf16",
    "nvidia/Nemotron-Mini-4B-Instruct",
    "h2oai/h2o-danube2-1.8b-base",
]

# OLMo is the house family; keep its 1B and 7B rungs regardless of caps.
MUST_INCLUDE = [
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-0425-1B-DPO",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-7B-0724-hf",
]

BIN_EDGES = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0001]
BIN_LABELS = ["0.2-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0", "6.0-7.0"]
BIN_TARGETS = {
    "0.2-1.0": 12,
    "1.0-2.0": 55,
    "2.0-3.0": 40,
    "3.0-4.0": 42,
    "4.0-5.0": 20,
    "5.0-6.0": 18,
    "6.0-7.0": 13,
}

MAX_SIZES_PER_FAMILY = 3
MAX_PER_FAMILY = 5
# The org cap is only a secondary diversity guard — labs that ship several
# genuinely distinct families (Qwen1.5/2/2.5/3) should not be starved by it,
# since the family cap and the (family, size, tuning) identity rule already
# prevent redundancy.
MAX_PER_ORG = 10
ORG_CAP_OVERRIDES = {"Qwen": 16, "allenai": 10, "ibm-granite": 8, "microsoft": 8, "meta-llama": 8}
FAMILY_CAP_OVERRIDES = {"OLMo2": 6, "OLMo": 3}


def bin_of(p: float) -> str:
    for i, label in enumerate(BIN_LABELS):
        if BIN_EDGES[i] <= p < BIN_EDGES[i + 1]:
            return label
    return BIN_LABELS[-1]


class Selector:
    def __init__(self, pool: list[Candidate]):
        self.pool = pool
        self.by_id = {c.mid: c for c in pool}
        self.chosen: list[Candidate] = []
        self.chosen_ids: set[str] = set()
        self.orgs: Counter = Counter()
        self.fams: Counter = Counter()
        self.fam_sizes: dict[str, set[float]] = defaultdict(set)
        self.identities: set[tuple[str, float, str]] = set()
        self.max_sizes = MAX_SIZES_PER_FAMILY
        self.max_fam = MAX_PER_FAMILY
        self.max_org = MAX_PER_ORG

    def identity(self, c: Candidate) -> tuple[str, float, str]:
        return (c.family, round(c.params_b, 2), tuning_of(c.mid))

    def eligible(self, c: Candidate) -> bool:
        if c.mid in self.chosen_ids:
            return False
        # base vs instruct at one size is a wanted pair; the same tuning twice is not
        if self.identity(c) in self.identities:
            return False
        # A third-party finetune recipe contributes a data point or two, not a
        # size ladder; only first-party families earn the full allowance.
        third_party = org_rank(c.org) == 1
        org_cap = ORG_CAP_OVERRIDES.get(c.org, 2 if third_party else self.max_org)
        if self.orgs[c.org] >= org_cap:
            return False
        fam_cap = FAMILY_CAP_OVERRIDES.get(c.family, 2 if third_party else self.max_fam)
        if self.fams[c.family] >= fam_cap:
            return False
        size_cap = 2 if third_party else self.max_sizes
        size = round(c.params_b, 2)
        if size not in self.fam_sizes[c.family] and len(self.fam_sizes[c.family]) >= size_cap:
            return False
        return True

    def take(self, c: Candidate) -> None:
        self.chosen.append(c)
        self.chosen_ids.add(c.mid)
        self.orgs[c.org] += 1
        self.fams[c.family] += 1
        self.fam_sizes[c.family].add(round(c.params_b, 2))
        self.identities.add(self.identity(c))

    def force(self, mid: str) -> bool:
        c = self.by_id.get(mid)
        if c is None or mid in self.chosen_ids:
            return False
        self.take(c)
        return True

    def score(self, c: Candidate) -> tuple:
        return (
            0 if "olmo" in c.low else 1,
            org_rank(c.org),
            0 if is_family_home(c) else 1,
            0 if c.sources & {"models_200", "edullm_100"} else 1,
            self.orgs[c.org],
            self.fams[c.family],
            -(c.openlm_avg or 0.0),
            c.mid,
        )

    def fill_bin(self, label: str, need: int, tiers: tuple[int, ...]) -> int:
        got = 0
        while got < need:
            cands = [
                c
                for c in self.pool
                if bin_of(c.params_b) == label and tier_of(c) in tiers and self.eligible(c)
            ]
            if not cands:
                break
            cands.sort(key=self.score)
            self.take(cands[0])
            got += 1
        return got

    def have(self, label: str) -> int:
        return sum(1 for c in self.chosen if bin_of(c.params_b) == label)


def select(pool: list[Candidate], target: int = TARGET_N) -> tuple[list[Candidate], list[str]]:
    """Fill the roster in strict priority order.

    The locked size mix (BIN_TARGETS) is treated as a *preference*, not a
    constraint: once language-targeted filler and mirrors are excluded, the
    reputable English pool is bimodal (dense at <=2B and at 7B, near-empty at
    5-6B), so a bin that cannot be filled yields its slots to bins that can.
    """
    sel = Selector(pool)
    notes: list[str] = []

    for mid in ANCHORS:
        if not sel.force(mid):
            notes.append(f"anchor MISSING from pool: {mid}")
    n_anchor = len(sel.chosen)
    for mid in MUST_INCLUDE:
        sel.force(mid)

    # Scarcest bins first so the thin middle is not crowded out by the 7B glut.
    scarce_first = ["5.0-6.0", "4.0-5.0", "2.0-3.0", "3.0-4.0", "1.0-2.0", "0.2-1.0", "6.0-7.0"]
    for label in scarce_first:
        sel.fill_bin(label, max(0, BIN_TARGETS[label] - sel.have(label)), (1, 2))

    # Redistribute unfillable slots: relax diversity caps, mid-sizes first, and
    # only then let the 6-7B tail grow past its target.
    mid_bins = ["2.0-3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0", "1.0-2.0", "0.2-1.0"]
    for sizes, fam, org in ((4, 6, 8), (5, 7, 10), (6, 9, 14)):
        if len(sel.chosen) >= target:
            break
        sel.max_sizes, sel.max_fam, sel.max_org = sizes, fam, org
        for label in mid_bins + ["6.0-7.0"]:
            if len(sel.chosen) >= target:
                break
            sel.fill_bin(label, target - len(sel.chosen), (1, 2))

    n_general = len(sel.chosen)

    # Specialists are filler only — they enter last, under tight caps so a
    # coder/math family cannot colonise the roster.
    if len(sel.chosen) < target:
        sel.max_sizes, sel.max_fam, sel.max_org = 2, 2, 3
        for label in mid_bins + ["6.0-7.0"]:
            if len(sel.chosen) >= target:
                break
            sel.fill_bin(label, target - len(sel.chosen), (3,))

    notes.append(f"anchors force-taken: {n_anchor}/{len(ANCHORS)}")
    notes.append(f"general-purpose (tier 1-2): {n_general}")
    notes.append(f"specialist filler (tier 3): {len(sel.chosen) - n_general}")
    if len(sel.chosen) < target:
        notes.append(
            f"SHORT by {target - len(sel.chosen)}: reputable English pool exhausted "
            f"(widen REPUTABLE_ORGS or accept the smaller roster)"
        )
    sel.chosen.sort(key=lambda c: (c.params_b, c.mid))
    return sel.chosen, notes


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

RUNTIME_HEADER = """\
# Curated {n} open-source checkpoints in [0.2B, 7.0B] for the 200-model sweep.
#
# Curation rules (see Experiments/models_200/curate_models_200.py):
#   * English-first. Language-targeted checkpoints (Polish/German/Japanese/
#     Korean/Russian/Spanish/... only) are excluded; genuinely multilingual
#     models that include English are kept.
#   * vLLM-clean. Excludes every family that failed the P6 sweep (mamba,
#     openelm, gemma-3/-it, classic falcon, pythia, opt, gpt-neo/-j, bloomz,
#     cerebras, mpt, internlm, danube3, tinyllama, phi-2, redpajama, rwkv,
#     zamba, olmoe). Falcon3 and bloom (non-z) are kept.
#   * No mirrors. unsloth/* and other byte-identical re-uploads are dropped so
#     the IRT matrix cannot contain duplicate persons.
#   * No merges, quantisations, roleplay tunes, or <think>-trace models.
#   * Family diversity: one checkpoint per (family, size, tuning). Matched
#     base<->instruct pairs at the SAME size are deliberately kept — they are
#     the cleanest source of scaffolding variance at fixed correctness
#     (200-Model Run Pre-Registration §2.2).
#   * Domain specialists (code/math) are filler only, added last.
#   * All {n_anchor} pre-registered common-person anchors are force-included (§2.3).
#
# Registry (models_registry.py) derives chat/gated/backend flags as usual.

defaults:
  dtype: bfloat16
  trust_remote_code: true
  # max_model_len intentionally unset: each model resolves to
  # min(its declared window, max_model_len_cap). A flat 4096 both over-requested
  # on 1024/2048-position checkpoints (vLLM rejects that at init) and starved
  # FRQ - TutorEval prompts reach ~10k tokens, collapsing the answer budget to
  # the 256-token floor even for models supporting 32k+.
  tp: 1
  apply_chat_template: null
  scoring_method: loglikelihood

models:
"""

TUTOR_HEADER = """\
# TutorBench response-generation manifest — the curated {n} "common person" models.
#
# Person (row) roster of the MIRT response matrix: each of these checkpoints
# answers every TutorBench scenario, and the between-model spread is what
# identifies the item parameters. Selection is the same curated set as
# AdaptiveTesting/Inputs/Models/models_200.yaml (built by
# Experiments/models_200/curate_models_200.py) — English-first, vLLM-clean, no
# mirrors/merges/quantisations, matched base<->instruct pairs kept, domain
# specialists as filler only, and all {n_anchor} pre-registered anchors included.
#
# `defaults` apply to every model; a per-model key overrides them. registry.py
# derives apply_chat_template / gated / backend / architecture / max_model_len
# from the model id, so most entries need only `id`.
#
# NOTE (numeric fields): keep temperature/top_p/repetition_penalty as bare
# numbers — a trailing comma (e.g. `1.1,`) makes YAML parse a *string* and the
# loader will reject it. `params_b` is metadata only (kept in ModelSpec.extra).

defaults:
  max_model_len_cap: 32768      # hard ceiling; clamped down to the model's own limit
  max_new_tokens: 4096          # output budget (bounds base-model rambling)
  tensor_parallel_size: 1       # one <=7B model per B200; parallelism is across models
  scoring_method: generate
  temperature: 0.0              # deterministic decode for construct validity
  top_p: 1.0                    # inert at temperature 0
  repetition_penalty: 1.1       # low + uniform; still shifts argmax at temp 0
  seed: 0

models:
"""

# Per-model overrides the heuristics would get wrong. Carried over from the
# hand-maintained rosters so the generated files keep the fixes.
OVERRIDES: dict[str, dict] = {
    "Qwen/Qwen3-0.6B": {"apply_chat_template": True, "enable_thinking": False},
    "Qwen/Qwen3-1.7B": {"apply_chat_template": True, "enable_thinking": False},
    "Qwen/Qwen3-4B": {"apply_chat_template": True, "enable_thinking": False},
    "HuggingFaceTB/SmolLM3-3B": {"apply_chat_template": True, "enable_thinking": False},
}
OVERRIDE_NOTES = {
    "Qwen/Qwen3-0.6B": "bare name is the instruct/thinking hybrid; -Base is the base model",
    "Qwen/Qwen3-1.7B": "bare name is the instruct/thinking hybrid; -Base is the base model",
    "Qwen/Qwen3-4B": "bare name is the instruct/thinking hybrid; -Base is the base model",
    "HuggingFaceTB/SmolLM3-3B": "bare name is the instruct checkpoint; -Base is the base model",
}


def _fmt_p(p: float) -> str:
    return f"{p:g}"


def write_runtime_yaml(models: list[Candidate], path: Path) -> None:
    lines = [RUNTIME_HEADER.format(n=len(models), n_anchor=len(ANCHORS)).rstrip("\n")]
    cur = None
    for c in models:
        b = bin_of(c.params_b)
        if b != cur:
            lines.append(f"  # --- {b} B ---")
            cur = b
        note = OVERRIDE_NOTES.get(c.mid)
        if note:
            lines.append(f"  # {note}")
        parts = [f"id: {c.mid}", f"params_b: {_fmt_p(c.params_b)}", f"family: {c.family}", f"arch: {c.arch}"]
        for k, v in OVERRIDES.get(c.mid, {}).items():
            parts.append(f"{k}: {str(v).lower()}")
        lines.append("  - {" + ", ".join(parts) + "}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tutor_yaml(models: list[Candidate], path: Path) -> None:
    lines = [TUTOR_HEADER.format(n=len(models), n_anchor=len(ANCHORS)).rstrip("\n")]
    cur = None
    for c in models:
        b = bin_of(c.params_b)
        if b != cur:
            lines.append(f"  # --- {b} B ---")
            cur = b
        note = OVERRIDE_NOTES.get(c.mid)
        if note:
            lines.append(f"  # {note}")
        lines.append(f"  - id: {c.mid}")
        lines.append(f"    params_b: {_fmt_p(c.params_b)}")
        for k, v in OVERRIDES.get(c.mid, {}).items():
            lines.append(f"    {k}: {str(v).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recovery_candidates(chosen: list[Candidate]) -> list[Candidate]:
    """Models the vLLM blocklist is costing us.

    Re-runs the pipeline with the blocklist off and returns what it would add.
    Most of the blocklist is a stale artefact: the previous builder wrote a flat
    `max_model_len: 4096`, which vLLM rejects at init on the 1024/2048-position
    checkpoints that dominate the list. That default is now unset, so these are
    worth one load attempt each before being written off.
    """
    global ALLOW_VLLM_BROKEN
    before = ALLOW_VLLM_BROKEN
    try:
        ALLOW_VLLM_BROKEN = True
        pool = build_pool()
        kept = [c for c in pool.values() if not classify(c)[0]]
        alt, _ = select(kept)
    finally:
        ALLOW_VLLM_BROKEN = before
    have = {c.mid for c in chosen}
    return [c for c in alt if c.mid not in have and is_vllm_blocked(c.mid)]


def write_report(
    chosen: list[Candidate],
    dropped: dict[str, list[tuple[str, str]]],
    notes: list[str],
    path: Path,
) -> None:
    fams = Counter(c.family for c in chosen)
    orgs = Counter(c.org for c in chosen)
    tun = Counter(tuning_of(c.mid) for c in chosen)
    flagged = [(c, *confidence_of(c)) for c in chosen]
    review = [(c, lvl, why) for c, lvl, why in flagged if lvl != "high"]
    recovery = _recovery_candidates(chosen)

    L: list[str] = []
    L.append("# Curated model roster — handoff report")
    L.append("")
    L.append(f"**{len(chosen)} checkpoints, {PARAM_MIN}–{PARAM_MAX}B.** Generated by")
    L.append("`AdaptiveTesting/Experiments/models_200/curate_models_200.py`. Consumed by")
    L.append("`AdaptiveTesting/Inputs/Models/models_200.yaml` (loglikelihood/MCQ) and")
    L.append("`eduLLM-Evals/models_200.yaml` (generate/TutorBench). The two rosters are identical.")
    L.append("")
    L.append("## Composition")
    L.append("")
    L.append(f"- families: **{len(fams)}**, orgs: **{len(orgs)}**")
    L.append(f"- tuning: **{tun.get('base', 0)} base / {tun.get('instruct', 0)} instruct**")
    L.append(f"- anchors: **{sum(1 for c in chosen if c.mid in set(ANCHORS))}/{len(ANCHORS)}** "
             "pre-registered common-person anchors present")
    L.append("")
    L.append("| size bin | selected | pre-registered target |")
    L.append("|---|---|---|")
    for label in BIN_LABELS:
        got = sum(1 for c in chosen if bin_of(c.params_b) == label)
        L.append(f"| {label} B | {got} | {BIN_TARGETS[label]} |")
    L.append("")
    L.append("The 5–6B bin is empty and 6–7B overshoots because, once language-targeted")
    L.append("filler and mirrors are removed, the reputable English pool is genuinely")
    L.append("bimodal — dense at <=2B and at 7B, near-empty in between. The locked size mix")
    L.append("was achievable only by admitting monolingual checkpoints to pad the middle.")
    L.append("")

    L.append("## TO VERIFY (handoff checklist)")
    L.append("")
    L.append("1. **Hub resolution** — none of these ids has been checked against the")
    L.append("   HuggingFace Hub. They come from CSV banks (ATLAS / OpenLM leaderboard), so a")
    L.append("   renamed, removed, or newly-gated repo would only surface at load time.")
    L.append("2. **Gated access** — confirm the token can read the gated orgs")
    L.append("   (`meta-llama`, `mistralai`, `google/gemma`).")
    L.append(f"3. **Review the {len(review)} flagged entries below** — these came from the")
    L.append("   candidate banks rather than a hand-maintained roster.")
    L.append(f"4. **Smoke-test the {len(recovery)} recovery candidates below** to grow the roster.")
    L.append("")

    L.append(f"### Flagged for review ({len(review)})")
    L.append("")
    L.append("| model | params_b | family | flag | why |")
    L.append("|---|---|---|---|---|")
    for c, lvl, why in sorted(review, key=lambda t: (t[1], t[0].params_b, t[0].mid)):
        L.append(f"| `{c.mid}` | {_fmt_p(c.params_b)} | {c.family} | {lvl} | {why} |")
    L.append("")

    L.append(f"### Recovery candidates — likely add-backs ({len(recovery)})")
    L.append("")
    L.append("Excluded only by the vLLM blocklist in `build_models_200.py`. That blocklist")
    L.append("cites a `backfill_vllm_skips/models_all.txt` skip list that is no longer in the")
    L.append("repo, and its members are overwhelmingly sub-4096-context checkpoints — which")
    L.append("is exactly what the old flat `max_model_len: 4096` default would reject at init")
    L.append("with a ModelConfig ValidationError. That default is now unset, so each of these")
    L.append("deserves one short load attempt. They are also the pilot's low-ability floor,")
    L.append("whose loss the pre-registration flags as validity risk #4.")
    L.append("")
    L.append("| model | params_b | family | blocklist pattern |")
    L.append("|---|---|---|---|")
    for c in sorted(recovery, key=lambda c: (c.params_b, c.mid)):
        L.append(f"| `{c.mid}` | {_fmt_p(c.params_b)} | {c.family} | {is_vllm_blocked(c.mid)} |")
    L.append("")
    L.append("Genuinely unsupported by vLLM regardless of context window, and needing the")
    L.append("`hf_fallback` backend rather than a retry: OpenELM (ships no tokenizer),")
    L.append("Mamba / RWKV / Zamba (state-space), GPT-Neo (`GPTNeoForCausalLM` is absent from")
    L.append("vLLM though `GPTNeoX` is present), and any MoE.")
    L.append("")

    L.append("## Exclusions applied")
    L.append("")
    L.append("| reason | dropped |")
    L.append("|---|---|")
    for reason, items in sorted(dropped.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {reason} | {len(items)} |")
    L.append("")

    L.append("## Roster")
    L.append("")
    for label in BIN_LABELS:
        ms = [c for c in chosen if bin_of(c.params_b) == label]
        if not ms:
            continue
        L.append(f"### {label} B — {len(ms)} models")
        L.append("")
        L.append("| model | params_b | tuning | family | anchor |")
        L.append("|---|---|---|---|---|")
        anchors = set(ANCHORS)
        for c in sorted(ms, key=lambda c: (c.params_b, c.mid)):
            L.append(
                f"| `{c.mid}` | {_fmt_p(c.params_b)} | {tuning_of(c.mid)} | "
                f"{c.family} | {'yes' if c.mid in anchors else ''} |"
            )
        L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    global ALLOW_VLLM_BROKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_N)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--allow-vllm-broken",
        action="store_true",
        help="re-admit the families that fail under vLLM (valid for the CPU/HF sweep)",
    )
    args = ap.parse_args()
    ALLOW_VLLM_BROKEN = args.allow_vllm_broken

    pool = build_pool()
    kept: list[Candidate] = []
    dropped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in pool.values():
        reason, detail = classify(c)
        if reason:
            dropped[reason].append((c.mid, detail))
        else:
            kept.append(c)

    chosen, notes = select(kept, args.target)

    print(f"pool: {len(pool)} unique candidates")
    for reason, items in sorted(dropped.items(), key=lambda kv: -len(kv[1])):
        print(f"  dropped {reason:<18} {len(items)}")
    print(f"eligible after exclusions: {len(kept)}")
    print(f"selected: {len(chosen)}")
    for n in notes:
        print(f"  {n}")
    print("\nbins:")
    for label in BIN_LABELS:
        got = sum(1 for c in chosen if bin_of(c.params_b) == label)
        print(f"  {label:<9} {got:>3} (target {BIN_TARGETS[label]})")
    fams = Counter(c.family for c in chosen)
    print(f"\nfamilies: {len(fams)}  orgs: {len(Counter(c.org for c in chosen))}")
    print(f"tiers: {dict(Counter(tier_of(c) for c in chosen))}")
    tun = Counter(tuning_of(c.mid) for c in chosen)
    print(f"tuning: {dict(tun)}")

    if not args.dry_run:
        write_runtime_yaml(chosen, RUNTIME_YAML)
        write_tutor_yaml(chosen, TUTOR_OUT)
        write_report(chosen, dropped, notes, REPORT_OUT)
        print(f"\nwrote {RUNTIME_YAML}")
        print(f"wrote {TUTOR_OUT}")
        print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
