"""Classify merged candidates: org, canonical family, and exclusion flags 3a-3g.

v2 fixes vs v1:
  * `Model-Stock` / `model_stock` is a mergekit method, not finance -> moved to 3b
  * `task_arithmetic` no longer counts as math
  * word-boundaries on limo / ziya / allam / belle (were matching slimorca,
    Maziyar, ...llama+allam..., etc.)
  * naming-hygiene markers (v0.N, test, experiment, hash suffix) are suppressed
    for known-reputable orgs so mistralai/Mistral-7B-Instruct-v0.2 stays clean
  * non-English split into hard (language-specific tune) vs soft (CJK vendor
    whose base checkpoints are standard on English leaderboards)
  * new quality gates: f_ablation (hyper-parameter grid dumps) and f_orgspam

Writes _scratch_atlas/candidates_flagged.csv
"""

from __future__ import annotations

import os
import re

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")

cand = pd.read_csv(os.path.join(OUT, "candidates.csv"))

# ------------------------------------------------------------------ params fill
_PARAM_RE = re.compile(r"(?<![a-z0-9.])(\d{1,3}(?:[.,x]\d{1,2})?)\s*[bB](?![a-z0-9])")
_PARAM_M_RE = re.compile(r"(?<![a-z0-9.])(\d{2,4})\s*[mM](?![a-z0-9])")


def infer_params(model_id: str) -> float | None:
    name = model_id.split("/")[-1]
    m = _PARAM_RE.search(name)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
            if 0.05 <= v <= 1000:
                return v
        except ValueError:
            pass
    m = _PARAM_M_RE.search(name)
    if m:
        try:
            v = float(m.group(1)) / 1000.0
            if 0.05 <= v <= 10:
                return v
        except ValueError:
            pass
    return None


cand["params_source"] = "pool"
missing = cand["params_b"].isna()
cand.loc[missing, "params_b"] = cand.loc[missing, "model"].map(infer_params)
cand.loc[missing & cand["params_b"].notna(), "params_source"] = "inferred_from_name"
cand.loc[cand["params_b"].isna(), "params_source"] = "unknown"

# ==============================================================================
# 3a  NON-ENGLISH / LANGUAGE-SPECIFIC
# ==============================================================================
LANG_WORD = {
    "german": r"german|deutsch|sauerkraut|discolm|leo-?hessianai|em_?german|kafkalm|brezn|bueble|occiglot-?7b-?de|hammerlm|(?:^|[-_])de(?:[-_.]|$)",
    "polish": r"polish|polka|bielik|polanka|trurl|(?:^|[-_])qra-|pllum|apt3|curie-7b",
    "russian": r"russian|rugpt|(?:^|[-_])ru-?gpt|vikhr|saiga|rulm|sberbank|ai-forever|yandex|ruadapt|gigachat|(?:^|[-_])ru(?:[-_.]|$)",
    "japanese": r"japanese|japan|(?:^|[-_])jpn(?:[-_.]|$)|llm-?jp|open-?calm|(?:^|[-_])rinna|elyza|swallow|youri|weblab|karasu|shisa|nekomata|sarashina|calm2|calm3|stockmark|tanuki|rakuten|houou|plamo|abeja|(?:^|[-_])ja(?:[-_.]|$)",
    "korean": r"korean|korea|polyglot-?ko|(?:^|[-_])eeve|kanana|kullm|koalpaca|(?:^|[-_])ko-?en|open-?ko|kogpt|(?:^|[-_])komt|synatra|solar-?ko|yanolja|hyperclova|kullama|(?:^|[-_])ko-?llama|ko-?platypus|kolong|sft-?ko|(?:^|[-_])krx|(?:^|[-_])ko(?:[-_.]|$)",
    "chinese": r"chinese|zh-?cn|zhongwen|baichuan|chatglm|glm-?4-?9b|yuan2|skywork|xverse|telechat|orion-?14b|hunyuan|internlm|firefly|(?:^|[-_])belle(?:[-_.]|$)|(?:^|[-_])ziya|linly|moss-?moon|tigerbot|colossal|bluelm|aquila|cpm-?bee|silk-?road|shenzhi|sunsimiao|huatuo|zhihu|panda-?llm|chinese-?alpaca|chinese-?llama|(?:^|[-_])zh(?:[-_.]|$)",
    "spanish": r"spanish|espanol|español|castellano|hispano|salamandra|(?:^|[-_])lince|neurona|bertin|(?:^|[-_])beto-|clibrain|maritaca|latxa|(?:^|[-_])es(?:[-_.]|$)",
    "italian": r"italian|italia|(?:^|[-_])italo|minerva|llamantino|maestrale|cerbero|zefiro|modello|(?:^|[-_])fauno|camoscio|(?:^|[-_])anita|(?:^|[-_])velvet|(?:^|[-_])ita(?:[-_.]|$)",
    "french": r"french|francais|français|croissant|vigogne|vigostral|(?:^|[-_])claire-?7b|(?:^|[-_])lucie|(?:^|[-_])mistral-?fr|occiglot-?7b-?fr|bofenghuang|pantagruel|(?:^|[-_])fr(?:[-_.]|$)",
    "portuguese": r"portuguese|portugues|português|brazil|brasil|cabrita|(?:^|[-_])sabia|gervasio|gloria|(?:^|[-_])bode-?7b|(?:^|[-_])pt(?:[-_.]|$)",
    "dutch": r"dutch|nederlands|geitje|fietje|(?:^|[-_])boreas|robbert|tulu-?nl|(?:^|[-_])nl(?:[-_.]|$)",
    "swedish": r"swedish|svenska|gpt-?sw3|ai-?sweden|bellman|(?:^|[-_])sv(?:[-_.]|$)",
    "finnish": r"finnish|(?:^|[-_])suomi|turkunlp|gpt3-?finnish|(?:^|[-_])poro-?|(?:^|[-_])ahma-|(?:^|[-_])fi(?:[-_.]|$)",
    "norwegian": r"norwegian|(?:^|[-_])norsk|nb-?gpt|norallm|normistral|(?:^|[-_])nb(?:[-_.]|$)",
    "danish": r"danish|(?:^|[-_])dansk|(?:^|[-_])munin|danskgpt|(?:^|[-_])da(?:[-_.]|$)",
    "czech": r"czech|(?:^|[-_])cesky|csmpt|(?:^|[-_])cs(?:[-_.]|$)",
    "turkish": r"turkish|turkce|türk|kanarya|trendyol|cosmosllama|turkcell|ytu-?ce|(?:^|[-_])tr(?:[-_.]|$)",
    "arabic": r"arabic|(?:^|[-_])arabi|(?:^|[-_])jais|(?:^|[-_])allam(?:[-_.]|$)|acegpt|(?:^|[-_])silma|(?:^|[-_])noon-?7b|arabert|(?:^|[-_])fanar|(?:^|[-_])ar(?:[-_.]|$)",
    "hindi_indic": r"hindi|(?:^|[-_])indic|sarvam|airavata|krutrim|navarasa|open-?hathi|(?:^|[-_])ambari|(?:^|[-_])tamil|telugu|bengali|marathi|(?:^|[-_])nanda(?:[-_.]|$)|(?:^|[-_])hi(?:[-_.]|$)",
    "indonesian": r"indonesia|(?:^|[-_])indo-?|(?:^|[-_])komodo|(?:^|[-_])merak|(?:^|[-_])cendol|sea-?lion|sahabat|(?:^|[-_])kancil|bahasa|(?:^|[-_])id(?:[-_.]|$)",
    "vietnamese": r"vietnam|(?:^|[-_])viet-?|vinallama|phogpt|vbd-?llama|ghost-?7b|(?:^|[-_])vi(?:[-_.]|$)",
    "thai": r"(?:^|[-_])thai|typhoon|openthaigpt|wangchan|(?:^|[-_])sailor|(?:^|[-_])th(?:[-_.]|$)",
    "hebrew": r"hebrew|dictalm|ivrit|(?:^|[-_])he(?:[-_.]|$)",
    "greek": r"greek|meltemi|(?:^|[-_])el(?:[-_.]|$)",
    "hungarian": r"hungarian|magyar|(?:^|[-_])puli-|(?:^|[-_])hu(?:[-_.]|$)",
    "romanian": r"romanian|romania|rollama|rogpt|(?:^|[-_])ro(?:[-_.]|$)",
    "ukrainian": r"ukrain|(?:^|[-_])uk(?:[-_.]|$)",
    "persian": r"persian|(?:^|[-_])farsi|(?:^|[-_])maral|(?:^|[-_])dorna|(?:^|[-_])fa(?:[-_.]|$)",
    "multilingual_nonen": r"multilingual|polyglot|occiglot|eurollm|(?:^|[-_])aya-|glot500|(?:^|[-_])xglm|(?:^|[-_])mgpt|bloomz|(?:^|[-_])mt0-|umt5|(?:^|[-_])nllb|teuken",
}
LANG_ORG = {
    "ai-forever": "russian", "sberbank-ai": "russian", "ai-sage": "russian",
    "vikhrmodels": "russian", "ilyagusev": "russian", "msuai": "russian",
    "t-tech": "russian", "yandexgpt": "russian", "ruadapt": "russian",
    "rinna": "japanese", "elyza": "japanese", "cyberagent": "japanese",
    "llm-jp": "japanese", "pfnet": "japanese", "tokyotech-llm": "japanese",
    "sbintuitions": "japanese", "rakuten": "japanese", "lightblue": "japanese",
    "aixsatoshi": "japanese", "shisa-ai": "japanese", "augmxnt": "japanese",
    "weblab-geniac": "japanese", "abeja": "japanese", "stockmark": "japanese",
    "alfredplpl": "japanese", "hotchpotch": "japanese", "nitky": "japanese",
    "beomi": "korean", "maywell": "korean", "yanolja": "korean",
    "nlpai-lab": "korean", "davidkim205": "korean", "heegyu": "korean",
    "kyujinpy": "korean", "megastudy": "korean", "naver-hyperclovax": "korean",
    "krevas": "korean", "hannayeoniee": "korean", "jiwoochris": "korean",
    "gangyeolkim": "korean", "ldcc": "korean", "42dot": "korean",
    "sionic-ai": "korean", "chihoonlee10": "korean", "hyeogi": "korean",
    "jaeyeon-kang": "korean", "wkshin89": "korean", "gaonai": "korean",
    "ai-human-lab": "korean", "sanghwa-na": "korean", "cocoirun": "korean",
    "shleeeee": "korean", "gwonny": "korean", "blueapple8259": "korean",
    "jjourney1125": "korean", "youjunhyeok": "korean", "spow12": "korean",
    "werty1248": "korean", "x2bee": "korean", "lgai-exaone": "korean",
    "kakaocorp": "korean", "kt-ai": "korean", "juhwanlee": "korean",
    "changgil": "korean", "skyorbis": "korean", "korabbit": "korean",
    "hanryeoniee": "korean", "sungwoo1": "korean", "haeun161": "korean",
    "orionstarai": "chinese", "tigerresearch": "chinese", "yeungnlp": "chinese",
    "bellegroup": "chinese", "idea-ccnl": "chinese", "shenzhi-wang": "chinese",
    "zhengr": "chinese", "wangrongsheng": "chinese", "linly-ai": "chinese",
    "fnlp": "chinese", "hfl": "chinese", "ziqingyang": "chinese",
    "duxiaoman-di": "chinese", "tencent": "chinese", "iic": "chinese",
    "modelscope": "chinese", "silk-road": "chinese", "mediatek-research": "chinese",
    "taide": "chinese", "twinkle-ai": "chinese", "ckip-joint": "chinese",
    "bsc-lt": "spanish", "plantl-gob-es": "spanish", "projecte-aina": "spanish",
    "clibrain": "spanish", "iker": "spanish", "hitz": "spanish",
    "sapienzanlp": "italian", "mii-llm": "italian", "swap-uniba": "italian",
    "detomo": "italian", "raicrits": "italian", "almawave": "italian",
    "deepmount00": "italian", "croissantllm": "french", "openllm-france": "french",
    "bofenghuang": "french", "jpacifico": "french",
    "occiglot": "multilingual_nonen", "utter-project": "multilingual_nonen",
    "cohereforai": "multilingual_nonen", "coherelabs": "multilingual_nonen",
    "ai-sweden-models": "swedish", "turkunlp": "finnish", "lumiopen": "finnish",
    "norallm": "norwegian", "nb-ai-lab": "norwegian", "bineric": "norwegian",
    "danish-foundation-models": "danish", "sdadas": "polish", "speakleash": "polish",
    "voicelab": "polish", "eryk-mazus": "polish", "azurro": "polish",
    "trendyol": "turkish", "ytu-ce-cosmos": "turkish", "kocdigital": "turkish",
    "core42": "arabic", "inceptionai": "arabic", "freedomintelligence": "arabic",
    "silma-ai": "arabic", "malhajar": "turkish", "sarvamai": "hindi_indic",
    "ai4bharat": "hindi_indic", "krutrim-ai-labs": "hindi_indic",
    "abhinand": "hindi_indic", "telekom": "german", "vagosolutions": "german",
    "dfki-nlp": "german", "malteos": "german", "seedboxai": "german",
    "flozi00": "german", "jphme": "german", "disco-research": "german",
    "aisingapore": "indonesian", "gotocompany": "indonesian", "azale-ai": "indonesian",
    "scb10x": "thai", "airesearch": "thai", "vilm": "vietnamese",
    "vinai": "vietnamese", "vietai": "vietnamese", "ghost-x": "vietnamese",
    "dicta-il": "hebrew", "yam-peleg": "hebrew", "ilsp": "greek",
    "bramvanroy": "dutch", "robinsmits": "dutch", "bllossom": "korean",
    "openbuddy": "multilingual_nonen", "keyonzeng": "chinese",
}
# mirrors / re-uploads of someone else's weights (duplicate checkpoints)
MIRROR_ORGS = {
    "unsloth", "bartowski", "thebloke", "modelcloud", "mradermacher", "prunaai",
    "lmstudio-community", "neuralmagic", "redhatai", "second-state", "nm-testing",
    "qwp4w3hyb", "solidrust", "openmodels4all",
}
# CJK vendors whose *base* checkpoints are standard English-leaderboard models.
SOFT_CJK_ORGS = {
    "qwen", "01-ai", "internlm", "deepseek-ai", "thudm", "openbmb",
    "baichuan-inc", "xverse", "skywork", "m-a-p", "zai-org", "inclusionai",
}
SOFT_CJK_NAME = re.compile(r"internlm|(?:^|[-_/])qwen|(?:^|[-_/])yi[-_]|minicpm|deepseek", re.I)

# ==============================================================================
# 3b-3g  REGEX FLAG TABLE   (category, marker_label, pattern, hygiene_only)
# ==============================================================================
FLAGS: list[tuple[str, str, str, bool]] = [
    # ---- 3b merges / frankenmodels -------------------------------------------
    ("merge", "mergekit/lazymergekit", r"lazymergekit|mergekit|autogen-?merge|model[-_ ]?stock", False),
    ("merge", "merge", r"(?:^|[-_/])merged?(?:[-_.]|$)|merge[-_]?of|model[-_]?merge|automerger", False),
    ("merge", "slerp", r"slerp", False),
    ("merge", "dare", r"(?:^|[-_])dare(?:[-_]|$)|dare[-_]?ties|dare[-_]?linear", False),
    ("merge", "ties", r"(?:^|[-_])ties(?:[-_.]|$)|ties[-_]?merge", False),
    ("merge", "task_arith/linear", r"(?:^|[-_])linear(?:[-_.]|$)|task[-_]?arith", False),
    ("merge", "passthrough/frankenmerge", r"passthrough|franken|frankenstein|frankenmerge", False),
    ("merge", "moe_merge", r"(?:^|[-_])moe(?:[-_.]|$)|mixture[-_]?of[-_]?expert|[0-9]x[0-9]+b|(?:^|[-_])clown|beyonder|laserxtral|mergemoe", False),
    ("merge", "-mix", r"(?:^|[-_])mix(?:[-_.]|$)|[-_]mixed(?:[-_.]|$)", False),
    ("merge", "experiment", r"experiment|exp[-_]?[0-9]|trial[-_]?[0-9]", True),
    ("merge", "test/debug-name", r"(?:^|[-_])tests?(?:[-_.0-9]|$)|(?:^|[-_])testing(?:[-_.]|$)", True),
    ("merge", "v0.0.x", r"[-_]v0[._]0[._][0-9]|[-_]v0[._][0-9]{1,2}[a-z]?(?:[-_.]|$)", True),
    ("merge", "random_hash_suffix", r"[-_](?=[a-z0-9]{8,12}(?:$|[-_.]))(?=[a-z0-9]*[0-9])(?=[a-z0-9]*[a-z])[a-z0-9]{8,12}(?:$|[-_.])", True),
    ("merge", "cookbook_random_name", r"(?:^|[-_])(?:ognoexperiment|multi[-_]?verse|blur|blend|fusion|chimera|hydra|amalgam)(?:[-_.0-9]|$)", True),
    # ---- 3c quantized / adapter ----------------------------------------------
    ("quant", "gptq", r"gptq", False),
    ("quant", "awq", r"awq", False),
    ("quant", "gguf", r"gguf|ggml", False),
    ("quant", "4bit", r"(?:^|[-_])4[-_]?bit|[-_]4bit|(?:^|[-_])nf4", False),
    ("quant", "8bit", r"(?:^|[-_])8[-_]?bit|[-_]8bit", False),
    ("quant", "bnb", r"(?:^|[-_])bnb|bitsandbytes", False),
    ("quant", "int4/int8", r"(?:^|[-_])int4|(?:^|[-_])int8|w4a16|w8a8|w8a16|(?:^|[-_])fp8|(?:^|[-_])fp4", False),
    ("quant", "exl2", r"exl2|exllama", False),
    ("quant", "lora/adapter", r"(?:^|[-_])q?lora(?:[-_.]|$)|adapter|(?:^|[-_])peft(?:[-_.]|$)", False),
    ("quant", "smoothquant/hqq/aqlm", r"smoothquant|(?:^|[-_])hqq|(?:^|[-_])aqlm|quantized|(?:^|[-_])quant(?:[-_.]|$)", False),
    # ---- 3d roleplay / nsfw / uncensored -------------------------------------
    ("rp_nsfw", "roleplay", r"role[-_]?play|(?:^|[-_])rp(?:[-_.0-9]|$)|(?:^|[-_])erp(?:[-_.]|$)|(?:^|[-_])rpg[-_]", False),
    ("rp_nsfw", "nsfw/erotic", r"(?:^|[-_])nsfw|erotic|(?:^|[-_])lewd|hentai|(?:^|[-_])smut|lustful|horny", False),
    ("rp_nsfw", "uncensored", r"uncensored|unaligned|unfiltered|no[-_]?refusal|(?:^|[-_])amoral|liberated", False),
    ("rp_nsfw", "abliterated", r"abliterat|orthogonal[-_]?ablat|decensor", False),
    ("rp_nsfw", "waifu/anime", r"(?:^|[-_])waifu|(?:^|[-_])anime|senpai|(?:^|[-_])neko|kitsune|catgirl", False),
    ("rp_nsfw", "toxic", r"(?:^|[-_])toxic|(?:^|[-_])evil(?:[-_.]|$)|dark[-_]?idol|degenerate", False),
    ("rp_nsfw", "dolphin", r"dolphin", False),
    ("rp_nsfw", "pygmalion", r"pygmalion|metharme", False),
    ("rp_nsfw", "mythomax/rp-merges", r"mytho|noromaid|kunoichi|silicon[-_]?maid|(?:^|[-_])silicon|estopia|echidna|(?:^|[-_])chronos|thespis|psyfighter|amethyst|(?:^|[-_])toppy|undi95|(?:^|[-_])tavern|(?:^|[-_])character(?:[-_.]|$)|companion|girlfriend|boyfriend", False),
    ("rp_nsfw", "rp_org", r"^(?:undi95|sao10k|thebloke|kooten|lewdiculous|ikarigendo|nothingiisreal|thedrummer|anthracite-org|bluuwhale|hanamizuki|nitral-ai|chaoticneutrals|icefog72|jeiku|resplendentai|922ca|athirdpath|cgato|sanjiwatsuki|test157t|flammenai)/", False),
    # ---- 3e domain: code ------------------------------------------------------
    ("code", "coder", r"(?:^|[-_])coder?(?:[-_.0-9]|$)|codellama|code[-_]?llama|starcoder|santacoder|replit[-_]?code|codegen|magicoder|deepseek[-_]?coder|wizardcoder|codeqwen|codegemma|stable[-_]?code|codestral|codebooga|opencodeinterpreter|artigenz|codeninja|nxcode|speechless[-_]?coder|codefuse|code[-_]?instruct|sql[-_]?coder|sqlcoder|text2sql|programming|python[-_]?code|(?:^|[-_])evmind", False),
    # ---- 3e domain: math ------------------------------------------------------
    ("math", "math", r"(?:^|[-_])math(?:[-_.0-9]|s?$)|mathstral|metamath|numina|mammoth|wizardmath|(?:^|[-_])abel[-_]?7b|deepseek[-_]?math|internlm2[-_]?math|rho[-_]?math|(?:^|[-_])tora(?:[-_.]|$)|openmath|acemath|dart[-_]?math|orca[-_]?math|(?:^|[-_])gsm8k", False),
    # ---- 3e domain: bio / med / legal / finance ------------------------------
    ("domain_bmf", "bio/med", r"(?:^|[-_])bio(?:[-_.]|med|gpt|mistral|llama|nlp|clinical)|(?:^|[-_])med(?!ium)(?:[-_.]|ical|itron|llama|alpaca|palm|mcqa|qa|gemma|phi|-)|clinical|pubmed|biomed|healthcare|(?:^|[-_])health(?:[-_.]|care)|(?:^|[-_])doctor|(?:^|[-_])patient|radiology|meditron|openbio|internist|diagnos|nursing|psychiatr|therapist", False),
    ("domain_bmf", "legal", r"(?:^|[-_])legal|(?:^|[-_])law(?:[-_.]|yer)|(?:^|[-_])juris|contract[-_]?nli|saul[-_]?7b|saullm", False),
    ("domain_bmf", "finance", r"(?:^|[-_])fin(?:ance|ancial|gpt)(?:[-_.]|$|[a-z])|(?:^|[-_])financ|invest(?:ment|or)|(?:^|[-_])trading|stock[-_]?(?:market|price|trading)|bloomberg|accounting|(?:^|[-_])banking(?:[-_.]|$)", False),
    # ---- 3e domain: vision / multimodal --------------------------------------
    ("vision", "vl/vision/mm", r"(?:^|[-_])vl(?:[-_.0-9]|$)|(?:^|[-_])vision|llava|(?:^|[-_])mm(?:[-_.0-9]|$)|idefics|multimodal|(?:^|[-_])vlm(?:[-_.]|$)|bakllava|moondream|paligemma|instructblip|minigpt|cogvlm|internvl|(?:^|[-_])obsidian|(?:^|[-_])image|(?:^|[-_])clip[-_]|siglip|phi[-_]?3[-_]?vision|florence", False),
    ("vision", "audio/speech", r"(?:^|[-_])audio|speech(?!less)|whisper|(?:^|[-_])asr(?:[-_.]|$)|(?:^|[-_])tts(?:[-_.]|$)|(?:^|[-_])voice", False),
    # ---- 3f reasoning / thinking ---------------------------------------------
    ("reasoning", "r1-distill", r"r1[-_]?distill|deepseek[-_]?r1|distill[-_]?r1", False),
    ("reasoning", "qwq", r"(?:^|[-_])qwq(?:[-_.0-9]|$)", False),
    ("reasoning", "thinking", r"think(?:ing|er)?(?:[-_.]|$)|cot[-_]?distill|(?:^|[-_])cot(?:[-_.]|$)|chain[-_]?of[-_]?thought|(?:^|[-_])reflect", False),
    ("reasoning", "reason", r"reason(?:ing|er)?(?:[-_.0-9]|$)|(?:^|[-_])o1(?:[-_.]|$)|marco[-_]?o1|(?:^|[-_])o3(?:[-_.]|$)", False),
    ("reasoning", "openthinker/sky-t1", r"openthinker|sky[-_]?t1|still[-_]?2|simplescaling|bespoke[-_]?stratos|(?:^|[-_])limo(?:[-_.]|$)", False),
    # ---- 3g degenerate --------------------------------------------------------
    ("degenerate", "random/untrained", r"(?:^|[-_])random(?:[-_.]|$)|untrained|(?:^|[-_])dummy|(?:^|[-_])debug|(?:^|[-_])sanity|(?:^|[-_])tiny[-_]?random|(?:^|[-_])scratch(?:[-_.]|$)|placeholder|(?:^|[-_])null(?:[-_.]|$)|(?:^|[-_])broken|deprecated", False),
]

# hyper-parameter / ablation-grid naming (quality gate, not one of 3a-3g).
# STRONG markers mean "this is a training artefact" no matter who published it.
ABLATION_STRONG = re.compile(
    r"[-_][0-9](?:\.[0-9])?e-[0-9]"            # -1e-4  -2e-5
    r"|[-_][0-9]{1,3}ep(?:och)?s?(?:[-_.]|$)"  # -3ep  -5epoch
    r"|[-_][0-9]+epoch"
    r"|[-_](?:ckpt|checkpoint)[-_]?[0-9]*"
    r"|[-_]step[-_]?[0-9]+|[-_][0-9]+steps?(?:[-_.]|$)|intermediate[-_]step"
    r"|[-_](?:bs|lr|wd|seed)[-_]?[0-9]"
    r"|autoredteam|autotrain|no_issue",
    re.I,
)
# WEAK markers are ambiguous - `-0924` is a release date at allenai but a sample
# count at an ablation farm - so they only bite for non-reputable orgs.
ABLATION_WEAK = re.compile(
    r"[-_][0-9]{3,6}(?:[-_.]|$)"
    r"|(?:^|[-_])(?:sample[ds]?|subset|ablation|grid|run)[-_]?[0-9]"
    r"|(?:^|[-_])stage?[0-9]|(?:^|[-_])ss[0-9]",
    re.I,
)

COMPILED = [(c, l, re.compile(p, re.I), h) for c, l, p, h in FLAGS]
LANG_COMPILED = [(lang, re.compile(pat, re.I)) for lang, pat in LANG_WORD.items()]
NON_ASCII = re.compile(r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0370-\u03FF\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# ==============================================================================
# CANONICAL FAMILIES  (ordered; first match wins - explicit bases before tunes)
# ==============================================================================
FAMILIES: list[tuple[str, str]] = [
    ("Qwen3", r"qwen[-_]?3"),
    ("Qwen2.5", r"qwen[-_]?2[._]5|qwen2\.5"),
    ("Qwen2", r"qwen[-_]?2(?![._]5)"),
    ("Qwen1.5", r"qwen[-_]?1[._]5"),
    ("Qwen1", r"(?:^|[-_/])qwen(?![-_]?[0-9])"),
    ("TinyLlama", r"tinyllama|tiny[-_]llama"),
    ("Llama-3.3", r"llama[-_]?3[._]?3|l3\.3"),
    ("Llama-3.2", r"llama[-_]?3[._]?2|l3\.2"),
    ("Llama-3.1", r"llama[-_]?3[._]?1(?![0-9])|l3\.1"),
    ("Llama-3", r"llama[-_]?3(?![._]?[0-9])|(?:^|[-_])l3[-_]"),
    ("Llama-2", r"llama[-_]?2(?![._][0-9])"),
    ("Mixtral", r"mixtral"),
    ("Mistral-Nemo", r"mistral[-_]?nemo|nemo[-_]?12b"),
    ("Mistral", r"mistral"),
    ("Gemma-3", r"gemma[-_]?3(?=[-_.]|$)"),
    ("Gemma-2", r"gemma[-_]?2(?=[-_.]|$)"),
    ("Gemma", r"(?:^|[-_/])gemma|codegemma|recurrentgemma|paligemma|shieldgemma"),
    ("Phi", r"(?:^|[-_/])phi[-_]?[0-9]|phi[-_]?mini|phimoe|(?:^|[-_/])phi(?:[-_.]|$)"),
    ("OLMo", r"(?:^|[-_/])olmo|olmoe"),
    ("StableLM", r"stablelm|stable[-_]?lm|stable[-_]?code|stablebeluga"),
    ("Falcon", r"(?:^|[-_/])falcon"),
    ("Granite", r"granite"),
    ("MiniCPM", r"minicpm|mini[-_]?cpm"),
    ("SmolLM", r"smollm|smol[-_]?llama|smol[-_]?lm"),
    ("Yi", r"(?:^|[-_/])yi[-_]?(?:1[._]5[-_]?)?[0-9]|(?:^|[-_/])yi[-_](?:base|chat|coder)|^01-ai/"),
    ("InternLM", r"internlm"),
    ("DeepSeek", r"deepseek"),
    ("Pythia", r"pythia"),
    ("GPT-2", r"(?:^|[-_/])gpt2|gpt[-_]?2(?![0-9])"),
    ("GPT-Neo/NeoX/J", r"gpt[-_]?neo|gpt[-_]?j|gpt4all|dolly[-_]?v2"),
    ("BLOOM", r"bloom"),
    ("OPT", r"(?:^|[-_/])opt[-_]?[0-9]"),
    ("XGLM", r"xglm"),
    ("EXAONE", r"exaone"),
    ("Salamandra", r"salamandra"),
    ("EuroLLM", r"eurollm"),
    ("Nemotron/Minitron", r"nemotron|minitron|hymba"),
    ("Command-R/Aya", r"command[-_]?r|(?:^|[-_/])aya[-_]"),
    ("Baichuan", r"baichuan"),
    ("ChatGLM/GLM", r"chatglm|(?:^|[-_/])glm[-_]?[0-9]"),
    ("MPT", r"(?:^|[-_/])mpt[-_]?[0-9]"),
    ("RedPajama", r"redpajama"),
    ("Cerebras", r"cerebras|(?:^|[-_/])btlm"),
    ("MobileLLM/OpenELM", r"mobilellm|openelm"),
    ("Danube", r"danube"),
    ("Solar", r"(?:^|[-_/])solar[-_]?[0-9]|solar[-_]?10"),
    ("Zamba/Mamba/RWKV/SSM", r"zamba|(?:^|[-_/])mamba|rwkv|(?:^|[-_/])jamba|griffin|hgrn|(?:^|[-_/])rene"),
    ("Amber/LLM360", r"(?:^|[-_/])amber|crystalcoder|llm360"),
    ("Sheared/Pruned-Llama", r"sheared|(?:^|[-_/])pruned?[-_]|litellama|lite[-_]?llama"),
    ("Cosmo/TinyStories", r"(?:^|[-_/])cosmo[-_]?[0-9]|tinystories|tiny[-_]?gpt"),
    ("OpenLLaMA", r"open[-_]?llama"),
    ("Vicuna", r"vicuna"),
    ("Zephyr", r"zephyr"),
    ("OpenChat/Starling", r"openchat|starling|neural[-_]?chat"),
    ("Nous/Hermes", r"hermes|(?:^|[-_/])nous[-_]"),
    ("Orca/Platypus", r"(?:^|[-_/])orca|platypus|(?:^|[-_/])beluga"),
    ("Tulu", r"(?:^|[-_/])tulu"),
    ("WizardLM/Airoboros", r"wizardlm|airoboros|guanaco|(?:^|[-_/])alpaca"),
    ("Llama-1/other", r"(?:^|[-_/])llama(?![-_]?[0-9])"),
    ("Other-named-lab", r"(?:^|[-_/])fox[-_]?1|index[-_]?1[._]9b|(?:^|[-_/])teuken|(?:^|[-_/])poro|(?:^|[-_/])viking[-_]?7b|(?:^|[-_/])h2ogpt|(?:^|[-_/])koala|(?:^|[-_/])baize|(?:^|[-_/])moss[-_]|bactrian|jetmoe|(?:^|[-_/])pleias|(?:^|[-_/])lfm"),
]
FAM_COMPILED = [(f, re.compile(p, re.I)) for f, p in FAMILIES]

REPUTABLE_ORGS = {
    "qwen", "meta-llama", "mistralai", "google", "microsoft", "allenai", "stabilityai",
    "tiiuae", "ibm-granite", "openbmb", "huggingfacetb", "01-ai", "internlm",
    "deepseek-ai", "tinyllama", "eleutherai", "openai-community", "bigscience",
    "facebook", "lgai-exaone", "bsc-lt", "utter-project", "nvidia", "huggingfaceh4",
    "lmsys", "apple", "zyphra", "state-spaces", "cerebras", "mosaicml",
    "togethercomputer", "databricks", "salesforce", "bigcode", "cohereforai",
    "coherelabs", "upstage", "nousresearch", "teknium", "princeton-nlp", "h2oai",
    "nexusflow", "deci", "thudm", "baichuan-inc", "xverse", "skywork", "m-a-p",
    "huggingfacem4", "ai21labs", "arcee-ai", "berkeley-nest", "openchat",
    "argilla", "petals-team", "bee-spoke-data", "llm360", "jetmoe",
    "openlm-research", "rwkv", "ai-sweden-models", "ai-forever", "cyberagent",
    "rinna", "elyza", "sarvamai", "core42", "inceptionai", "sapienzanlp",
    "openllm-france", "croissantllm", "occiglot", "speakleash", "turkunlp",
    "lumiopen", "norallm", "trendyol", "naver-hyperclovax", "kakaocorp",
    "amd", "intel", "neuralmagic", "vilm", "scb10x", "aisingapore",
    "meta-math", "wizardlmteam", "ise-uiuc", "codellama", "defog", "replit",
    "abacusai", "gair", "sail", "lightblue", "epfl-llm", "medalpaca", "biomistral",
    "kyutai", "liquid", "liquidai", "answerdotai", "motif-technologies",
    "moonshotai", "zai-org", "inclusionai", "pleias", "hplt", "trl-lib",
    "simplescaling", "open-thoughts", "bespokelabs", "agentica-org",
    "mediatek-research", "taide", "cognitivecomputations", "jondurbin",
    "migtissera", "ehartford", "garage-baind", "openaccess-ai-collective",
    "pankajmathur", "vikhrmodels", "beomi", "yanolja", "42dot", "sbintuitions",
    "tokyotech-llm", "llm-jp", "stanford-crfm", "carperai", "declare-lab",
    "lmms-lab", "opengvlab", "alignment-handbook", "mlabonne", "openbuddy",
    "nvidia-nemo", "powerinfer", "unsloth", "prunaai", "ai-mo", "tencent",
}

INSTRUCT_RE = re.compile(
    r"instruct|(?:^|[-_])chat(?:[-_.0-9]|$)|(?:^|[-_])it(?:[-_.]|$)|(?:^|[-_])sft(?:[-_.]|$)"
    r"|(?:^|[-_])dpo(?:[-_.]|$)|orpo|(?:^|[-_])kto(?:[-_.]|$)|rlhf|hermes|zephyr|tulu|vicuna"
    r"|alpaca|guanaco|wizard|openchat|starling|dolphin|assistant|neural[-_]?chat"
    r"|tuned|aligned|(?:^|[-_])ift(?:[-_.]|$)|(?:^|[-_])rl(?:[-_.]|$)|(?:^|[-_])im(?:[-_.]|$)",
    re.I,
)
BASE_RE = re.compile(r"(?:^|[-_])base(?:[-_.]|$)|(?:^|[-_])pt(?:[-_.]|$)|pretrain", re.I)


def classify(model_id: str) -> dict:
    low = model_id.lower()
    org = low.split("/")[0] if "/" in low else "(no-org)"
    name = low.split("/", 1)[1] if "/" in low else low
    reputable = org in REPUTABLE_ORGS
    out: dict = {"reputable_org": int(reputable)}

    # ---- 3a non-English
    langs, markers = [], []
    for lang, rx in LANG_COMPILED:
        m = rx.search(name) or rx.search(org)
        if m:
            langs.append(lang)
            markers.append(f"{lang}:{m.group(0)}")
    if org in LANG_ORG:
        lg = LANG_ORG[org]
        if lg not in langs:
            langs.append(lg)
            markers.append(f"{lg}:org={org}")
    m = NON_ASCII.search(model_id)
    if m:
        langs.append("non_latin_script")
        markers.append(f"non_latin_script:{m.group(0)}")
    soft_cjk = org in SOFT_CJK_ORGS or (reputable and bool(SOFT_CJK_NAME.search(low)))
    hard_langs = [x for x in langs if not (soft_cjk and x == "chinese")]
    out["f_nonenglish"] = int(bool(hard_langs))
    out["nonenglish_strength"] = (
        "hard" if hard_langs else ("soft_cjk_vendor" if langs else "")
    )
    out["nonenglish_langs"] = ";".join(sorted(set(langs)))
    out["nonenglish_markers"] = ";".join(markers[:4])

    # ---- 3b-3g regex categories
    hits: dict[str, list[str]] = {}
    for cat, lab, rx, hygiene_only in COMPILED:
        if hygiene_only and reputable:
            continue                       # canonical `-v0.2` etc. are legitimate
        m = rx.search(name)
        if m is None and lab == "rp_org":
            m = rx.search(low)
        if m:
            hits.setdefault(cat, []).append(f"{lab}:{m.group(0)}")
    for cat in ("merge", "quant", "rp_nsfw", "code", "math", "domain_bmf", "vision",
                "reasoning", "degenerate"):
        out[f"f_{cat}"] = int(cat in hits)
        out[f"{cat}_markers"] = ";".join(hits.get(cat, [])[:4])

    # ---- family
    fam = None
    for f, rx in FAM_COMPILED:
        if rx.search(low):
            fam = f
            break
    out["family"] = fam or ""
    out["has_family"] = int(fam is not None)

    # ---- quality gate: hyper-parameter / ablation-grid naming
    am = ABLATION_STRONG.search(name)
    if am is None and not reputable:
        am = ABLATION_WEAK.search(name)
    out["f_ablation"] = int(bool(am))
    out["ablation_marker"] = am.group(0) if am else ""
    out["f_mirror"] = int(org in MIRROR_ORGS)

    # ---- base vs instruct
    if INSTRUCT_RE.search(name):
        out["variant"] = "instruct"
    elif BASE_RE.search(name):
        out["variant"] = "base"
    else:
        out["variant"] = "base?"
    return out


cls = pd.DataFrame([classify(m) for m in cand["model"]])
df = pd.concat([cand.reset_index(drop=True), cls], axis=1)

# 3g tiny
df["f_tiny"] = ((df["params_b"].notna()) & (df["params_b"] < 0.15)).astype(int)
df["f_degenerate"] = ((df["f_degenerate"] == 1) | (df["f_tiny"] == 1)).astype(int)
df.loc[df["f_tiny"] == 1, "degenerate_markers"] = (
    df.loc[df["f_tiny"] == 1, "degenerate_markers"].fillna("") + ";params<0.15B"
).str.strip(";")

# org spam: non-reputable org dumping many near-identical checkpoints
win = df["params_b"].notna() & (df["params_b"] > 0) & (df["params_b"] <= 7.0)
orgn = df[win].groupby("org").size()
df["org_window_count"] = df["org"].map(orgn).fillna(0).astype(int)
df["f_orgspam"] = ((df["reputable_org"] == 0) & (df["org_window_count"] >= 10)).astype(int)

FLAGCOLS = ["f_nonenglish", "f_merge", "f_quant", "f_rp_nsfw", "f_code", "f_math",
            "f_domain_bmf", "f_vision", "f_reasoning", "f_degenerate"]
df["n_flags"] = df[FLAGCOLS].sum(axis=1)
df["clean"] = (df["n_flags"] == 0).astype(int)
df["quality_gate_ok"] = (
    (df["f_ablation"] == 0) & (df["f_orgspam"] == 0) & (df["f_mirror"] == 0)
).astype(int)

df.to_csv(os.path.join(OUT, "candidates_flagged.csv"), index=False)
print(f"wrote candidates_flagged.csv  rows={len(df)}")
print(f"params filled from name: {(df.params_source == 'inferred_from_name').sum()}, "
      f"still unknown: {(df.params_source == 'unknown').sum()}")
W = df[win]
print(f"window={len(W)}  clean={int(W.clean.sum())}  "
      f"clean+family={int(((W.clean == 1) & (W.has_family == 1)).sum())}  "
      f"clean+family+qgate={int(((W.clean == 1) & (W.has_family == 1) & (W.quality_gate_ok == 1)).sum())}  "
      f"...+reputable={int(((W.clean == 1) & (W.has_family == 1) & (W.quality_gate_ok == 1) & (W.reputable_org == 1)).sum())}")
