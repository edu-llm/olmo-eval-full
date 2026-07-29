#!/usr/bin/env python3
"""Build a 200-model roster (0.2-7B) with a truncated-normal size mix and
org/architecture diversity. Writes models_200.yaml + stats + a size histogram.

Includes OLMo family (1B + 7B). Halves the 0.2-1B and 6-7B tails vs the prior
roster and drops families that hit vLLM load/hang failures on the P6 node.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(EXP)))
OUT_YAML = os.path.join(REPO, "AdaptiveTesting/Inputs/Models/models_200.yaml")
OUT_PNG = os.path.join(EXP, "figures/models_200_param_dist.png")
OUT_STATS = os.path.join(EXP, "results/models_200_stats.txt")

# (hf_id, params_b, family, architecture)
# Prefer distinct orgs/families/architectures; avoid stacking many near-twins.
# Range: 0.2B .. 7.0B inclusive. OLMo family included; vLLM-broken families filtered later.
RAW: list[tuple[str, float, str, str]] = [
    # ===== ~0.2-1.0B (left tail) =====
    ("facebook/opt-350m", 0.35, "OPT", "transformer"),
    ("EleutherAI/pythia-410m", 0.41, "Pythia", "transformer"),
    ("openai-community/gpt2-medium", 0.36, "GPT-2", "transformer"),
    ("openai-community/gpt2-large", 0.77, "GPT-2", "transformer"),
    ("HuggingFaceTB/SmolLM2-360M", 0.36, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM-360M", 0.36, "SmolLM", "transformer"),
    ("bigscience/bloom-560m", 0.56, "BLOOM", "transformer"),
    ("Qwen/Qwen2.5-0.5B", 0.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2-0.5B", 0.5, "Qwen2", "transformer"),
    ("Qwen/Qwen3-0.6B", 0.6, "Qwen3", "transformer"),
    ("EleutherAI/pythia-1b", 1.0, "Pythia", "transformer"),
    ("google/gemma-3-1b-it", 1.0, "Gemma3", "transformer"),
    ("meta-llama/Llama-3.2-1B", 1.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-1B-Instruct", 1.0, "Llama3.2", "transformer"),
    ("tiiuae/Falcon3-1B-Base", 1.0, "Falcon3", "transformer"),
    ("microsoft/phi-1", 1.3, "Phi", "transformer"),  # will sit in 1-2
    ("facebook/opt-125m", 0.125, "OPT", "transformer"),  # below floor; filtered later if <0.2
    ("state-spaces/mamba-790m-hf", 0.79, "Mamba", "ssm"),
    ("RWKV/rwkv-4-169m-pile", 0.17, "RWKV", "rwkv"),  # below floor
    ("cerebras/Cerebras-GPT-590M", 0.59, "Cerebras-GPT", "transformer"),
    ("EleutherAI/gpt-neo-125m", 0.125, "GPT-Neo", "transformer"),  # below
    ("bigscience/bloom-1b1", 1.1, "BLOOM", "transformer"),
    ("apple/OpenELM-270M", 0.27, "OpenELM", "transformer"),
    ("apple/OpenELM-450M", 0.45, "OpenELM", "transformer"),
    ("HuggingFaceTB/SmolLM2-135M", 0.135, "SmolLM2", "transformer"),  # below
    ("Salesforce/codegen-350M-mono", 0.35, "CodeGen", "transformer"),
    ("Salesforce/codegen-350M-multi", 0.35, "CodeGen", "transformer"),
    ("facebook/xglm-564M", 0.56, "XGLM", "transformer"),
    ("google/flan-t5-base", 0.25, "Flan-T5", "encoder-decoder"),
    ("google/flan-t5-large", 0.78, "Flan-T5", "encoder-decoder"),
    ("google/byt5-base", 0.58, "ByT5", "encoder-decoder"),
    ("microsoft/DialoGPT-medium", 0.35, "DialoGPT", "transformer"),
    ("microsoft/DialoGPT-large", 0.77, "DialoGPT", "transformer"),
    ("distilbert/distilgpt2", 0.088, "DistilGPT2", "transformer"),  # below
    ("sshleifer/tiny-gpt2", 0.05, "GPT-2", "transformer"),  # below
    ("facebook/opt-1.3b", 1.3, "OPT", "transformer"),
    ("Qwen/Qwen1.5-0.5B", 0.5, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-0.5B-Chat", 0.5, "Qwen1.5", "transformer"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1.1, "TinyLlama", "transformer"),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 1.1, "TinyLlama", "transformer"),
    ("PKU-Alignment/alpaca-7b-reproduced", 7.0, "Alpaca", "transformer"),  # 7B
    # ===== 1.0-2.0B =====
    ("microsoft/phi-1_5", 1.3, "Phi", "transformer"),
    ("EleutherAI/gpt-neo-1.3B", 1.3, "GPT-Neo", "transformer"),
    ("EleutherAI/pythia-1.4b", 1.4, "Pythia", "transformer"),
    ("openai-community/gpt2-xl", 1.5, "GPT-2", "transformer"),
    ("Qwen/Qwen2.5-1.5B", 1.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2-1.5B", 1.5, "Qwen2", "transformer"),
    ("Qwen/Qwen3-1.7B", 1.7, "Qwen3", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B", 1.7, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM-1.7B", 1.7, "SmolLM", "transformer"),
    ("bigscience/bloom-1b7", 1.7, "BLOOM", "transformer"),
    ("bigscience/bloomz-1b7", 1.7, "BLOOMz", "transformer"),
    ("stabilityai/stablelm-2-1_6b", 1.6, "StableLM2", "transformer"),
    ("stabilityai/stablelm-2-zephyr-1_6b", 1.6, "StableLM2", "transformer"),
    ("h2oai/h2o-danube2-1.8b-base", 1.8, "Danube", "transformer"),
    ("internlm/internlm2-1_8b", 1.8, "InternLM2", "transformer"),
    ("apple/OpenELM-1_1B", 1.1, "OpenELM", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-base", 1.3, "DeepSeek-Coder", "transformer"),
    ("cerebras/Cerebras-GPT-1.3B", 1.3, "Cerebras-GPT", "transformer"),
    ("Salesforce/codegen-2B-mono", 2.0, "CodeGen", "transformer"),
    ("facebook/xglm-1.7B", 1.7, "XGLM", "transformer"),
    ("google/flan-t5-xl", 3.0, "Flan-T5", "encoder-decoder"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),
    ("google/gemma-2-2b-it", 2.0, "Gemma2", "transformer"),
    ("google/gemma-2b", 2.5, "Gemma1", "transformer"),
    ("google/gemma-2b-it", 2.5, "Gemma1", "transformer"),
    ("state-spaces/mamba-1.4b-hf", 1.4, "Mamba", "ssm"),
    ("RWKV/rwkv-4-1b5-pile", 1.5, "RWKV", "rwkv"),
    ("microsoft/phi-2", 2.7, "Phi", "transformer"),
    ("Qwen/Qwen1.5-1.8B", 1.8, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-1.8B-Chat", 1.8, "Qwen1.5", "transformer"),
    ("openlm-research/open_llama_3b_v2", 3.0, "OpenLLaMA", "transformer"),
    ("togethercomputer/RedPajama-INCITE-Base-3B-v1", 3.0, "RedPajama", "transformer"),
    ("facebook/opt-2.7b", 2.7, "OPT", "transformer"),
    ("EleutherAI/gpt-neo-2.7B", 2.7, "GPT-Neo", "transformer"),
    ("EleutherAI/pythia-2.8b", 2.8, "Pythia", "transformer"),
    ("cerebras/Cerebras-GPT-2.7B", 2.7, "Cerebras-GPT", "transformer"),
    ("bigscience/bloom-3b", 3.0, "BLOOM", "transformer"),
    ("bigscience/bloomz-3b", 3.0, "BLOOMz", "transformer"),
    ("tiiuae/Falcon3-3B-Base", 3.0, "Falcon3", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.0, "Llama3.2", "transformer"),
    ("Qwen/Qwen2.5-3B", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5", "transformer"),
    ("stabilityai/stablelm-zephyr-3b", 3.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-3b-4e1t", 3.0, "StableLM", "transformer"),
    ("apple/OpenELM-3B", 3.0, "OpenELM", "transformer"),
    ("openbmb/MiniCPM-2B-sft-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM3-4B", 4.0, "MiniCPM3", "transformer"),
    ("ibm-granite/granite-3.1-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.1-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.0-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.0-2b-instruct", 2.0, "Granite", "transformer"),
    ("state-spaces/mamba-2.8b-hf", 2.8, "Mamba", "ssm"),
    ("RWKV/rwkv-4-3b-pile", 3.0, "RWKV", "rwkv"),
    ("Salesforce/codegen-2B-multi", 2.0, "CodeGen", "transformer"),
    ("bigcode/starcoder2-3b", 3.0, "StarCoder2", "transformer"),
    ("bigcode/starcoderbase-3b", 3.0, "StarCoder", "transformer"),
    ("codellama/CodeLlama-7b-hf", 7.0, "CodeLlama", "transformer"),
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, "Phi3.5", "transformer"),
    ("microsoft/Phi-4-mini-instruct", 3.8, "Phi4", "transformer"),
    ("Qwen/Qwen1.5-4B", 4.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen3-4B", 4.0, "Qwen3", "transformer"),
    ("google/gemma-3-4b-it", 4.0, "Gemma3", "transformer"),
    ("h2oai/h2o-danube3-4b-base", 4.0, "Danube3", "transformer"),
    ("nvidia/Nemotron-Mini-4B-Instruct", 4.0, "Nemotron", "transformer"),
    ("nvidia/Minitron-4B-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Width-Base", 4.0, "Minitron", "transformer"),
    ("allenai/OLMo-2-1124-7B", 7.0, "OLMo2", "transformer"),  # 7B OK; no 1B
    ("allenai/OLMo-2-1124-7B-Instruct", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-7B-0724-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMoE-1B-7B-0924", 6.9, "OLMoE", "moe"),  # total ~7B, active ~1B
    ("01-ai/Yi-6B", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-6B-Chat", 6.0, "Yi", "transformer"),
    ("mistralai/Mistral-7B-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-v0.3", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.0, "Mistral", "transformer"),
    ("mistralai/Mathstral-7B-v0.1", 7.0, "Mathstral", "transformer"),
    ("Qwen/Qwen2.5-7B", 7.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-7B-Instruct", 7.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-Math-7B", 7.0, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen2.5-Coder-7B", 7.0, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2-7B", 7.0, "Qwen2", "transformer"),
    ("Qwen/Qwen2-7B-Instruct", 7.0, "Qwen2", "transformer"),
    ("tiiuae/falcon-7b", 7.0, "Falcon", "transformer"),
    ("tiiuae/falcon-7b-instruct", 7.0, "Falcon", "transformer"),
    ("tiiuae/Falcon3-7B-Base", 7.0, "Falcon3", "transformer"),
    ("mosaicml/mpt-7b", 7.0, "MPT", "transformer"),
    ("mosaicml/mpt-7b-instruct", 7.0, "MPT", "transformer"),
    ("facebook/opt-6.7b", 6.7, "OPT", "transformer"),
    ("EleutherAI/pythia-6.9b", 6.9, "Pythia", "transformer"),
    ("EleutherAI/gpt-j-6b", 6.0, "GPT-J", "transformer"),
    ("bigscience/bloom-7b1", 7.1, "BLOOM", "transformer"),  # slightly over 7; clip filter
    ("bigscience/bloomz-7b1", 7.1, "BLOOMz", "transformer"),
    ("cerebras/Cerebras-GPT-6.7B", 6.7, "Cerebras-GPT", "transformer"),
    ("deepseek-ai/deepseek-llm-7b-base", 7.0, "DeepSeek", "transformer"),
    ("deepseek-ai/deepseek-math-7b-base", 7.0, "DeepSeek-Math", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-base", 6.7, "DeepSeek-Coder", "transformer"),
    ("HuggingFaceH4/zephyr-7b-beta", 7.0, "Zephyr", "transformer"),
    ("lmsys/vicuna-7b-v1.5", 7.0, "Vicuna", "transformer"),
    ("LLM360/Amber", 7.0, "Amber", "transformer"),
    ("internlm/internlm2-7b", 7.0, "InternLM2", "transformer"),
    ("internlm/internlm2_5-7b", 7.0, "InternLM2.5", "transformer"),
    ("togethercomputer/RedPajama-INCITE-7B-Base", 7.0, "RedPajama", "transformer"),
    ("openlm-research/open_llama_7b_v2", 7.0, "OpenLLaMA", "transformer"),
    ("Deci/DeciLM-7B", 7.0, "DeciLM", "transformer"),
    ("Zyphra/Zamba-7B-v1", 7.0, "Zamba", "ssm-transformer"),
    ("LumiOpen/Viking-7B", 7.0, "Viking", "transformer"),
    ("ibm-granite/granite-3.1-8b-base", 8.0, "Granite", "transformer"),  # over; filter
    ("ibm/granite-7b-base", 7.0, "Granite7", "transformer"),
    ("m-a-p/neo_7b", 7.0, "NEO", "transformer"),
    ("bigcode/starcoder2-7b", 7.0, "StarCoder2", "transformer"),
    ("THUDM/chatglm3-6b", 6.0, "ChatGLM3", "transformer"),
    ("THUDM/glm-4-9b-chat", 9.0, "GLM4", "transformer"),  # over
    ("baichuan-inc/Baichuan2-7B-Base", 7.0, "Baichuan2", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Chat", 7.0, "Baichuan2", "transformer"),
    ("BAAI/Aquila-7B", 7.0, "Aquila", "transformer"),
    ("BAAI/AquilaChat-7B", 7.0, "Aquila", "transformer"),
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("databricks/dolly-v2-7b", 7.0, "Dolly", "transformer"),
    ("Writer/Palmyra-3B", 3.0, "Palmyra", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("CohereForAI/aya-23-8B", 8.0, "Aya", "transformer"),  # over
    ("CohereForAI/c4ai-command-r7b-12-2024", 7.0, "Command-R", "transformer"),
    ("upstage/SOLAR-10.7B-v1.0", 10.7, "SOLAR", "transformer"),  # over
    ("NousResearch/Hermes-2-Pro-Mistral-7B", 7.0, "Hermes", "transformer"),
    ("teknium/OpenHermes-2.5-Mistral-7B", 7.0, "OpenHermes", "transformer"),
    ("Open-Orca/Mistral-7B-OpenOrca", 7.0, "OpenOrca", "transformer"),
    ("berkeley-nest/Starling-LM-7B-alpha", 7.0, "Starling", "transformer"),
    ("Nexusflow/Starling-LM-7B-beta", 7.0, "Starling", "transformer"),
    ("WizardLM/WizardLM-7B-V1.0", 7.0, "WizardLM", "transformer"),
    ("WizardLMTeam/WizardMath-7B-V1.1", 7.0, "WizardMath", "transformer"),
    ("cognitivecomputations/dolphin-2.9.4-llama3.1-8b", 8.0, "Dolphin", "transformer"),  # over
    ("abacusai/Smaug-Llama-3-70B-Instruct", 70.0, "Smaug", "transformer"),  # over
    ("princeton-nlp/Sheared-LLaMA-2.7B", 2.7, "Sheared-LLaMA", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-1.3B", 1.3, "Sheared-LLaMA", "transformer"),
    ("facebook/MobileLLM-1B", 1.0, "MobileLLM", "transformer"),
    ("facebook/MobileLLM-350M", 0.35, "MobileLLM", "transformer"),
    ("facebook/MobileLLM-125M", 0.125, "MobileLLM", "transformer"),  # below
    ("ML-GSAI/LLaDA-8B-Base", 8.0, "LLaDA", "diffusion"),  # over
    ("tencent/Hunyuan-7B-Pretrain", 7.0, "Hunyuan", "transformer"),
    ("tencent/Hunyuan-7B-Instruct", 7.0, "Hunyuan", "transformer"),
    ("meituan-longcat/LongCat-Flash-Chat", 560.0, "LongCat", "transformer"),  # over
    ("AI-MO/NuminaMath-7B-TIR", 7.0, "NuminaMath", "transformer"),
    ("Qwen/Qwen2.5-Math-1.5B", 1.5, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen2.5-Coder-1.5B", 1.5, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Coder-3B", 3.0, "Qwen2.5-Coder", "transformer"),
    ("google/gemma-7b", 7.0, "Gemma1", "transformer"),
    ("google/gemma-7b-it", 7.0, "Gemma1", "transformer"),
    ("google/gemma-2-9b", 9.0, "Gemma2", "transformer"),  # over
    ("meta-llama/Llama-2-7b-hf", 7.0, "Llama2", "transformer"),
    ("meta-llama/Llama-2-7b-chat-hf", 7.0, "Llama2", "transformer"),
    ("meta-llama/Meta-Llama-3-8B", 8.0, "Llama3", "transformer"),  # over
    ("TencentARC/LLaMA-Pro-8B", 8.0, "LLaMA-Pro", "transformer"),  # over
    ("Biomistral/BioMistral-7B", 7.0, "BioMistral", "transformer"),
    ("Meditron/Meditron-7B", 7.0, "Meditron", "transformer"),
    ("epfl-llm/meditron-7b", 7.0, "Meditron", "transformer"),
    ("FreedomIntelligence/Apollo-7B", 7.0, "Apollo", "transformer"),
    ("internlm/internlm2-chat-7b", 7.0, "InternLM2", "transformer"),
    ("openchat/openchat-3.5-0106", 7.0, "OpenChat", "transformer"),
    ("HuggingFaceH4/zephyr-7b-gemma-v0.1", 7.0, "Zephyr-Gemma", "transformer"),
    ("google/recurrentgemma-2b", 2.0, "RecurrentGemma", "rg-lru"),
    ("google/recurrentgemma-2b-it", 2.0, "RecurrentGemma", "rg-lru"),
    ("google/gemma-1.1-2b-it", 2.5, "Gemma1.1", "transformer"),
    ("google/gemma-1.1-7b-it", 7.0, "Gemma1.1", "transformer"),
    ("microsoft/Phi-3-mini-128k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/phi-4", 14.0, "Phi4", "transformer"),  # over
    ("HuggingFaceTB/SmolLM-135M", 0.135, "SmolLM", "transformer"),  # below
    ("EleutherAI/pythia-160m", 0.16, "Pythia", "transformer"),  # below
    ("RWKV/rwkv-4-430m-pile", 0.43, "RWKV", "rwkv"),
    ("state-spaces/mamba-370m-hf", 0.37, "Mamba", "ssm"),
    ("facebook/opt-125m", 0.125, "OPT", "transformer"),  # below duplicate
    ("Salesforce/codegen-6B-mono", 6.0, "CodeGen", "transformer"),
    ("Salesforce/codegen25-7b-mono", 7.0, "CodeGen25", "transformer"),
    ("bigcode/starcoderbase-1b", 1.0, "StarCoder", "transformer"),
    ("bigcode/tiny_starcoder_py", 0.16, "StarCoder", "transformer"),  # below
    ("replit/replit-code-v1-3b", 3.0, "Replit", "transformer"),
    ("StabilityAI/stablecode-3b", 3.0, "StableCode", "transformer"),
    ("stabilityai/stable-code-3b", 3.0, "StableCode", "transformer"),
    ("WizardLM/WizardCoder-Python-7B-V1.0", 7.0, "WizardCoder", "transformer"),
    ("ise-uiuc/Magicoder-S-DS-6.7B", 6.7, "Magicoder", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", 6.7, "DeepSeek-Coder", "transformer"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", 7.0, "Qwen2.5-Coder", "transformer"),
    ("codellama/CodeLlama-7b-Instruct-hf", 7.0, "CodeLlama", "transformer"),
    ("codellama/CodeLlama-7b-Python-hf", 7.0, "CodeLlama", "transformer"),
    ("defog/sqlcoder-7b-2", 7.0, "SQLCoder", "transformer"),
    ("Xwin-LM/Xwin-LM-7B-V0.2", 7.0, "Xwin", "transformer"),
    ("garage-bAInd/Platypus2-7B", 7.0, "Platypus", "transformer"),
    ("OpenAssistant/oasst-sft-4-pythia-12b", 12.0, "OASST", "transformer"),  # over
    ("OpenAssistant/oasst-sft-6-llama-30b", 30.0, "OASST", "transformer"),  # over
    ("LAION-AI/OpenAssistant-SFT-7B", 7.0, "OASST", "transformer"),
    ("tiiuae/falcon-rw-1b", 1.0, "Falcon-RW", "transformer"),
    ("togethercomputer/GPT-JT-6B-v1", 6.0, "GPT-JT", "transformer"),
    ("EleutherAI/gpt-neox-20b", 20.0, "GPT-NeoX", "transformer"),  # over
    ("EleutherAI/llemma_7b", 7.0, "Llemma", "transformer"),
    ("meta-math/MetaMath-7B-V1.0", 7.0, "MetaMath", "transformer"),
    ("WizardLMTeam/WizardMath-7B-V1.0", 7.0, "WizardMath", "transformer"),
    ("Qwen/Qwen2-Math-7B", 7.0, "Qwen2-Math", "transformer"),
    ("deepseek-ai/deepseek-math-7b-instruct", 7.0, "DeepSeek-Math", "transformer"),
    ("microsoft/Orca-2-7b", 7.0, "Orca2", "transformer"),
    ("microsoft/Orca-2-13b", 13.0, "Orca2", "transformer"),  # over
    ("Intel/neural-chat-7b-v3-3", 7.0, "NeuralChat", "transformer"),
    ("Intel/neural-chat-7b-v3-1", 7.0, "NeuralChat", "transformer"),
    ("HuggingFaceH4/starchat2-15b-v0.1", 15.0, "StarChat", "transformer"),  # over
    ("HuggingFaceH4/starchat-beta", 15.0, "StarChat", "transformer"),  # over
    ("TencentARC/LLaMA-Pro-8B-Instruct", 8.0, "LLaMA-Pro", "transformer"),  # over
    ("BAAI/bge-large-en-v1.5", 0.34, "BGE", "embedding"),  # embedding, skip style
    ("sentence-transformers/all-MiniLM-L6-v2", 0.022, "MiniLM", "embedding"),  # below
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1.5, "GTE", "embedding"),
    ("nomic-ai/nomic-embed-text-v1.5", 0.14, "Nomic", "embedding"),  # below
    ("jinaai/jina-embeddings-v2-base-en", 0.14, "Jina", "embedding"),  # below
    ("intfloat/e5-large-v2", 0.34, "E5", "embedding"),
    # More mid-size diverse
    ("microsoft/biogpt", 0.35, "BioGPT", "transformer"),
    ("microsoft/BioGPT-Large", 1.5, "BioGPT", "transformer"),
    ("stanford-crfm/BioMedLM", 2.7, "BioMedLM", "transformer"),
    ("nlpie/tiny-clinicalbert", 0.014, "ClinicalBERT", "encoder"),  # below
    ("emilyalsentzer/Bio_ClinicalBERT", 0.11, "ClinicalBERT", "encoder"),  # below
    ("allenai/scibert_scivocab_uncased", 0.11, "SciBERT", "encoder"),  # below
    ("facebook/bart-large", 0.41, "BART", "encoder-decoder"),
    ("facebook/bart-base", 0.14, "BART", "encoder-decoder"),  # below
    ("google/pegasus-xsum", 0.57, "Pegasus", "encoder-decoder"),
    ("google/t5-v1_1-base", 0.25, "T5", "encoder-decoder"),
    ("google/t5-v1_1-large", 0.77, "T5", "encoder-decoder"),
    ("google/t5-v1_1-xl", 3.0, "T5", "encoder-decoder"),
    ("google/flan-t5-small", 0.08, "Flan-T5", "encoder-decoder"),  # below
    ("MBZUAI/LaMini-GPT-774M", 0.77, "LaMini", "transformer"),
    ("MBZUAI/LaMini-GPT-1.5B", 1.5, "LaMini", "transformer"),
    ("MBZUAI/LaMini-Flan-T5-783M", 0.78, "LaMini", "encoder-decoder"),
    ("declare-lab/flan-alpaca-base", 0.25, "Flan-Alpaca", "encoder-decoder"),
    ("declare-lab/flan-alpaca-large", 0.78, "Flan-Alpaca", "encoder-decoder"),
    ("declare-lab/flan-alpaca-xl", 3.0, "Flan-Alpaca", "encoder-decoder"),
    ("lmsys/fastchat-t5-3b-v1.0", 3.0, "FastChat-T5", "encoder-decoder"),
    ("togethercomputer/RedPajama-INCITE-Chat-3B-v1", 3.0, "RedPajama", "transformer"),
    ("OpenAssistant/oasst-sft-1-pythia-12b", 12.0, "OASST", "transformer"),  # over
    ("CarperAI/stable-vicuna-13b", 13.0, "StableVicuna", "transformer"),  # over
    ("TheBloke/guanaco-7B-HF", 7.0, "Guanaco", "transformer"),
    ("timdettmers/guanaco-7b", 7.0, "Guanaco", "transformer"),
    ("project-baize/baize-v2-7b", 7.0, "Baize", "transformer"),
    ("uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b", 13.0, "Speechless", "transformer"),  # over
    ("cognitivecomputations/dolphin-2.6-mistral-7b", 7.0, "Dolphin", "transformer"),
    ("jondurbin/airoboros-7b", 7.0, "Airoboros", "transformer"),
    ("NousResearch/Nous-Hermes-llama-2-7b", 7.0, "Nous-Hermes", "transformer"),
    ("teknium/OpenHermes-2-Mistral-7B", 7.0, "OpenHermes", "transformer"),
    ("argilla/notus-7b-v1", 7.0, "Notus", "transformer"),
    ("HuggingFaceH4/zephyr-7b-alpha", 7.0, "Zephyr", "transformer"),
    ("alignment-handbook/zephyr-7b-sft-full", 7.0, "Zephyr", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.2", 7.0, "Mistral", "transformer"),
    ("OpenPipe/mistral-ft-optimized-1218", 7.0, "Mistral-FT", "transformer"),
    ("amazon/MistralLite", 7.0, "MistralLite", "transformer"),
    ("upstage/SOLAR-10.7B-Instruct-v1.0", 10.7, "SOLAR", "transformer"),  # over
    ("01-ai/Yi-1.5-6B", 6.0, "Yi1.5", "transformer"),
    ("01-ai/Yi-1.5-6B-Chat", 6.0, "Yi1.5", "transformer"),
    ("01-ai/Yi-1.5-9B", 9.0, "Yi1.5", "transformer"),  # over
    ("Qwen/Qwen1.5-7B", 7.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-7B-Chat", 7.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-4B-Chat", 4.0, "Qwen1.5", "transformer"),
    ("internlm/internlm-7b", 7.0, "InternLM", "transformer"),
    ("internlm/internlm-chat-7b", 7.0, "InternLM", "transformer"),
    ("openbmb/MiniCPM-2B-dpo-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/cpm-bee-1b", 1.0, "CPM-Bee", "transformer"),
    ("openbmb/cpm-bee-5b", 5.0, "CPM-Bee", "transformer"),
    ("THUDM/chatglm2-6b", 6.0, "ChatGLM2", "transformer"),
    ("THUDM/chatglm-6b", 6.0, "ChatGLM", "transformer"),
    ("THUDM/codegeex2-6b", 6.0, "CodeGeeX2", "transformer"),
    ("bigscience/T0_3B", 3.0, "T0", "encoder-decoder"),
    ("bigscience/T0pp", 11.0, "T0", "encoder-decoder"),  # over
    ("google/flan-ul2", 20.0, "Flan-UL2", "encoder-decoder"),  # over
    ("allenai/tk-instruct-3b-def", 3.0, "Tk-Instruct", "encoder-decoder"),
    ("allenai/tk-instruct-base-def-pos", 0.25, "Tk-Instruct", "encoder-decoder"),
    ("allenai/OLMo-1B-hf", 1.0, "OLMo", "transformer"),
    ("allenai/OLMo-1B-0724-hf", 1.0, "OLMo", "transformer"),
    ("allenai/OLMo-2-0425-1B", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-0425-1B-Instruct", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-7B-Instruct-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-7B-Twin-2T-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-1.7-7B-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-2-1124-13B", 13.0, "OLMo2", "transformer"),  # over
    ("HuggingFaceH4/mistral-7b-sft-beta", 7.0, "Mistral-SFT", "transformer"),
    ("mlabonne/NeuralBeagle14-7B", 7.0, "NeuralBeagle", "transformer"),
    ("mlabonne/NeuralHermes-2.5-Mistral-7B", 7.0, "NeuralHermes", "transformer"),
    ("CultriX/NeuralTrix-7B-dpo", 7.0, "NeuralTrix", "transformer"),
    ("SenseTime/SenseChat-7B", 7.0, "SenseChat", "transformer"),
    ("Xwin-LM/Xwin-Math-7B-V1.0", 7.0, "Xwin-Math", "transformer"),
    ("deepseek-ai/deepseek-moe-16b-base", 16.0, "DeepSeek-MoE", "moe"),  # over
    ("Qwen/Qwen1.5-MoE-A2.7B", 14.0, "Qwen1.5-MoE", "moe"),  # over total
    ("google/gemma-2-2b-jpn-it", 2.0, "Gemma2", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("rinna/japanese-gpt-1b", 1.0, "rinna-GPT", "transformer"),
    ("cyberagent/open-calm-7b", 7.0, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-3b", 3.0, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-1b", 1.0, "OpenCALM", "transformer"),
    ("elyza/ELYZA-japanese-Llama-2-7b", 7.0, "ELYZA", "transformer"),
    ("tokyotech-llm/Swallow-7b-hf", 7.0, "Swallow", "transformer"),
    ("tokyotech-llm/Swallow-7b-instruct-hf", 7.0, "Swallow", "transformer"),
    ("lmsys/vicuna-7b-v1.3", 7.0, "Vicuna", "transformer"),
    ("eachadea/vicuna-7b-1.1", 7.0, "Vicuna", "transformer"),
    ("AlekseyKorshuk/vicuna-7b", 7.0, "Vicuna", "transformer"),
    ("chavinlo/alpaca-native", 7.0, "Alpaca", "transformer"),
    ("tatsu-lab/alpaca-7b-wdiff", 7.0, "Alpaca", "transformer"),
    ("circlefork/starchat-alpha", 15.0, "StarChat", "transformer"),  # over
    ("HuggingFaceH4/starchat-alpha", 15.0, "StarChat", "transformer"),  # over
    ("bigcode/santacoder", 1.1, "SantaCoder", "transformer"),
    ("bigcode/starcoderbase-7b", 7.0, "StarCoder", "transformer"),
    ("WizardLM/WizardCoder-15B-V1.0", 15.0, "WizardCoder", "transformer"),  # over
    ("Phind/Phind-CodeLlama-34B-v2", 34.0, "Phind", "transformer"),  # over
    ("WizardLM/WizardLM-13B-V1.2", 13.0, "WizardLM", "transformer"),  # over
    ("garage-bAInd/Platypus2-13B", 13.0, "Platypus", "transformer"),  # over
    ("NousResearch/Nous-Hermes-13b", 13.0, "Nous-Hermes", "transformer"),  # over
    ("CalderaAI/13B-Ouroboros", 13.0, "Ouroboros", "transformer"),  # over
    ("PygmalionAI/pygmalion-7b", 7.0, "Pygmalion", "transformer"),
    ("PygmalionAI/pygmalion-2-7b", 7.0, "Pygmalion", "transformer"),
    ("KoboldAI/GPT-J-6B-Janeway", 6.0, "Janeway", "transformer"),
    ("KoboldAI/GPT-NeoX-20B-Erebus", 20.0, "Erebus", "transformer"),  # over
    ("EleutherAI/pythia-410m-deduped", 0.41, "Pythia", "transformer"),
    ("EleutherAI/pythia-1b-deduped", 1.0, "Pythia", "transformer"),
    ("EleutherAI/pythia-1.4b-deduped", 1.4, "Pythia", "transformer"),
    ("EleutherAI/pythia-2.8b-deduped", 2.8, "Pythia", "transformer"),
    ("EleutherAI/pythia-6.9b-deduped", 6.9, "Pythia", "transformer"),
    ("facebook/opt-350m", 0.35, "OPT", "transformer"),  # duplicate id check
    ("facebook/xglm-2.9B", 2.9, "XGLM", "transformer"),
    ("facebook/xglm-4.5B", 4.5, "XGLM", "transformer"),
    ("facebook/xglm-7.5B", 7.5, "XGLM", "transformer"),  # over
    ("ai21labs/Jamba-v0.1", 52.0, "Jamba", "ssm-transformer"),  # over
    ("ai21labs/AI21-Jamba-1.5-Mini", 12.0, "Jamba", "ssm-transformer"),  # over
    ("state-spaces/mamba-2.8b", 2.8, "Mamba", "ssm"),
    ("tri-ml/mamba-7b-rw", 7.0, "Mamba", "ssm"),
    ("hazyresearch/based-1b", 1.0, "Based", "hybrid"),
    ("hazyresearch/based-360m", 0.36, "Based", "hybrid"),
    ("togethercomputer/mamba-2.8b-slimpj", 2.8, "Mamba", "ssm"),
    ("RWKV/rwkv-5-world-3b", 3.0, "RWKV5", "rwkv"),
    ("RWKV/rwkv-5-world-1b5", 1.5, "RWKV5", "rwkv"),
    ("RWKV/rwkv-5-world-7b", 7.0, "RWKV5", "rwkv"),
    ("BlinkDL/rwkv-4-pile-3b", 3.0, "RWKV", "rwkv"),
    ("BlinkDL/rwkv-4-pile-1b5", 1.5, "RWKV", "rwkv"),
    ("BlinkDL/rwkv-4-pile-7b", 7.0, "RWKV", "rwkv"),
    ("microsoft/phi-1", 1.3, "Phi", "transformer"),  # duplicate
    ("microsoft/Phi-3-small-8k-instruct", 7.0, "Phi3-small", "transformer"),
    ("Qwen/Qwen2.5-0.5B", 0.5, "Qwen2.5", "transformer"),  # duplicate
    ("Qwen/Qwen1.5-14B", 14.0, "Qwen1.5", "transformer"),  # over
    ("h2oai/h2o-danube3-4b-chat", 4.0, "Danube3", "transformer"),
    ("h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b", 7.0, "h2oGPT", "transformer"),
    ("h2oai/h2ogpt-oasst1-512-12b", 12.0, "h2oGPT", "transformer"),  # over
    ("mosaicml/mpt-7b-chat", 7.0, "MPT", "transformer"),
    ("mosaicml/mpt-7b-storywriter", 7.0, "MPT", "transformer"),
    ("mosaicml/mpt-30b", 30.0, "MPT", "transformer"),  # over
    ("tiiuae/falcon-40b", 40.0, "Falcon", "transformer"),  # over
    ("tiiuae/Falcon3-10B-Base", 10.0, "Falcon3", "transformer"),  # over
    ("stabilityai/stablelm-base-alpha-7b", 7.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-tuned-alpha-7b", 7.0, "StableLM", "transformer"),
    ("stabilityai/japanese-stablelm-base-alpha-7b", 7.0, "Japanese-StableLM", "transformer"),
    ("stabilityai/japanese-stablelm-instruct-gamma-7b", 7.0, "Japanese-StableLM", "transformer"),
    ("rinna/nekomata-7b", 7.0, "Nekomata", "transformer"),
    ("rinna/youri-7b", 7.0, "Youri", "transformer"),
    ("pfnet/plamo-13b", 13.0, "PLaMo", "transformer"),  # over
    ("line-corporation/japanese-large-lm-3.6b", 3.6, "LINE-LM", "transformer"),
    ("line-corporation/japanese-large-lm-1.7b", 1.7, "LINE-LM", "transformer"),
    ("sbintuitions/sarashina2-7b", 7.0, "Sarashina", "transformer"),
    ("sbintuitions/sarashina1-7b", 7.0, "Sarashina", "transformer"),
    ("llm-jp/llm-jp-3-7.2b", 7.2, "LLM-JP", "transformer"),  # slightly over 7
    ("llm-jp/llm-jp-3-3.7b", 3.7, "LLM-JP", "transformer"),
    ("llm-jp/llm-jp-3-1.8b", 1.8, "LLM-JP", "transformer"),
    ("tokyotech-llm/Llama-3-Swallow-8B-v0.1", 8.0, "Swallow", "transformer"),  # over
    ("microsoft/Phi-3-medium-4k-instruct", 14.0, "Phi3", "transformer"),  # over
    ("Qwen/Qwen2.5-14B", 14.0, "Qwen2.5", "transformer"),  # over
    ("google/gemma-3-12b-it", 12.0, "Gemma3", "transformer"),  # over
    ("meta-llama/Llama-3.1-8B", 8.0, "Llama3.1", "transformer"),  # over
    ("meta-llama/Llama-3.2-11B-Vision", 11.0, "Llama3.2", "transformer"),  # over
    # Fill mid-range with more diverse orgs
    ("BAAI/Aquila2-7B", 7.0, "Aquila2", "transformer"),
    ("FlagAlpha/Llama2-Chinese-7b-Chat", 7.0, "Llama2-Chinese", "transformer"),
    ("LinkSoul/Chinese-Llama-2-7b", 7.0, "Chinese-Llama2", "transformer"),
    ("IDEA-CCNL/Ziya-LLaMA-13B-v1.1", 13.0, "Ziya", "transformer"),  # over
    ("IDEA-CCNL/Ziya-LLaMA-7B-Reward", 7.0, "Ziya", "transformer"),
    ("fnlp/moss-moon-003-base", 16.0, "MOSS", "transformer"),  # over
    ("fnlp/moss-moon-003-sft", 16.0, "MOSS", "transformer"),  # over
    ("OpenBMB/MiniCPM4-0.5B", 0.5, "MiniCPM4", "transformer"),
    ("OpenBMB/MiniCPM4-8B", 8.0, "MiniCPM4", "transformer"),  # over
    ("openbmb/MiniCPM-S-1B-sft", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-1B-sft-bf16", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-Llama3-V-2_5", 8.0, "MiniCPM-V", "transformer"),  # over
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1.5, "DeepSeek-R1-Distill", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, "DeepSeek-R1-Distill", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 8.0, "DeepSeek-R1-Distill", "transformer"),  # over
    ("microsoft/Phi-4-mini-flash-reasoning", 3.8, "Phi4", "transformer"),
    ("microsoft/Phi-4-reasoning", 14.0, "Phi4", "transformer"),  # over
    ("HuggingFaceTB/SmolLM3-3B", 3.0, "SmolLM3", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B-Base", 3.0, "SmolLM3", "transformer"),
    ("ibm-granite/granite-3.2-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-8b-instruct", 8.0, "Granite", "transformer"),  # over
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", 8.0, "Nemotron", "transformer"),  # over
    ("nvidia/Nemotron-H-8B-Base-8K", 8.0, "Nemotron-H", "hybrid"),  # over
    ("nvidia/Mistral-NeMo-Minitron-8B-Base", 8.0, "Minitron", "transformer"),  # over
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", 2.4, "EXAONE", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", 7.8, "EXAONE", "transformer"),  # over
    ("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", 7.8, "EXAONE", "transformer"),  # over
    ("kakaocorp/kanana-nano-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("kakaocorp/kanana-1.5-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", 1.5, "HyperCLOVAX", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", 0.5, "HyperCLOVAX", "transformer"),
    ("skt/A.X-3.1", 7.0, "A.X", "transformer"),
    ("skt/kogpt2-base-v2", 0.13, "KoGPT2", "transformer"),  # below
    ("beomi/KoAlpaca-Polyglot-5.8B", 5.8, "KoAlpaca", "transformer"),
    ("EleutherAI/polyglot-ko-1.3b", 1.3, "Polyglot-KO", "transformer"),
    ("EleutherAI/polyglot-ko-3.8b", 3.8, "Polyglot-KO", "transformer"),
    ("EleutherAI/polyglot-ko-5.8b", 5.8, "Polyglot-KO", "transformer"),
    ("kakaobrain/kogpt", 6.0, "KoGPT", "transformer"),
    ("nlpai-lab/KULLM-Polyglot-5.8B-v2", 5.8, "KULLM", "transformer"),
    ("nlpai-lab/kullm-polyglot-12.8b-v2", 12.8, "KULLM", "transformer"),  # over
    ("upstage/SOLAR-10.7B-v1.0", 10.7, "SOLAR", "transformer"),  # over dup
    ("upstage/solar-1-mini-chat", 10.7, "SOLAR", "transformer"),  # over
    ("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", 10.8, "EEVE", "transformer"),  # over
    ("yanolja/EEVE-Korean-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("NCSOFT/Llama-VARCO-8B-Instruct", 8.0, "VARCO", "transformer"),  # over
    ("KF-Complex/Korean-Qwen2.5-7B-Instruct", 7.0, "Korean-Qwen", "transformer"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5", "transformer"),  # dup possible
    ("Qwen/QwQ-32B", 32.0, "QwQ", "transformer"),  # over
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "Qwen2.5", "transformer"),
    ("moonshotai/Kimi-VL-A3B-Instruct", 16.0, "Kimi-VL", "moe"),  # over
    ("zai-org/GLM-4-9B-0414", 9.0, "GLM4", "transformer"),  # over
    ("THUDM/glm-4-9b", 9.0, "GLM4", "transformer"),  # over
    ("internlm/internlm2_5-1_8b", 1.8, "InternLM2.5", "transformer"),
    ("internlm/internlm2_5-1_8b-chat", 1.8, "InternLM2.5", "transformer"),
    ("OpenGVLab/InternVL2-2B", 2.0, "InternVL2", "vlm"),
    ("OpenGVLab/InternVL2-4B", 4.0, "InternVL2", "vlm"),
    ("OpenGVLab/InternVL2-8B", 8.0, "InternVL2", "vlm"),  # over
    ("Qwen/Qwen2-VL-2B-Instruct", 2.0, "Qwen2-VL", "vlm"),
    ("Qwen/Qwen2-VL-7B-Instruct", 7.0, "Qwen2-VL", "vlm"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", 3.0, "Qwen2.5-VL", "vlm"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", 7.0, "Qwen2.5-VL", "vlm"),
    ("microsoft/Phi-3.5-vision-instruct", 4.2, "Phi3.5-V", "vlm"),
    ("microsoft/Phi-3-vision-128k-instruct", 4.2, "Phi3-V", "vlm"),
    ("llava-hf/llava-1.5-7b-hf", 7.0, "LLaVA", "vlm"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", 7.0, "LLaVA", "vlm"),
    ("Salesforce/blip2-opt-2.7b", 3.8, "BLIP2", "vlm"),
    ("Salesforce/blip2-flan-t5-xl", 4.0, "BLIP2", "vlm"),
    ("liuhaotian/llava-v1.5-7b", 7.0, "LLaVA", "vlm"),
    ("vikhyatk/moondream2", 1.9, "Moondream", "vlm"),
    ("vikhyatk/moondream1", 1.9, "Moondream", "vlm"),
    ("HuggingFaceM4/idefics2-8b", 8.0, "IDEFICS2", "vlm"),  # over
    ("HuggingFaceM4/idefics-9b", 9.0, "IDEFICS", "vlm"),  # over
    ("openbmb/MiniCPM-V-2", 2.8, "MiniCPM-V", "vlm"),
    ("openbmb/MiniCPM-V-2_6", 8.0, "MiniCPM-V", "vlm"),  # over
    ("rhymes-ai/Aria", 25.0, "Aria", "moe"),  # over
    ("Efficient-Large-Model/VILA1.5-3b", 3.0, "VILA", "vlm"),
    ("Efficient-Large-Model/VILA1.5-3b-s2", 3.0, "VILA", "vlm"),
    ("microsoft/Florence-2-base", 0.23, "Florence2", "vlm"),
    ("microsoft/Florence-2-large", 0.77, "Florence2", "vlm"),
    ("google/paligemma-3b-mix-224", 3.0, "PaliGemma", "vlm"),
    ("google/paligemma-3b-pt-224", 3.0, "PaliGemma", "vlm"),
    ("google/paligemma2-3b-pt-224", 3.0, "PaliGemma2", "vlm"),
    # ---- extra mid-size uniques to thicken the normal peak (2-6B) ----
    ("microsoft/phi-2", 2.7, "Phi", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-2.7B", 2.7, "Sheared-LLaMA", "transformer"),
    ("openlm-research/open_llama_3b", 3.0, "OpenLLaMA", "transformer"),
    ("openlm-research/open_llama_3b_v2", 3.0, "OpenLLaMA", "transformer"),
    ("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", 3.0, "RedPajama", "transformer"),
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("Writer/Palmyra-3B", 3.0, "Palmyra", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("openbmb/cpm-bee-5b", 5.0, "CPM-Bee", "transformer"),
    ("facebook/xglm-4.5B", 4.5, "XGLM", "transformer"),
    ("beomi/KoAlpaca-Polyglot-5.8B", 5.8, "KoAlpaca", "transformer"),
    ("EleutherAI/polyglot-ko-5.8b", 5.8, "Polyglot-KO", "transformer"),
    ("nlpai-lab/KULLM-Polyglot-5.8B-v2", 5.8, "KULLM", "transformer"),
    ("kakaobrain/kogpt", 6.0, "KoGPT", "transformer"),
    ("THUDM/chatglm3-6b", 6.0, "ChatGLM3", "transformer"),
    ("THUDM/chatglm2-6b", 6.0, "ChatGLM2", "transformer"),
    ("THUDM/chatglm-6b", 6.0, "ChatGLM", "transformer"),
    ("THUDM/codegeex2-6b", 6.0, "CodeGeeX2", "transformer"),
    ("01-ai/Yi-6B", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-6B-Chat", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-1.5-6B", 6.0, "Yi1.5", "transformer"),
    ("01-ai/Yi-1.5-6B-Chat", 6.0, "Yi1.5", "transformer"),
    ("Salesforce/codegen-6B-mono", 6.0, "CodeGen", "transformer"),
    ("Salesforce/codegen-6B-multi", 6.0, "CodeGen", "transformer"),
    ("EleutherAI/gpt-j-6b", 6.0, "GPT-J", "transformer"),
    ("togethercomputer/GPT-JT-6B-v1", 6.0, "GPT-JT", "transformer"),
    ("KoboldAI/GPT-J-6B-Janeway", 6.0, "Janeway", "transformer"),
    ("facebook/opt-6.7b", 6.7, "OPT", "transformer"),
    ("cerebras/Cerebras-GPT-6.7B", 6.7, "Cerebras-GPT", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-base", 6.7, "DeepSeek-Coder", "transformer"),
    ("ise-uiuc/Magicoder-S-DS-6.7B", 6.7, "Magicoder", "transformer"),
    ("Qwen/Qwen1.5-4B", 4.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-4B-Chat", 4.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen3-4B", 4.0, "Qwen3", "transformer"),
    ("google/gemma-3-4b-it", 4.0, "Gemma3", "transformer"),
    ("h2oai/h2o-danube3-4b-base", 4.0, "Danube3", "transformer"),
    ("h2oai/h2o-danube3-4b-chat", 4.0, "Danube3", "transformer"),
    ("nvidia/Nemotron-Mini-4B-Instruct", 4.0, "Nemotron", "transformer"),
    ("nvidia/Minitron-4B-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Width-Base", 4.0, "Minitron", "transformer"),
    ("openbmb/MiniCPM3-4B", 4.0, "MiniCPM3", "transformer"),
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, "Phi3.5", "transformer"),
    ("microsoft/Phi-3-mini-128k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-4-mini-instruct", 3.8, "Phi4", "transformer"),
    ("EleutherAI/polyglot-ko-3.8b", 3.8, "Polyglot-KO", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", 2.4, "EXAONE", "transformer"),
    ("kakaocorp/kanana-nano-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("kakaocorp/kanana-1.5-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("yanolja/EEVE-Korean-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B", 3.0, "SmolLM3", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B-Base", 3.0, "SmolLM3", "transformer"),
    ("llm-jp/llm-jp-3-3.7b", 3.7, "LLM-JP", "transformer"),
    ("line-corporation/japanese-large-lm-3.6b", 3.6, "LINE-LM", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("cyberagent/open-calm-3b", 3.0, "OpenCALM", "transformer"),
    ("replit/replit-code-v1-3b", 3.0, "Replit", "transformer"),
    ("stabilityai/stable-code-3b", 3.0, "StableCode", "transformer"),
    ("bigcode/starcoder2-3b", 3.0, "StarCoder2", "transformer"),
    ("bigcode/starcoderbase-3b", 3.0, "StarCoder", "transformer"),
    ("facebook/xglm-2.9B", 2.9, "XGLM", "transformer"),
    ("RWKV/rwkv-5-world-3b", 3.0, "RWKV5", "rwkv"),
    ("BlinkDL/rwkv-4-pile-3b", 3.0, "RWKV", "rwkv"),
    ("state-spaces/mamba-2.8b-hf", 2.8, "Mamba", "ssm"),
    ("togethercomputer/mamba-2.8b-slimpj", 2.8, "Mamba", "ssm"),
    ("google/recurrentgemma-2b", 2.0, "RecurrentGemma", "rg-lru"),
    ("google/recurrentgemma-2b-it", 2.0, "RecurrentGemma", "rg-lru"),
    ("ibm-granite/granite-3.1-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.2-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-2b-instruct", 2.0, "Granite", "transformer"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),
    ("google/gemma-2-2b-it", 2.0, "Gemma2", "transformer"),
    ("google/gemma-2b", 2.5, "Gemma1", "transformer"),
    ("google/gemma-2b-it", 2.5, "Gemma1", "transformer"),
    ("google/gemma-1.1-2b-it", 2.5, "Gemma1.1", "transformer"),
    ("Qwen/Qwen2.5-3B", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-Coder-3B", 3.0, "Qwen2.5-Coder", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.0, "Llama3.2", "transformer"),
    ("tiiuae/Falcon3-3B-Base", 3.0, "Falcon3", "transformer"),
    ("apple/OpenELM-3B", 3.0, "OpenELM", "transformer"),
    ("stabilityai/stablelm-zephyr-3b", 3.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-3b-4e1t", 3.0, "StableLM", "transformer"),
    ("bigscience/bloom-3b", 3.0, "BLOOM", "transformer"),
    ("bigscience/bloomz-3b", 3.0, "BLOOMz", "transformer"),
    ("bigscience/T0_3B", 3.0, "T0", "encoder-decoder"),
    ("allenai/tk-instruct-3b-def", 3.0, "Tk-Instruct", "encoder-decoder"),
    ("declare-lab/flan-alpaca-xl", 3.0, "Flan-Alpaca", "encoder-decoder"),
    ("lmsys/fastchat-t5-3b-v1.0", 3.0, "FastChat-T5", "encoder-decoder"),
    ("google/t5-v1_1-xl", 3.0, "T5", "encoder-decoder"),
    ("google/flan-t5-xl", 3.0, "Flan-T5", "encoder-decoder"),
    ("google/paligemma-3b-mix-224", 3.0, "PaliGemma", "vlm"),
    ("Efficient-Large-Model/VILA1.5-3b", 3.0, "VILA", "vlm"),
    ("facebook/musicgen-large", 3.3, "MusicGen", "audio"),
    ("microsoft/Phi-3.5-vision-instruct", 4.2, "Phi3.5-V", "vlm"),
    ("microsoft/Phi-3-vision-128k-instruct", 4.2, "Phi3-V", "vlm"),
    ("Salesforce/blip2-opt-2.7b", 3.8, "BLIP2", "vlm"),
    ("Salesforce/blip2-flan-t5-xl", 4.0, "BLIP2", "vlm"),
    ("OpenGVLab/InternVL2-4B", 4.0, "InternVL2", "vlm"),
    ("OpenGVLab/InternVL2-2B", 2.0, "InternVL2", "vlm"),
    ("Qwen/Qwen2-VL-2B-Instruct", 2.0, "Qwen2-VL", "vlm"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", 3.0, "Qwen2.5-VL", "vlm"),
    ("vikhyatk/moondream2", 1.9, "Moondream", "vlm"),
    ("openbmb/MiniCPM-V-2", 2.8, "MiniCPM-V", "vlm"),
    ("openbmb/MiniCPM-2B-sft-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-2B-dpo-bf16", 2.7, "MiniCPM", "transformer"),
    ("EleutherAI/gpt-neo-2.7B", 2.7, "GPT-Neo", "transformer"),
    ("EleutherAI/pythia-2.8b", 2.8, "Pythia", "transformer"),
    ("EleutherAI/pythia-2.8b-deduped", 2.8, "Pythia", "transformer"),
    ("facebook/opt-2.7b", 2.7, "OPT", "transformer"),
    ("cerebras/Cerebras-GPT-2.7B", 2.7, "Cerebras-GPT", "transformer"),
    ("stanford-crfm/BioMedLM", 2.7, "BioMedLM", "transformer"),
    ("Salesforce/codegen-2B-mono", 2.0, "CodeGen", "transformer"),
    ("Salesforce/codegen-2B-multi", 2.0, "CodeGen", "transformer"),
    ("hazyresearch/based-1b", 1.0, "Based", "hybrid"),
    ("hazyresearch/based-360m", 0.36, "Based", "hybrid"),
    # more ~5B uniques if they exist as distinct checkpoints
    ("EleutherAI/polyglot-ko-1.3b", 1.3, "Polyglot-KO", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", 1.5, "HyperCLOVAX", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", 0.5, "HyperCLOVAX", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1.5, "DeepSeek-R1-Distill", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, "DeepSeek-R1-Distill", "transformer"),
    ("allenai/OLMoE-1B-7B-0924", 6.9, "OLMoE", "moe"),
    ("Zyphra/Zamba-7B-v1", 7.0, "Zamba", "ssm-transformer"),
    ("tri-ml/mamba-7b-rw", 7.0, "Mamba", "ssm"),
    ("RWKV/rwkv-5-world-7b", 7.0, "RWKV5", "rwkv"),
    ("BlinkDL/rwkv-4-pile-7b", 7.0, "RWKV", "rwkv"),
    ("microsoft/Phi-3-small-8k-instruct", 7.0, "Phi3-small", "transformer"),
    ("skt/A.X-3.1", 7.0, "A.X", "transformer"),
    ("LumiOpen/Viking-7B", 7.0, "Viking", "transformer"),
    ("Deci/DeciLM-7B", 7.0, "DeciLM", "transformer"),
    ("m-a-p/neo_7b", 7.0, "NEO", "transformer"),
    ("ibm/granite-7b-base", 7.0, "Granite7", "transformer"),
    ("Biomistral/BioMistral-7B", 7.0, "BioMistral", "transformer"),
    ("epfl-llm/meditron-7b", 7.0, "Meditron", "transformer"),
    ("EleutherAI/llemma_7b", 7.0, "Llemma", "transformer"),
    ("AI-MO/NuminaMath-7B-TIR", 7.0, "NuminaMath", "transformer"),
    ("mistralai/Mathstral-7B-v0.1", 7.0, "Mathstral", "transformer"),
    ("CohereForAI/c4ai-command-r7b-12-2024", 7.0, "Command-R", "transformer"),
    ("tencent/Hunyuan-7B-Pretrain", 7.0, "Hunyuan", "transformer"),
    ("tencent/Hunyuan-7B-Instruct", 7.0, "Hunyuan", "transformer"),
    ("BAAI/Aquila2-7B", 7.0, "Aquila2", "transformer"),
    ("BAAI/Aquila-7B", 7.0, "Aquila", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Base", 7.0, "Baichuan2", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Chat", 7.0, "Baichuan2", "transformer"),
    ("allenai/OLMo-2-1124-7B", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-1124-7B-Instruct", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-7B-0724-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-7B-Instruct-hf", 7.0, "OLMo", "transformer"),

    ("Qwen/Qwen2-Audio-7B", 7.0, "Qwen2-Audio", "audio"),
    ("Qwen/Qwen2-Audio-7B-Instruct", 7.0, "Qwen2-Audio", "audio"),
    ("openai/whisper-large-v3", 1.5, "Whisper", "audio"),
    ("openai/whisper-medium", 0.77, "Whisper", "audio"),
    ("openai/whisper-small", 0.24, "Whisper", "audio"),
    ("facebook/seamless-m4t-v2-large", 2.3, "SeamlessM4T", "audio"),
    ("facebook/mms-1b-all", 1.0, "MMS", "audio"),
    ("microsoft/speecht5_tts", 0.14, "SpeechT5", "audio"),  # below
    ("suno/bark", 1.0, "Bark", "audio"),
    ("facebook/musicgen-small", 0.3, "MusicGen", "audio"),
    ("facebook/musicgen-medium", 1.5, "MusicGen", "audio"),
    ("facebook/musicgen-large", 3.3, "MusicGen", "audio"),
    ("stabilityai/stable-audio-open-1.0", 1.0, "StableAudio", "audio"),
    # Keep generative LLMs primary; filter embeddings/audio/vlm later for "LLM eval" list
    # ---- vLLM-safe fill (decoder-only; used to refill mid bins after blocklist) ----
    ("Qwen/Qwen2-1.5B-Instruct", 1.5, "Qwen2", "transformer"),
    ("Qwen/Qwen2.5-Math-1.5B-Instruct", 1.5, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", 3.0, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen3-0.6B-Instruct", 0.6, "Qwen3", "transformer"),
    ("Qwen/Qwen3-1.7B-Instruct", 1.7, "Qwen3", "transformer"),
    ("Qwen/Qwen3-4B-Instruct", 4.0, "Qwen3", "transformer"),
    ("Qwen/Qwen2.5-7B-Instruct-1M", 7.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen1.5-1.8B", 1.8, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-1.8B-Chat", 1.8, "Qwen1.5", "transformer"),
    ("meta-llama/Llama-3.2-1B", 1.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-1B-Instruct", 1.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-2-7b-hf", 7.0, "Llama2", "transformer"),
    ("meta-llama/Llama-2-7b-chat-hf", 7.0, "Llama2", "transformer"),
    ("mistralai/Mistral-7B-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-v0.2", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-v0.3", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.2", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.0, "Mistral", "transformer"),
    ("mistralai/Mathstral-7B-v0.1", 7.0, "Mathstral", "transformer"),
    ("microsoft/phi-1", 1.3, "Phi", "transformer"),
    ("microsoft/phi-1_5", 1.3, "Phi", "transformer"),
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3-mini-128k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, "Phi3.5", "transformer"),
    ("microsoft/Phi-4-mini-instruct", 3.8, "Phi4", "transformer"),
    ("microsoft/Phi-4-mini-flash-reasoning", 3.8, "Phi4", "transformer"),
    ("microsoft/Orca-2-7b", 7.0, "Orca2", "transformer"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),  # base only (not -it)
    ("google/gemma-7b", 7.0, "Gemma1", "transformer"),     # base only
    ("google/gemma-2b", 2.5, "Gemma1", "transformer"),
    ("stabilityai/stablelm-2-1_6b", 1.6, "StableLM2", "transformer"),
    ("stabilityai/stablelm-2-zephyr-1_6b", 1.6, "StableLM2", "transformer"),
    ("stabilityai/stablelm-zephyr-3b", 3.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-3b-4e1t", 3.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-base-alpha-7b", 7.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-tuned-alpha-7b", 7.0, "StableLM", "transformer"),
    ("stabilityai/stable-code-3b", 3.0, "StableCode", "transformer"),
    ("HuggingFaceTB/SmolLM2-360M", 0.36, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B", 1.7, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM-360M", 0.36, "SmolLM", "transformer"),
    ("HuggingFaceTB/SmolLM-1.7B", 1.7, "SmolLM", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B", 3.0, "SmolLM3", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B-Base", 3.0, "SmolLM3", "transformer"),
    ("openbmb/MiniCPM-2B-sft-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-2B-dpo-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-1B-sft-bf16", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-S-1B-sft", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM3-4B", 4.0, "MiniCPM3", "transformer"),
    ("OpenBMB/MiniCPM4-0.5B", 0.5, "MiniCPM4", "transformer"),
    ("ibm-granite/granite-3.0-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.0-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.1-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.1-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.2-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm/granite-7b-base", 7.0, "Granite7", "transformer"),
    ("01-ai/Yi-6B", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-6B-Chat", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-1.5-6B", 6.0, "Yi1.5", "transformer"),
    ("01-ai/Yi-1.5-6B-Chat", 6.0, "Yi1.5", "transformer"),
    ("deepseek-ai/deepseek-llm-7b-base", 7.0, "DeepSeek", "transformer"),
    ("deepseek-ai/deepseek-llm-7b-chat", 7.0, "DeepSeek", "transformer"),
    ("deepseek-ai/deepseek-math-7b-base", 7.0, "DeepSeek-Math", "transformer"),
    ("deepseek-ai/deepseek-math-7b-instruct", 7.0, "DeepSeek-Math", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-base", 1.3, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-instruct", 1.3, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-base", 6.7, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", 6.7, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1.5, "DeepSeek-R1-Distill", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, "DeepSeek-R1-Distill", "transformer"),
    ("nvidia/Nemotron-Mini-4B-Instruct", 4.0, "Nemotron", "transformer"),
    ("nvidia/Minitron-4B-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Width-Base", 4.0, "Minitron", "transformer"),
    ("HuggingFaceH4/zephyr-7b-beta", 7.0, "Zephyr", "transformer"),
    ("HuggingFaceH4/zephyr-7b-alpha", 7.0, "Zephyr", "transformer"),
    ("lmsys/vicuna-7b-v1.5", 7.0, "Vicuna", "transformer"),
    ("lmsys/vicuna-7b-v1.3", 7.0, "Vicuna", "transformer"),
    ("openchat/openchat-3.5-0106", 7.0, "OpenChat", "transformer"),
    ("berkeley-nest/Starling-LM-7B-alpha", 7.0, "Starling", "transformer"),
    ("Nexusflow/Starling-LM-7B-beta", 7.0, "Starling", "transformer"),
    ("teknium/OpenHermes-2.5-Mistral-7B", 7.0, "OpenHermes", "transformer"),
    ("NousResearch/Hermes-2-Pro-Mistral-7B", 7.0, "Hermes", "transformer"),
    ("Open-Orca/Mistral-7B-OpenOrca", 7.0, "OpenOrca", "transformer"),
    ("Intel/neural-chat-7b-v3-3", 7.0, "NeuralChat", "transformer"),
    ("Intel/neural-chat-7b-v3-1", 7.0, "NeuralChat", "transformer"),
    ("cognitivecomputations/dolphin-2.6-mistral-7b", 7.0, "Dolphin", "transformer"),
    ("cognitivecomputations/dolphin-2.9.4-mistral-7b", 7.0, "Dolphin", "transformer"),
    ("argilla/notus-7b-v1", 7.0, "Notus", "transformer"),
    ("amazon/MistralLite", 7.0, "MistralLite", "transformer"),
    ("WizardLMTeam/WizardMath-7B-V1.1", 7.0, "WizardMath", "transformer"),
    ("meta-math/MetaMath-7B-V1.0", 7.0, "MetaMath", "transformer"),
    ("EleutherAI/llemma_7b", 7.0, "Llemma", "transformer"),
    ("AI-MO/NuminaMath-7B-TIR", 7.0, "NuminaMath", "transformer"),
    ("bigcode/starcoder2-3b", 3.0, "StarCoder2", "transformer"),
    ("bigcode/starcoder2-7b", 7.0, "StarCoder2", "transformer"),
    ("bigcode/starcoderbase-1b", 1.0, "StarCoder", "transformer"),
    ("bigcode/starcoderbase-3b", 3.0, "StarCoder", "transformer"),
    ("bigcode/santacoder", 1.1, "SantaCoder", "transformer"),
    ("codellama/CodeLlama-7b-hf", 7.0, "CodeLlama", "transformer"),
    ("codellama/CodeLlama-7b-Instruct-hf", 7.0, "CodeLlama", "transformer"),
    ("codellama/CodeLlama-7b-Python-hf", 7.0, "CodeLlama", "transformer"),
    ("Salesforce/codegen-350M-mono", 0.35, "CodeGen", "transformer"),
    ("Salesforce/codegen-2B-mono", 2.0, "CodeGen", "transformer"),
    ("Salesforce/codegen-6B-mono", 6.0, "CodeGen", "transformer"),
    ("defog/sqlcoder-7b-2", 7.0, "SQLCoder", "transformer"),
    ("ise-uiuc/Magicoder-S-DS-6.7B", 6.7, "Magicoder", "transformer"),
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("databricks/dolly-v2-7b", 7.0, "Dolly", "transformer"),
    ("Writer/Palmyra-3B", 3.0, "Palmyra", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", 2.4, "EXAONE", "transformer"),
    ("kakaocorp/kanana-nano-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("kakaocorp/kanana-1.5-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("yanolja/EEVE-Korean-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", 0.5, "HyperCLOVAX", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", 1.5, "HyperCLOVAX", "transformer"),
    ("llm-jp/llm-jp-3-1.8b", 1.8, "LLM-JP", "transformer"),
    ("llm-jp/llm-jp-3-3.7b", 3.7, "LLM-JP", "transformer"),
    ("line-corporation/japanese-large-lm-1.7b", 1.7, "LINE-LM", "transformer"),
    ("line-corporation/japanese-large-lm-3.6b", 3.6, "LINE-LM", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("rinna/japanese-gpt-1b", 1.0, "rinna-GPT", "transformer"),
    ("cyberagent/open-calm-1b", 1.0, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-3b", 3.0, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-7b", 7.0, "OpenCALM", "transformer"),
    ("tokyotech-llm/Swallow-7b-hf", 7.0, "Swallow", "transformer"),
    ("tokyotech-llm/Swallow-7b-instruct-hf", 7.0, "Swallow", "transformer"),
    ("elyza/ELYZA-japanese-Llama-2-7b", 7.0, "ELYZA", "transformer"),
    ("sbintuitions/sarashina2-7b", 7.0, "Sarashina", "transformer"),
    ("openai-community/gpt2-medium", 0.36, "GPT-2", "transformer"),
    ("openai-community/gpt2-large", 0.77, "GPT-2", "transformer"),
    ("openai-community/gpt2-xl", 1.5, "GPT-2", "transformer"),
    ("EleutherAI/gpt-j-6b", 6.0, "GPT-J", "transformer"),
    ("bigscience/bloom-560m", 0.56, "BLOOM", "transformer"),
    ("bigscience/bloom-1b1", 1.1, "BLOOM", "transformer"),
    ("bigscience/bloom-1b7", 1.7, "BLOOM", "transformer"),
    ("bigscience/bloom-3b", 3.0, "BLOOM", "transformer"),
    ("bigscience/bloom-7b1", 7.0, "BLOOM", "transformer"),  # 7.1 clipped? use 7.0
    ("h2oai/h2o-danube2-1.8b-base", 1.8, "Danube", "transformer"),  # v2 not v3
    ("h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b", 7.0, "h2oGPT", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-1.3B", 1.3, "Sheared-LLaMA", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-2.7B", 2.7, "Sheared-LLaMA", "transformer"),
    ("openlm-research/open_llama_3b_v2", 3.0, "OpenLLaMA", "transformer"),
    ("openlm-research/open_llama_7b_v2", 7.0, "OpenLLaMA", "transformer"),
    ("Deci/DeciLM-7B", 7.0, "DeciLM", "transformer"),
    ("Deci/DeciLM-7B-instruct", 7.0, "DeciLM", "transformer"),
    ("LumiOpen/Viking-7B", 7.0, "Viking", "transformer"),
    ("m-a-p/neo_7b", 7.0, "NEO", "transformer"),
    ("Biomistral/BioMistral-7B", 7.0, "BioMistral", "transformer"),
    ("epfl-llm/meditron-7b", 7.0, "Meditron", "transformer"),
    ("CohereForAI/c4ai-command-r7b-12-2024", 7.0, "Command-R", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Base", 7.0, "Baichuan2", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Chat", 7.0, "Baichuan2", "transformer"),
    ("BAAI/Aquila2-7B", 7.0, "Aquila2", "transformer"),
    ("THUDM/chatglm3-6b", 6.0, "ChatGLM3", "transformer"),
    ("THUDM/chatglm2-6b", 6.0, "ChatGLM2", "transformer"),
    ("THUDM/codegeex2-6b", 6.0, "CodeGeeX2", "transformer"),
    ("replit/replit-code-v1-3b", 3.0, "Replit", "transformer"),
    ("facebook/MobileLLM-350M", 0.35, "MobileLLM", "transformer"),
    ("facebook/MobileLLM-1B", 1.0, "MobileLLM", "transformer"),
    ("microsoft/DialoGPT-medium", 0.35, "DialoGPT", "transformer"),
    ("microsoft/DialoGPT-large", 0.77, "DialoGPT", "transformer"),
    ("allenai/OLMo-2-0425-1B", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-0425-1B-Instruct", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-1B-hf", 1.0, "OLMo", "transformer"),
    ("allenai/OLMo-1B-0724-hf", 1.0, "OLMo", "transformer"),
    ("allenai/OLMo-2-1124-7B", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-1124-7B-Instruct", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-7B-0724-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-7B-Instruct-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-1.7-7B-hf", 7.0, "OLMo", "transformer"),
    # ---- extra mid-size decoder-only (vLLM-safe) to hit n=200 with halved tails ----
    ("Qwen/Qwen2-Math-1.5B", 1.5, "Qwen2-Math", "transformer"),
    ("Qwen/Qwen2-Math-1.5B-Instruct", 1.5, "Qwen2-Math", "transformer"),
    ("Qwen/Qwen2.5-Math-1.5B-Instruct", 1.5, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", 3.0, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Math-7B-Instruct", 7.0, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen3-0.6B", 0.6, "Qwen3", "transformer"),
    ("Qwen/Qwen3-1.7B", 1.7, "Qwen3", "transformer"),
    ("Qwen/Qwen3-4B", 4.0, "Qwen3", "transformer"),
    ("Qwen/Qwen3-4B-Instruct-2507", 4.0, "Qwen3", "transformer"),
    ("Qwen/Qwen3-1.7B-Instruct-2507", 1.7, "Qwen3", "transformer"),
    ("Qwen/Qwen3-0.6B-Instruct-2507", 0.6, "Qwen3", "transformer"),
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2-0.5B-Instruct", 0.5, "Qwen2", "transformer"),
    ("Qwen/Qwen2-1.5B-Instruct", 1.5, "Qwen2", "transformer"),
    ("Qwen/Qwen1.5-0.5B", 0.5, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-0.5B-Chat", 0.5, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-1.8B", 1.8, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-1.8B-Chat", 1.8, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-4B", 4.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-4B-Chat", 4.0, "Qwen1.5", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, "SmolLM2", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B", 3.0, "SmolLM3", "transformer"),
    ("HuggingFaceTB/SmolLM3-3B-Base", 3.0, "SmolLM3", "transformer"),
    ("HuggingFaceTB/SmolLM-1.7B-Instruct", 1.7, "SmolLM", "transformer"),
    ("HuggingFaceTB/SmolLM-360M-Instruct", 0.36, "SmolLM", "transformer"),
    ("microsoft/phi-1", 1.3, "Phi", "transformer"),
    ("microsoft/phi-1_5", 1.3, "Phi", "transformer"),
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3-mini-128k-instruct", 3.8, "Phi3", "transformer"),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, "Phi3.5", "transformer"),
    ("microsoft/Phi-4-mini-instruct", 3.8, "Phi4", "transformer"),
    ("microsoft/Phi-4-mini-flash-reasoning", 3.8, "Phi4", "transformer"),
    ("microsoft/Phi-3-small-8k-instruct", 7.0, "Phi3-small", "transformer"),
    ("ibm-granite/granite-3.0-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.0-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.1-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.1-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.2-2b-instruct", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-2b-base", 2.0, "Granite", "transformer"),
    ("ibm-granite/granite-3.3-2b-instruct", 2.0, "Granite", "transformer"),
    ("tiiuae/Falcon3-1B-Base", 1.0, "Falcon3", "transformer"),
    ("tiiuae/Falcon3-1B-Instruct", 1.0, "Falcon3", "transformer"),
    ("tiiuae/Falcon3-3B-Base", 3.0, "Falcon3", "transformer"),
    ("tiiuae/Falcon3-3B-Instruct", 3.0, "Falcon3", "transformer"),
    ("tiiuae/Falcon3-7B-Base", 7.0, "Falcon3", "transformer"),
    ("tiiuae/Falcon3-7B-Instruct", 7.0, "Falcon3", "transformer"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),
    ("google/gemma-2b", 2.5, "Gemma1", "transformer"),
    ("google/gemma-7b", 7.0, "Gemma1", "transformer"),
    ("meta-llama/Llama-3.2-1B", 1.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-1B-Instruct", 1.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.0, "Llama3.2", "transformer"),
    ("stabilityai/stablelm-2-1_6b", 1.6, "StableLM2", "transformer"),
    ("stabilityai/stablelm-2-zephyr-1_6b", 1.6, "StableLM2", "transformer"),
    ("stabilityai/stablelm-zephyr-3b", 3.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-3b-4e1t", 3.0, "StableLM", "transformer"),
    ("stabilityai/stable-code-3b", 3.0, "StableCode", "transformer"),
    ("openbmb/MiniCPM-2B-sft-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-2B-dpo-bf16", 2.7, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-1B-sft-bf16", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM-S-1B-sft", 1.0, "MiniCPM", "transformer"),
    ("openbmb/MiniCPM3-4B", 4.0, "MiniCPM3", "transformer"),
    ("OpenBMB/MiniCPM4-0.5B", 0.5, "MiniCPM4", "transformer"),
    ("nvidia/Nemotron-Mini-4B-Instruct", 4.0, "Nemotron", "transformer"),
    ("nvidia/Minitron-4B-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Width-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Depth-Base", 4.0, "Minitron", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", 2.4, "EXAONE", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", 7.8, "EXAONE", "transformer"),  # over
    ("kakaocorp/kanana-nano-2.1b-base", 2.1, "Kanana", "transformer"),
    ("kakaocorp/kanana-nano-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("kakaocorp/kanana-1.5-2.1b-instruct", 2.1, "Kanana", "transformer"),
    ("yanolja/EEVE-Korean-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("yanolja/EEVE-Korean-Instruct-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", 0.5, "HyperCLOVAX", "transformer"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", 1.5, "HyperCLOVAX", "transformer"),
    ("llm-jp/llm-jp-3-1.8b", 1.8, "LLM-JP", "transformer"),
    ("llm-jp/llm-jp-3-1.8b-instruct", 1.8, "LLM-JP", "transformer"),
    ("llm-jp/llm-jp-3-3.7b", 3.7, "LLM-JP", "transformer"),
    ("llm-jp/llm-jp-3-3.7b-instruct", 3.7, "LLM-JP", "transformer"),
    ("line-corporation/japanese-large-lm-1.7b", 1.7, "LINE-LM", "transformer"),
    ("line-corporation/japanese-large-lm-3.6b", 3.6, "LINE-LM", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b-instruction-sft", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("cyberagent/open-calm-small", 0.4, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-medium", 0.4, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-large", 0.8, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-1b", 1.0, "OpenCALM", "transformer"),
    ("cyberagent/open-calm-3b", 3.0, "OpenCALM", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-base", 1.3, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-instruct", 1.3, "DeepSeek-Coder", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1.5, "DeepSeek-R1-Distill", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-base", 6.7, "DeepSeek-Coder", "transformer"),
    ("bigcode/starcoder2-3b", 3.0, "StarCoder2", "transformer"),
    ("bigcode/starcoderbase-1b", 1.0, "StarCoder", "transformer"),
    ("bigcode/starcoderbase-3b", 3.0, "StarCoder", "transformer"),
    ("bigcode/santacoder", 1.1, "SantaCoder", "transformer"),
    ("Salesforce/codegen-350M-mono", 0.35, "CodeGen", "transformer"),
    ("Salesforce/codegen-2B-mono", 2.0, "CodeGen", "transformer"),
    ("Salesforce/codegen-2B-multi", 2.0, "CodeGen", "transformer"),
    ("replit/replit-code-v1-3b", 3.0, "Replit", "transformer"),
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("Writer/Palmyra-3B", 3.0, "Palmyra", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("openbmb/cpm-bee-1b", 1.0, "CPM-Bee", "transformer"),
    ("openbmb/cpm-bee-2b", 2.0, "CPM-Bee", "transformer"),
    ("openbmb/cpm-bee-5b", 5.0, "CPM-Bee", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-1.3B", 1.3, "Sheared-LLaMA", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-2.7B", 2.7, "Sheared-LLaMA", "transformer"),
    ("openlm-research/open_llama_3b", 3.0, "OpenLLaMA", "transformer"),
    ("openlm-research/open_llama_3b_v2", 3.0, "OpenLLaMA", "transformer"),
    ("h2oai/h2o-danube2-1.8b-base", 1.8, "Danube", "transformer"),
    ("h2oai/h2o-danube2-1.8b-chat", 1.8, "Danube", "transformer"),
    ("facebook/MobileLLM-350M", 0.35, "MobileLLM", "transformer"),
    ("facebook/MobileLLM-600M", 0.6, "MobileLLM", "transformer"),
    ("facebook/MobileLLM-1B", 1.0, "MobileLLM", "transformer"),
    ("facebook/xglm-564M", 0.56, "XGLM", "transformer"),
    ("facebook/xglm-1.7B", 1.7, "XGLM", "transformer"),
    ("facebook/xglm-2.9B", 2.9, "XGLM", "transformer"),
    ("facebook/xglm-4.5B", 4.5, "XGLM", "transformer"),
    ("EleutherAI/polyglot-ko-1.3b", 1.3, "Polyglot-KO", "transformer"),
    ("EleutherAI/polyglot-ko-3.8b", 3.8, "Polyglot-KO", "transformer"),
    ("EleutherAI/polyglot-ko-5.8b", 5.8, "Polyglot-KO", "transformer"),
    ("beomi/KoAlpaca-Polyglot-5.8B", 5.8, "KoAlpaca", "transformer"),
    ("nlpai-lab/KULLM-Polyglot-5.8B-v2", 5.8, "KULLM", "transformer"),
    ("kakaobrain/kogpt", 6.0, "KoGPT", "transformer"),
    ("01-ai/Yi-6B", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-6B-Chat", 6.0, "Yi", "transformer"),
    ("01-ai/Yi-1.5-6B", 6.0, "Yi1.5", "transformer"),
    ("01-ai/Yi-1.5-6B-Chat", 6.0, "Yi1.5", "transformer"),
    ("THUDM/chatglm3-6b", 6.0, "ChatGLM3", "transformer"),
    ("THUDM/chatglm2-6b", 6.0, "ChatGLM2", "transformer"),
    ("THUDM/codegeex2-6b", 6.0, "CodeGeeX2", "transformer"),
    ("EleutherAI/gpt-j-6b", 6.0, "GPT-J", "transformer"),
    ("togethercomputer/GPT-JT-6B-v1", 6.0, "GPT-JT", "transformer"),
    ("KoboldAI/GPT-J-6B-Janeway", 6.0, "Janeway", "transformer"),
    ("Salesforce/codegen-6B-mono", 6.0, "CodeGen", "transformer"),
    ("Salesforce/codegen-6B-multi", 6.0, "CodeGen", "transformer"),
    ("bigscience/bloom-560m", 0.56, "BLOOM", "transformer"),
    ("bigscience/bloom-1b1", 1.1, "BLOOM", "transformer"),
    ("bigscience/bloom-1b7", 1.7, "BLOOM", "transformer"),
    ("bigscience/bloom-3b", 3.0, "BLOOM", "transformer"),
    ("MBZUAI/LaMini-GPT-124M", 0.12, "LaMini", "transformer"),  # below
    ("MBZUAI/LaMini-GPT-774M", 0.77, "LaMini", "transformer"),
    ("MBZUAI/LaMini-GPT-1.5B", 1.5, "LaMini", "transformer"),
    ("microsoft/DialoGPT-medium", 0.35, "DialoGPT", "transformer"),
    ("microsoft/DialoGPT-large", 0.77, "DialoGPT", "transformer"),
    ("microsoft/biogpt", 0.35, "BioGPT", "transformer"),
    ("microsoft/BioGPT-Large", 1.5, "BioGPT", "transformer"),
    ("stanford-crfm/BioMedLM", 2.7, "BioMedLM", "transformer"),
    ("openai-community/gpt2", 0.12, "GPT-2", "transformer"),  # below
    ("openai-community/gpt2-medium", 0.36, "GPT-2", "transformer"),
    ("openai-community/gpt2-large", 0.77, "GPT-2", "transformer"),
    ("openai-community/gpt2-xl", 1.5, "GPT-2", "transformer"),
    ("facebook/opt-350m", 0.35, "OPT", "transformer"),  # small OPT; larger OPTs blocked
    # community mid-size finetunes (distinct orgs; decoder-only Llama/Qwen/Phi/Gemma bases)
    ("unsloth/Llama-3.2-1B", 1.0, "Llama3.2-unsloth", "transformer"),
    ("unsloth/Llama-3.2-1B-Instruct", 1.0, "Llama3.2-unsloth", "transformer"),
    ("unsloth/Llama-3.2-3B", 3.0, "Llama3.2-unsloth", "transformer"),
    ("unsloth/Llama-3.2-3B-Instruct", 3.0, "Llama3.2-unsloth", "transformer"),
    ("unsloth/Qwen2.5-0.5B", 0.5, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Qwen2.5-0.5B-Instruct", 0.5, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Qwen2.5-1.5B", 1.5, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Qwen2.5-1.5B-Instruct", 1.5, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Qwen2.5-3B", 3.0, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5-unsloth", "transformer"),
    ("unsloth/Phi-3-mini-4k-instruct", 3.8, "Phi3-unsloth", "transformer"),
    ("unsloth/Phi-3.5-mini-instruct", 3.8, "Phi3.5-unsloth", "transformer"),
    ("unsloth/gemma-2-2b", 2.0, "Gemma2-unsloth", "transformer"),
    ("unsloth/SmolLM2-1.7B", 1.7, "SmolLM2-unsloth", "transformer"),
    ("unsloth/SmolLM2-1.7B-Instruct", 1.7, "SmolLM2-unsloth", "transformer"),
    ("NousResearch/Hermes-3-Llama-3.2-3B", 3.0, "Hermes3", "transformer"),
    ("NousResearch/Hermes-2-Pro-Llama-3-8B", 8.0, "Hermes2", "transformer"),  # over
    ("cognitivecomputations/dolphin-2.9.4-llama3.1-8b", 8.0, "Dolphin", "transformer"),  # over
    ("cognitivecomputations/dolphin-2.9-llama3-8b", 8.0, "Dolphin", "transformer"),  # over
    ("mlabonne/NeuralDaredevil-8B-abliterated", 8.0, "NeuralDaredevil", "transformer"),  # over
    ("OpenPipe/mistral-ft-optimized-1218", 7.0, "Mistral-FT", "transformer"),
    ("HuggingFaceH4/zephyr-7b-beta", 7.0, "Zephyr", "transformer"),
    ("alignment-handbook/zephyr-7b-sft-full", 7.0, "Zephyr", "transformer"),
    ("HuggingFaceH4/mistral-7b-sft-beta", 7.0, "Mistral-SFT", "transformer"),
    ("argilla/notus-7b-v1", 7.0, "Notus", "transformer"),
    ("berkeley-nest/Starling-LM-7B-alpha", 7.0, "Starling", "transformer"),
    ("Nexusflow/Starling-LM-7B-beta", 7.0, "Starling", "transformer"),
    ("openchat/openchat-3.5-0106", 7.0, "OpenChat", "transformer"),
    ("openchat/openchat-3.5-1210", 7.0, "OpenChat", "transformer"),
    ("teknium/OpenHermes-2.5-Mistral-7B", 7.0, "OpenHermes", "transformer"),
    ("NousResearch/Hermes-2-Pro-Mistral-7B", 7.0, "Hermes", "transformer"),
    ("Open-Orca/Mistral-7B-OpenOrca", 7.0, "OpenOrca", "transformer"),
    ("Intel/neural-chat-7b-v3-3", 7.0, "NeuralChat", "transformer"),
    ("amazon/MistralLite", 7.0, "MistralLite", "transformer"),
    ("WizardLMTeam/WizardMath-7B-V1.1", 7.0, "WizardMath", "transformer"),
    ("meta-math/MetaMath-7B-V1.0", 7.0, "MetaMath", "transformer"),
    ("EleutherAI/llemma_7b", 7.0, "Llemma", "transformer"),
    ("AI-MO/NuminaMath-7B-TIR", 7.0, "NuminaMath", "transformer"),
    ("mistralai/Mathstral-7B-v0.1", 7.0, "Mathstral", "transformer"),
    ("microsoft/Orca-2-7b", 7.0, "Orca2", "transformer"),
    ("cognitivecomputations/dolphin-2.6-mistral-7b", 7.0, "Dolphin", "transformer"),
    ("cognitivecomputations/dolphin-2.9.4-mistral-7b", 7.0, "Dolphin", "transformer"),
    ("jondurbin/airoboros-mistral2.2-7b", 7.0, "Airoboros", "transformer"),
    ("garage-bAInd/Platypus2-7B", 7.0, "Platypus", "transformer"),
    ("Xwin-LM/Xwin-LM-7B-V0.2", 7.0, "Xwin", "transformer"),
    ("lmsys/vicuna-7b-v1.5", 7.0, "Vicuna", "transformer"),
    ("lmsys/vicuna-7b-v1.3", 7.0, "Vicuna", "transformer"),
    ("TheBloke/guanaco-7B-HF", 7.0, "Guanaco", "transformer"),
    ("project-baize/baize-v2-7b", 7.0, "Baize", "transformer"),
    ("PygmalionAI/pygmalion-2-7b", 7.0, "Pygmalion", "transformer"),
    ("NousResearch/Nous-Hermes-llama-2-7b", 7.0, "Nous-Hermes", "transformer"),
    ("meta-llama/Llama-2-7b-hf", 7.0, "Llama2", "transformer"),
    ("meta-llama/Llama-2-7b-chat-hf", 7.0, "Llama2", "transformer"),
    ("codellama/CodeLlama-7b-hf", 7.0, "CodeLlama", "transformer"),
    ("codellama/CodeLlama-7b-Instruct-hf", 7.0, "CodeLlama", "transformer"),
    ("codellama/CodeLlama-7b-Python-hf", 7.0, "CodeLlama", "transformer"),
    ("bigcode/starcoder2-7b", 7.0, "StarCoder2", "transformer"),
    ("deepseek-ai/deepseek-llm-7b-base", 7.0, "DeepSeek", "transformer"),
    ("deepseek-ai/deepseek-llm-7b-chat", 7.0, "DeepSeek", "transformer"),
    ("deepseek-ai/deepseek-math-7b-base", 7.0, "DeepSeek-Math", "transformer"),
    ("deepseek-ai/deepseek-math-7b-instruct", 7.0, "DeepSeek-Math", "transformer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, "DeepSeek-R1-Distill", "transformer"),
    ("Qwen/Qwen2.5-7B", 7.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-7B-Instruct", 7.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-Coder-7B", 7.0, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", 7.0, "Qwen2.5-Coder", "transformer"),
    ("Qwen/Qwen2.5-Math-7B", 7.0, "Qwen2.5-Math", "transformer"),
    ("Qwen/Qwen2-7B", 7.0, "Qwen2", "transformer"),
    ("Qwen/Qwen2-7B-Instruct", 7.0, "Qwen2", "transformer"),
    ("Qwen/Qwen1.5-7B", 7.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-7B-Chat", 7.0, "Qwen1.5", "transformer"),
    ("mistralai/Mistral-7B-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-v0.2", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-v0.3", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.1", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.2", 7.0, "Mistral", "transformer"),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.0, "Mistral", "transformer"),
    ("Biomistral/BioMistral-7B", 7.0, "BioMistral", "transformer"),
    ("epfl-llm/meditron-7b", 7.0, "Meditron", "transformer"),
    ("Deci/DeciLM-7B", 7.0, "DeciLM", "transformer"),
    ("Deci/DeciLM-7B-instruct", 7.0, "DeciLM", "transformer"),
    ("LumiOpen/Viking-7B", 7.0, "Viking", "transformer"),
    ("m-a-p/neo_7b", 7.0, "NEO", "transformer"),
    ("ibm/granite-7b-base", 7.0, "Granite7", "transformer"),
    ("CohereForAI/c4ai-command-r7b-12-2024", 7.0, "Command-R", "transformer"),
    ("tencent/Hunyuan-7B-Pretrain", 7.0, "Hunyuan", "transformer"),
    ("tencent/Hunyuan-7B-Instruct", 7.0, "Hunyuan", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Base", 7.0, "Baichuan2", "transformer"),
    ("baichuan-inc/Baichuan2-7B-Chat", 7.0, "Baichuan2", "transformer"),
    ("BAAI/Aquila2-7B", 7.0, "Aquila2", "transformer"),
    ("tokyotech-llm/Swallow-7b-hf", 7.0, "Swallow", "transformer"),
    ("tokyotech-llm/Swallow-7b-instruct-hf", 7.0, "Swallow", "transformer"),
    ("elyza/ELYZA-japanese-Llama-2-7b", 7.0, "ELYZA", "transformer"),
    ("sbintuitions/sarashina2-7b", 7.0, "Sarashina", "transformer"),
    ("cyberagent/open-calm-7b", 7.0, "OpenCALM", "transformer"),
    ("stabilityai/stablelm-base-alpha-7b", 7.0, "StableLM", "transformer"),
    ("stabilityai/stablelm-tuned-alpha-7b", 7.0, "StableLM", "transformer"),
    ("rinna/nekomata-7b", 7.0, "Nekomata", "transformer"),
    ("rinna/youri-7b", 7.0, "Youri", "transformer"),
    ("skt/A.X-3.1", 7.0, "A.X", "transformer"),
    ("defog/sqlcoder-7b-2", 7.0, "SQLCoder", "transformer"),
    ("ise-uiuc/Magicoder-S-DS-6.7B", 6.7, "Magicoder", "transformer"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", 6.7, "DeepSeek-Coder", "transformer"),
    # more distinct 2-4B instruct/base pairs
    ("Qwen/Qwen2.5-3B", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-Coder-3B", 3.0, "Qwen2.5-Coder", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("microsoft/Phi-4-mini-instruct", 3.8, "Phi4", "transformer"),
    ("allenai/OLMo-2-1124-7B-DPO", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-1124-7B-SFT", 7.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-0425-1B-DPO", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-2-0425-1B-SFT", 1.0, "OLMo2", "transformer"),
    ("allenai/OLMo-7B-Twin-2T-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-7B-Instruct-hf", 7.0, "OLMo", "transformer"),
    ("allenai/OLMo-1.7-7B-hf", 7.0, "OLMo", "transformer"),
    # additional mid finetunes / regional LLMs
    ("upstage/SOLAR-10.7B-v1.0", 10.7, "SOLAR", "transformer"),  # over
    ("KF-Complex/Korean-Qwen2.5-7B-Instruct", 7.0, "Korean-Qwen", "transformer"),
    ("beomi/llama-2-ko-7b", 7.0, "Llama2-KO", "transformer"),
    ("beomi/gemma-ko-2b", 2.5, "Gemma-KO", "transformer"),
    ("beomi/gemma-ko-7b", 7.0, "Gemma-KO", "transformer"),
    ("nlpai-lab/lee-solar-10.7b", 10.7, "SOLAR", "transformer"),  # over
    ("davidkim205/komt-mistral-7b-v1", 7.0, "KOMT", "transformer"),
    ("heegyu/kogpt-trinity-1.2B", 1.2, "KoGPT-Trinity", "transformer"),
    ("skt/kogpt2-base-v2", 0.13, "KoGPT2", "transformer"),  # below
    ("EleutherAI/polyglot-ko-1.3b", 1.3, "Polyglot-KO", "transformer"),
    ("facebook/xglm-564M", 0.56, "XGLM", "transformer"),
    ("ai-forever/mGPT", 1.3, "mGPT", "transformer"),
    ("ai-forever/mGPT-1.3B", 1.3, "mGPT", "transformer"),
    ("ai-forever/rugpt3large_based_on_gpt2", 0.76, "ruGPT3", "transformer"),
    ("ai-forever/rugpt3medium_based_on_gpt2", 0.36, "ruGPT3", "transformer"),
    ("ai-forever/rugpt3small_based_on_gpt2", 0.12, "ruGPT3", "transformer"),  # below
    ("sberbank-ai/rugpt3large_based_on_gpt2", 0.76, "ruGPT3", "transformer"),
    ("LightOn/lyso-3b", 3.0, "Lyso", "transformer"),
    ("LightOn/lyso-3b-chat", 3.0, "Lyso", "transformer"),
    ("occiglot/occiglot-7b-eu5", 7.0, "Occiglot", "transformer"),
    ("occiglot/occiglot-7b-de-en", 7.0, "Occiglot", "transformer"),
    ("malteos/gpt2-xl-wechsel-german", 1.5, "GPT2-DE", "transformer"),
    ("dbmdz/german-gpt2", 0.12, "GPT2-DE", "transformer"),  # below
    ("benjamin/gerpt2-large", 0.77, "GerPT2", "transformer"),
    ("flaubert/flaubert_large_cased", 0.37, "FlauBERT", "encoder"),  # encoder skip
    ("asi/gpt-fr-cased-base", 0.12, "GPT-FR", "transformer"),  # below
    ("asi/gpt-fr-cased-small", 0.12, "GPT-FR", "transformer"),  # below
    ("cedille/fr-boris", 1.3, "Boris", "transformer"),
    ("Geotrend/bert-base-fr-cased", 0.11, "BERT-FR", "encoder"),  # below/encoder
    ("croissantllm/CroissantLLMBase-1.3B", 1.3, "CroissantLLM", "transformer"),
    ("croissantllm/CroissantLLMChat-v0.1", 1.3, "CroissantLLM", "transformer"),
    ("swiss-ai/Apertus-8B-2509", 8.0, "Apertus", "transformer"),  # over
    ("utter-project/EuroLLM-1.7B", 1.7, "EuroLLM", "transformer"),
    ("utter-project/EuroLLM-1.7B-Instruct", 1.7, "EuroLLM", "transformer"),
    ("utter-project/EuroLLM-9B", 9.0, "EuroLLM", "transformer"),  # over
    ("openGPT-X/Teuken-7B-instruct-research-v0.4", 7.0, "Teuken", "transformer"),
    ("LeoLM/leo-hessianai-7b", 7.0, "LeoLM", "transformer"),
    ("LeoLM/leo-hessianai-7b-chat", 7.0, "LeoLM", "transformer"),
    ("malteos/bloom-6b4-clp-german", 6.4, "BLOOM-DE", "transformer"),
    ("german-nlp-group/germandpr-distilbert", 0.07, "GermanDPR", "encoder"),  # below
    ("deepset/gbert-large", 0.34, "GBERT", "encoder"),
    ("dbmdz/bert-base-german-cased", 0.11, "BERT-DE", "encoder"),
    ("LumiOpen/Poro-34B", 34.0, "Poro", "transformer"),  # over
    ("LumiOpen/Viking-7B", 7.0, "Viking", "transformer"),
    ("LumiOpen/Llama-Poro-2-8B-Instruct", 8.0, "Poro2", "transformer"),  # over
    ("AI-Sweden-Models/gpt-sw3-1.3b", 1.3, "GPT-SW3", "transformer"),
    ("AI-Sweden-Models/gpt-sw3-6.7b", 6.7, "GPT-SW3", "transformer"),
    ("AI-Sweden-Models/gpt-sw3-356m", 0.36, "GPT-SW3", "transformer"),
    ("NbAiLab/nb-gpt-j-6B", 6.0, "NB-GPT-J", "transformer"),
    ("NbAiLab/nb-norbert2", 0.11, "NorBERT", "encoder"),
    ("TurkuNLP/gpt3-finnish-small", 0.12, "GPT3-FI", "transformer"),  # below
    ("TurkuNLP/gpt3-finnish-large", 0.77, "GPT3-FI", "transformer"),
    ("Finnish-NLP/gpt2-finnish", 0.12, "GPT2-FI", "transformer"),  # below
    ("INTC/LLaMA-2-7B-Finnish", 7.0, "Llama2-FI", "transformer"),
    ("RikoteMaster/Llama-3.2-1B-Spanish", 1.0, "Llama3.2-ES", "transformer"),
    ("Iker/Llama-3-Instruct-Neurona-1b", 1.0, "Neurona", "transformer"),
    ("Iker/Llama-3-Instruct-Neurona-2.5b", 2.5, "Neurona", "transformer"),
    ("DeepESP/gpt2-spanish", 0.12, "GPT2-ES", "transformer"),  # below
    ("PlanTL-GOB-ES/gpt2-base-bne", 0.12, "GPT2-ES", "transformer"),  # below
    ("PlanTL-GOB-ES/gpt2-large-bne", 0.77, "GPT2-ES", "transformer"),
    ("BSC-LT/salamandra-2b", 2.0, "Salamandra", "transformer"),
    ("BSC-LT/salamandra-2b-instruct", 2.0, "Salamandra", "transformer"),
    ("BSC-LT/salamandra-7b", 7.0, "Salamandra", "transformer"),
    ("BSC-LT/salamandra-7b-instruct", 7.0, "Salamandra", "transformer"),
    ("projecte-aina/aguila-7b", 7.0, "Aguila", "transformer"),
    ("projecte-aina/FLOR-1.3B", 1.3, "FLOR", "transformer"),
    ("projecte-aina/FLOR-6.3B", 6.3, "FLOR", "transformer"),
    ("cliplab/gpt2-small-catalan", 0.12, "GPT2-CA", "transformer"),  # below
    ("sapienzanlp/Minerva-1B-base-v1.0", 1.0, "Minerva", "transformer"),
    ("sapienzanlp/Minerva-3B-base-v1.0", 3.0, "Minerva", "transformer"),
    ("sapienzanlp/Minerva-7B-base-v1.0", 7.0, "Minerva", "transformer"),
    ("gsarti/it5-base", 0.22, "IT5", "encoder-decoder"),
    ("gsarti/it5-large", 0.74, "IT5", "encoder-decoder"),
    ("musixmatch/umberto-commoncrawl-cased-v1", 0.11, "UmBERTo", "encoder"),
    ("iict-uwb/gpt2-small-czech", 0.12, "GPT2-CS", "transformer"),  # below
    ("MU-NLPC/CzeGPT-2", 0.12, "CzeGPT", "transformer"),  # below
    ("AZILA/Llama3.2-3B-Instruct-Czech", 3.0, "Llama3.2-CS", "transformer"),
    ("speakleash/Bielik-1.5B-v3.0-Instruct", 1.5, "Bielik", "transformer"),
    ("speakleash/Bielik-4.5B-v3.0-Instruct", 4.5, "Bielik", "transformer"),
    ("speakleash/Bielik-7B-v0.1", 7.0, "Bielik", "transformer"),
    ("speakleash/Bielik-11B-v2.3-Instruct", 11.0, "Bielik", "transformer"),  # over
    ("allegro/herbert-base-cased", 0.11, "HerBERT", "encoder"),
    ("sdadas/polish-gpt2-medium", 0.35, "Polish-GPT2", "transformer"),
    ("sdadas/polish-gpt2-large", 0.77, "Polish-GPT2", "transformer"),
    ("sdadas/polish-gpt2-xl", 1.5, "Polish-GPT2", "transformer"),
    ("ORCA_PL/llama2-7b-orca-pl", 7.0, "ORCA-PL", "transformer"),
    ("yandex/YandexGPT-5-Lite-8B-instruct", 8.0, "YandexGPT", "transformer"),  # over
    ("Vikhrmodels/Vikhr-Llama3.2-1B-Instruct-R-21-09-24", 1.0, "Vikhr", "transformer"),
    ("Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", 12.0, "Vikhr", "transformer"),  # over
    ("IlyaGusev/saiga_llama3_8b", 8.0, "Saiga", "transformer"),  # over
    ("IlyaGusev/saiga2_7b_lora", 7.0, "Saiga", "transformer"),
    ("ai-forever/mGPT-13B", 13.0, "mGPT", "transformer"),  # over
    ("INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0", 2.6, "BgGPT", "transformer"),
    ("INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0", 9.0, "BgGPT", "transformer"),  # over
    ("mistralai/Mistral-Nemo-Base-2407", 12.0, "Mistral-Nemo", "transformer"),  # over
    ("microsoft/phi-4", 14.0, "Phi4", "transformer"),  # over
    # more US/global mid-size
    ("allenai/OLMo-2-0425-1B", 1.0, "OLMo2", "transformer"),
    ("HuggingFaceTB/SmolLM2-135M", 0.135, "SmolLM2", "transformer"),  # below
    ("HuggingFaceTB/SmolLM2-135M-Instruct", 0.135, "SmolLM2", "transformer"),  # below
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),
    ("TinyLlama/TinyLlama_v1.1", 1.1, "TinyLlama", "transformer"),  # blocked family
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 1.1, "TinyLlama", "transformer"),
    ("openlm-research/open_llama_3b_v2", 3.0, "OpenLLaMA", "transformer"),
    ("togethercomputer/RedPajama-INCITE-Chat-3B-v1", 3.0, "RedPajama", "transformer"),  # blocked
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("openbmb/cpm-bee-5b", 5.0, "CPM-Bee", "transformer"),
    ("EleutherAI/polyglot-ko-5.8b", 5.8, "Polyglot-KO", "transformer"),
    ("beomi/KoAlpaca-Polyglot-5.8B", 5.8, "KoAlpaca", "transformer"),
    ("nlpai-lab/KULLM-Polyglot-5.8B-v2", 5.8, "KULLM", "transformer"),
    ("facebook/xglm-4.5B", 4.5, "XGLM", "transformer"),
    ("speakleash/Bielik-4.5B-v3.0-Instruct", 4.5, "Bielik", "transformer"),
    ("nvidia/Minitron-4B-Base", 4.0, "Minitron", "transformer"),
    ("nvidia/Nemotron-Mini-4B-Instruct", 4.0, "Nemotron", "transformer"),
    ("openbmb/MiniCPM3-4B", 4.0, "MiniCPM3", "transformer"),
    ("Qwen/Qwen3-4B", 4.0, "Qwen3", "transformer"),
    ("Qwen/Qwen1.5-4B", 4.0, "Qwen1.5", "transformer"),
    ("Qwen/Qwen1.5-4B-Chat", 4.0, "Qwen1.5", "transformer"),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, "Phi3.5", "transformer"),
    ("EleutherAI/polyglot-ko-3.8b", 3.8, "Polyglot-KO", "transformer"),
    ("llm-jp/llm-jp-3-3.7b", 3.7, "LLM-JP", "transformer"),
    ("line-corporation/japanese-large-lm-3.6b", 3.6, "LINE-LM", "transformer"),
    ("rinna/japanese-gpt-neox-3.6b", 3.6, "rinna-GPT-NeoX", "transformer"),
    ("sapienzanlp/Minerva-3B-base-v1.0", 3.0, "Minerva", "transformer"),
    ("BSC-LT/salamandra-2b", 2.0, "Salamandra", "transformer"),
    ("BSC-LT/salamandra-2b-instruct", 2.0, "Salamandra", "transformer"),
    ("IBM/granite-3.3-2b-base", 2.0, "Granite", "transformer"),
    ("IBM/granite-3.3-2b-instruct", 2.0, "Granite", "transformer"),
    # final mid-size fillers (decoder-only, vLLM-safe) to keep both tails halved
    ("Qwen/Qwen2.5-1.5B", 1.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "Qwen2.5", "transformer"),
    ("Qwen/Qwen2-1.5B", 1.5, "Qwen2", "transformer"),
    ("HuggingFaceTB/SmolLM2-1.7B", 1.7, "SmolLM2", "transformer"),
    ("stabilityai/stablelm-2-1_6b", 1.6, "StableLM2", "transformer"),
    ("h2oai/h2o-danube2-1.8b-base", 1.8, "Danube", "transformer"),
    ("openbmb/MiniCPM-2B-sft-bf16", 2.7, "MiniCPM", "transformer"),
    ("princeton-nlp/Sheared-LLaMA-2.7B", 2.7, "Sheared-LLaMA", "transformer"),
    ("databricks/dolly-v2-3b", 3.0, "Dolly", "transformer"),
    ("Writer/Palmyra-3B", 3.0, "Palmyra", "transformer"),
    ("BSC-LT/salamandra-2b-base", 2.0, "Salamandra", "transformer"),
    ("BSC-LT/salamandra-2b-instruct-af", 2.0, "Salamandra", "transformer"),
    ("utter-project/EuroLLM-1.7B-Instruct", 1.7, "EuroLLM", "transformer"),
    ("croissantllm/CroissantLLMBase", 1.3, "CroissantLLM", "transformer"),
    ("AI-Sweden-Models/gpt-sw3-1.3b-instruct", 1.3, "GPT-SW3", "transformer"),
    ("speakleash/Bielik-1.5B-v3", 1.5, "Bielik", "transformer"),
    ("Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct", 1.5, "Vikhr", "transformer"),
    ("deepseek-ai/deepseek-coder-1.3b-base", 1.3, "DeepSeek-Coder", "transformer"),
    ("bigcode/starcoderbase-1b", 1.0, "StarCoder", "transformer"),
    ("Salesforce/codegen-2B-multi", 2.0, "CodeGen", "transformer"),
    ("facebook/xglm-1.7B", 1.7, "XGLM", "transformer"),
    ("line-corporation/japanese-large-lm-1.7b-instruction-sft", 1.7, "LINE-LM", "transformer"),
    ("cyberagent/open-calm-1b", 1.0, "OpenCALM", "transformer"),
    ("EleutherAI/polyglot-ko-1.3b", 1.3, "Polyglot-KO", "transformer"),
    ("projecte-aina/FLOR-1.3B-Instructed", 1.3, "FLOR", "transformer"),
    ("sapienzanlp/Minerva-1B-base-v1.0", 1.0, "Minerva", "transformer"),
    ("Iker/Llama-3.2-1B-Instruct-Spanish", 1.0, "Llama3.2-ES", "transformer"),
    ("beomi/Llama-3-Open-Ko-8B", 8.0, "Llama3-KO", "transformer"),  # over
    ("microsoft/phi-1_5", 1.3, "Phi", "transformer"),
    ("tiiuae/Falcon3-1B-Base", 1.0, "Falcon3", "transformer"),
    ("ibm-granite/granite-3.0-1b-a400m-base", 1.0, "Granite-MoE", "moe"),  # moe skip via arch
    ("ibm-granite/granite-3.1-1b-a400m-base", 1.0, "Granite-MoE", "moe"),
    ("HuggingFaceTB/SmolLM-1.7B", 1.7, "SmolLM", "transformer"),
    ("OpenBMB/MiniCPM4-0.5B", 0.5, "MiniCPM4", "transformer"),
    ("Qwen/Qwen2.5-0.5B", 0.5, "Qwen2.5", "transformer"),
    ("meta-llama/Llama-3.2-1B", 1.0, "Llama3.2", "transformer"),
    ("google/gemma-2-2b", 2.0, "Gemma2", "transformer"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", 2.4, "EXAONE", "transformer"),
    ("kakaocorp/kanana-nano-2.1b-base", 2.1, "Kanana", "transformer"),
    ("yanolja/EEVE-Korean-2.8B-v1.0", 2.8, "EEVE", "transformer"),
    ("openlm-research/open_llama_3b", 3.0, "OpenLLaMA", "transformer"),
    ("stabilityai/stablelm-3b-4e1t", 3.0, "StableLM", "transformer"),
    ("bigscience/bloom-3b", 3.0, "BLOOM", "transformer"),
    ("replit/replit-code-v1_5-3b", 3.0, "Replit", "transformer"),
    ("bigcode/starcoder2-3b", 3.0, "StarCoder2", "transformer"),
    ("Qwen/Qwen2.5-3B", 3.0, "Qwen2.5", "transformer"),
    ("meta-llama/Llama-3.2-3B", 3.0, "Llama3.2", "transformer"),
    ("tiiuae/Falcon3-3B-Base", 3.0, "Falcon3", "transformer"),
    ("microsoft/Phi-3-mini-4k-instruct", 3.8, "Phi3", "transformer"),
    ("nvidia/Llama-3.1-Minitron-4B-Depth-Base", 4.0, "Minitron", "transformer"),
    ("Writer/camel-5b-hf", 5.0, "Camel", "transformer"),
    ("openbmb/cpm-bee-5b", 5.0, "CPM-Bee", "transformer"),
    ("EleutherAI/polyglot-ko-5.8b", 5.8, "Polyglot-KO", "transformer"),
    ("beomi/KoAlpaca-Polyglot-5.8B", 5.8, "KoAlpaca", "transformer"),
    ("nlpai-lab/KULLM-Polyglot-5.8B-v2", 5.8, "KULLM", "transformer"),
    ("facebook/xglm-4.5B", 4.5, "XGLM", "transformer"),
    ("speakleash/Bielik-4.5B-v3.0-Instruct", 4.5, "Bielik", "transformer"),
]

# Families / id patterns that failed under vLLM on the P6 node (engine load,
# ValidationError, hang, or forced hf_fallback). Match against lowercased id.
# Prefer precise patterns so Falcon3 / bloom (non-z) / opt-350m stay eligible.
VLLM_BLOCK_SUBSTR = (
    # hf_fallback markers + hangs
    "mamba",
    "openelm",
    "gemma-3",
    "recurrentgemma",
    # gemma instruct hung on FLASH_ATTN (keep gemma base only)
    "gemma-2-2b-it",
    "gemma-2b-it",
    "gemma-1.1-2b-it",
    "gemma-1.1-7b-it",
    "gemma-7b-it",
    "gemma-2-9b-it",
    # classic falcon (Engine core); Falcon3 kept
    "tiiuae/falcon-",
    # cerebras / mpt (repo-id / loader)
    "cerebras",
    "mosaicml/mpt",
    "mpt-7b",
    # internlm / danube3 (Engine core init failed)
    "internlm",
    "h2o-danube3",
    # ModelConfig ValidationError classes from the 32-model skip list
    # (exact ids also listed in VLLM_BLOCK_EXACT; substrings catch siblings)
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
    # alt arches that need special backends
    "rwkv",
    "zamba",
    "hazyresearch/based",
    "olmoe",
)

# Exact ids from AdaptiveTesting/Test/Inference/backfill_vllm_skips/models_all.txt
VLLM_BLOCK_EXACT = {
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
    "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1",
    "LLM360/Amber", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-2",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "togethercomputer/RedPajama-INCITE-7B-Base",
    "apple/OpenELM-1_1B", "apple/OpenELM-3B",
    "cerebras/Cerebras-GPT-1.3B", "cerebras/Cerebras-GPT-2.7B",
    "cerebras/Cerebras-GPT-6.7B",
    "mosaicml/mpt-7b", "mosaicml/mpt-7b-instruct",
    "h2oai/h2o-danube3-4b-base",
    "internlm/internlm2-1_8b", "internlm/internlm2-7b", "internlm/internlm2_5-7b",
    "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct",
}

# Decoder-only transformers only (alt arches were the vLLM pain points).
ALLOWED_ARCH = {"transformer"}


def in_range(p: float) -> bool:
    return 0.2 <= p <= 7.0


def is_vllm_blocked(mid: str) -> bool:
    if mid in VLLM_BLOCK_EXACT:
        return True
    low = mid.lower()
    if any(s in low for s in VLLM_BLOCK_SUBSTR):
        return True
    # Any Gemma instruct/IT checkpoint hung or forced fallback on P6.
    if "gemma" in low and ("-it" in low or "_it" in low or "/it-" in low):
        return True
    # GPT-NeoX shares the ModelConfig failure class with Pythia.
    if "gpt-neox" in low:
        return True
    return False


def is_excluded(mid: str) -> bool:
    """True if the model should not appear in the roster."""
    return is_vllm_blocked(mid)


def target_bin_counts(n: int = 200) -> dict[str, int]:
    """Near-normal size mix with BOTH tails halved vs the prior roster.

    Prior tails were ~24 (0.2-1) and ~26 (6-7). Halved → 12 and 13; the freed
    ~25 slots go into the 1-4B peak.
    """
    # After dropping vLLM-broken families, unique 2-5B bases are scarce, so the
    # peak leans on the 1-4B shoulder while BOTH tails stay halved.
    bins = {
        "0.2-1.0": 12,   # was 24 → half
        "1.0-2.0": 55,
        "2.0-3.0": 40,   # peak (pool-limited)
        "3.0-4.0": 42,
        "4.0-5.0": 20,
        "5.0-6.0": 18,
        "6.0-7.0": 13,   # was 26 → half
    }
    assert sum(bins.values()) == n
    return bins


def bin_of(p: float) -> str:
    edges = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0001]
    labels = ["0.2-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0", "6.0-7.0"]
    for i in range(len(labels)):
        if edges[i] <= p < edges[i + 1]:
            return labels[i]
    return labels[-1]


def select_200(raw: list[tuple[str, float, str, str]]) -> list[tuple[str, float, str, str]]:
    """Select 200 models with a near-normal size mix and org/family diversity.

    Hard constraint: 6-7B bin never exceeds its target (right tail stays short).
    Scarce 4-6B uniques are taken first; remaining slots fill the 1-4B peak.
    """
    seen: set[str] = set()
    pool: list[tuple[str, float, str, str]] = []
    for mid, p, fam, arch in raw:
        if mid in seen or is_excluded(mid) or not in_range(p) or arch not in ALLOWED_ARCH:
            continue
        seen.add(mid)
        pool.append((mid, p, fam, arch))

    targets = target_bin_counts(200)
    by_bin: dict[str, list[tuple[str, float, str, str]]] = defaultdict(list)
    for item in pool:
        by_bin[bin_of(item[1])].append(item)

    selected: list[tuple[str, float, str, str]] = []
    used_orgs: Counter = Counter()
    used_fams: Counter = Counter()
    selected_ids: set[str] = set()
    max_org, max_fam = 4, 3
    # allenai gets a higher org cap so 1B + 7B OLMo can coexist under diversity limits
    org_cap_overrides = {"allenai": 8}

    def eligible(item: tuple[str, float, str, str]) -> bool:
        mid, _, fam, _ = item
        if mid in selected_ids:
            return False
        org = mid.split("/")[0]
        cap = org_cap_overrides.get(org, max_org)
        if used_orgs[org] >= cap:
            return False
        if used_fams[fam] >= max_fam:
            return False
        return True

    def score(item: tuple[str, float, str, str]) -> tuple:
        mid, p, fam, arch = item
        org = mid.split("/")[0]
        # Prefer OLMo (user asked to add them back), then unused orgs/families.
        olmo_penalty = 0 if "olmo" in mid.lower() else 1
        return (
            olmo_penalty,
            used_orgs[org],
            used_fams[fam],
            abs(p - 3.0),
            mid,
        )

    def force_take(mid: str) -> bool:
        for item in pool:
            if item[0] == mid and mid not in selected_ids:
                selected.append(item)
                selected_ids.add(mid)
                used_orgs[mid.split("/")[0]] += 1
                used_fams[item[2]] += 1
                return True
        return False

    def take_from_bin(b: str, need: int) -> int:
        nonlocal selected
        got = 0
        remaining = list(by_bin[b])
        while got < need:
            cands = [x for x in remaining if eligible(x)]
            if not cands:
                break
            cands.sort(key=score)
            item = cands[0]
            remaining = [x for x in remaining if x[0] != item[0]]
            selected.append(item)
            selected_ids.add(item[0])
            used_orgs[item[0].split("/")[0]] += 1
            used_fams[item[2]] += 1
            got += 1
        return got

    # 0) Seed must-include OLMo (1B + 7B) before org caps crowd them out.
    for mid in (
        "allenai/OLMo-2-0425-1B",
        "allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-1B-hf",
        "allenai/OLMo-2-1124-7B",
        "allenai/OLMo-2-1124-7B-Instruct",
        "allenai/OLMo-7B-0724-hf",
    ):
        force_take(mid)

    # 1) Take ALL scarce mid-size uniques (4-6B), up to target.
    for b in ("4.0-5.0", "5.0-6.0"):
        take_from_bin(b, targets[b])

    # 2) Fill the normal peak (2-4B), then left shoulder, then tiny left tail.
    #    Account for already-seeded OLMo counts in each bin.
    def need(b: str) -> int:
        have = sum(1 for m in selected if bin_of(m[1]) == b)
        return max(0, targets[b] - have)

    for b in ("3.0-4.0", "2.0-3.0", "1.0-2.0", "0.2-1.0"):
        take_from_bin(b, need(b))

    # 3) Right tail — HARD CAP at target (never dump leftover 7B finetunes).
    take_from_bin("6.0-7.0", need("6.0-7.0"))

    # 4) Top up to 200 from the mid peak / shoulder only (NOT the halved tails).
    topup_bins = ["1.0-2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0"]
    for org_cap, fam_cap in ((5, 4), (6, 5), (8, 6), (12, 8), (999, 999)):
        if len(selected) >= 200:
            break
        max_org, max_fam = org_cap, fam_cap
        for b in topup_bins:
            if len(selected) >= 200:
                break
            take_from_bin(b, 200 - len(selected))

    # 5) Last resort: tiny right-tail overflow (still ≈half the old ~26).
    if len(selected) < 200:
        max_org, max_fam = 999, 999
        room = min(2, 200 - len(selected))
        take_from_bin("6.0-7.0", room)

    if len(selected) < 200:
        raise SystemExit(
            f"Only selected {len(selected)} models; mid-size pool exhausted."
        )

    selected = selected[:200]

    def enforce_tail_cap(bin_name: str, hard_max: int) -> None:
        nonlocal selected, max_org, max_fam
        while True:
            excess = [m for m in selected if bin_of(m[1]) == bin_name]
            if len(excess) <= hard_max:
                return
            excess.sort(key=lambda m: (-used_orgs[m[0].split("/")[0]], m[0]))
            victim = excess[0]
            selected.remove(victim)
            selected_ids.discard(victim[0])
            max_org, max_fam = 999, 999
            before = len(selected)
            for b in topup_bins:
                if take_from_bin(b, 1):
                    break
            if len(selected) == before:
                selected.append(victim)
                return

    # Both tails stay halved (allow +2 slack for scarce mid bins).
    enforce_tail_cap("6.0-7.0", targets["6.0-7.0"] + 2)
    enforce_tail_cap("0.2-1.0", targets["0.2-1.0"] + 2)

    selected = selected[:200]
    selected.sort(key=lambda x: (x[1], x[0]))
    return selected


def write_yaml(models: list[tuple[str, float, str, str]]) -> None:
    lines = [
        "# Curated 200 open-source checkpoints in [0.2B, 7.0B].",
        "# Size mix: truncated-normal with BOTH tails halved (0.2-1B=12, 6-7B=13).",
        "# Includes OLMo 1B + 7B family. Excludes vLLM-broken families from the",
        "# P6 sweep (mamba, openelm, gemma-3/it, classic falcon, pythia, opt-1.3+",
        "# gpt-neo, bloomz, cerebras, mpt, internlm, danube3, tinyllama, phi-2,",
        "# redpajama, rwkv, zamba, etc). Falcon3 kept.",
        "#",
        "# Registry (models_registry.py) derives chat/gated/backend flags as usual.",
        "",
        "defaults:",
        "  dtype: bfloat16",
        "  trust_remote_code: true",
        "  max_model_len: 4096",
        "  tp: 1",
        "  apply_chat_template: null",
        "  scoring_method: loglikelihood",
        "",
        "models:",
    ]
    cur_bin = None
    for mid, p, fam, arch in models:
        b = bin_of(p)
        if b != cur_bin:
            lines.append(f"  # --- {b} B  ({fam} / {arch} examples follow) ---")
            cur_bin = b
        lines.append(f"  - {{id: {mid}, params_b: {p}, family: {fam}, arch: {arch}}}")
    with open(OUT_YAML, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def stats_and_plot(models: list[tuple[str, float, str, str]]) -> str:
    params = np.array([m[1] for m in models], float)
    orgs = [m[0].split("/")[0] for m in models]
    fams = [m[2] for m in models]
    archs = [m[3] for m in models]
    bins = Counter(bin_of(p) for p in params)

    lines = []
    lines.append("=" * 64)
    lines.append("MODELS_200 ROSTER STATS")
    lines.append("=" * 64)
    lines.append(f"n_models          : {len(models)}")
    lines.append(f"param range       : {params.min():.3f} – {params.max():.3f} B")
    lines.append(f"mean / median     : {params.mean():.3f} / {np.median(params):.3f} B")
    lines.append(f"std               : {params.std():.3f} B")
    lines.append(f"unique orgs       : {len(set(orgs))}")
    lines.append(f"unique families   : {len(set(fams))}")
    lines.append(f"unique arches     : {len(set(archs))}")
    olmos = [m[0] for m in models if "olmo" in m[0].lower()]
    blocked = [m[0] for m in models if is_vllm_blocked(m[0])]
    lines.append(f"OLMo models        : {len(olmos)}  {olmos}")
    lines.append(f"vLLM-blocked left? : {len(blocked)}  {blocked[:5]}")
    lines.append("")
    lines.append("Param bin counts (target ≈ truncated normal):")
    targets = target_bin_counts()
    for b in targets:
        lines.append(f"  {b:8s}  actual={bins.get(b,0):3d}  target={targets[b]:3d}")
    lines.append("")
    lines.append("Architecture counts:")
    for a, n in Counter(archs).most_common():
        lines.append(f"  {a:20s} {n:3d}")
    lines.append("")
    lines.append("Top orgs by count:")
    for o, n in Counter(orgs).most_common(20):
        lines.append(f"  {o:28s} {n:3d}")
    lines.append("")
    lines.append("Top families by count:")
    for f, n in Counter(fams).most_common(20):
        lines.append(f"  {f:28s} {n:3d}")
    lines.append("")
    lines.append(f"yaml -> {OUT_YAML}")
    lines.append(f"plot -> {OUT_PNG}")
    text = "\n".join(lines)

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, "AdaptiveTesting/Outputs/.mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    edges = [0.2, 1, 2, 3, 4, 5, 6, 7]
    labels = ["0.2-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7"]
    counts = [bins.get(f"{a}-{b}" if a != 0.2 else "0.2-1.0", bins.get(f"{a:.1f}-{b:.1f}", 0))
              for a, b in zip(edges[:-1], edges[1:])]
    # fix labels map
    order = ["0.2-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0", "5.0-6.0", "6.0-7.0"]
    counts = [bins.get(b, 0) for b in order]
    ax.bar(labels, counts, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("parameter count (B)")
    ax.set_ylabel("# models")
    ax.set_title("Size distribution (target: truncated normal)")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    top = Counter(orgs).most_common(15)
    ax2.barh([t[0] for t in reversed(top)], [t[1] for t in reversed(top)],
             color="#ff7f0e", edgecolor="black")
    ax2.set_xlabel("# models")
    ax2.set_title("Top 15 orgs (diversity check)")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"200-model roster  mean={params.mean():.2f}B  "
                 f"median={np.median(params):.2f}B  "
                 f"{len(set(orgs))} orgs / {len(set(fams))} families / {len(set(archs))} arches",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)

    with open(OUT_STATS, "w") as fh:
        fh.write(text + "\n")
    return text


def main():
    models = select_200(RAW)
    bad = [m for m in models if is_vllm_blocked(m[0])]
    if bad:
        raise SystemExit(f"vLLM-blocked models slipped through: {bad}")
    olmos = [m[0] for m in models if "olmo" in m[0].lower()]
    if not olmos:
        raise SystemExit("expected OLMo models to be included")
    if len(models) != 200:
        raise SystemExit(f"expected 200, got {len(models)}")
    if len({m[0] for m in models}) != 200:
        raise SystemExit("duplicate ids")
    # right/left tail hard checks (halved targets)
    from collections import Counter as _C
    bc = _C(bin_of(m[1]) for m in models)
    if bc["0.2-1.0"] > 14 or bc["6.0-7.0"] > 16:
        raise SystemExit(f"tails not halved enough: {dict(bc)}")
    write_yaml(models)
    print(stats_and_plot(models))


if __name__ == "__main__":
    main()
