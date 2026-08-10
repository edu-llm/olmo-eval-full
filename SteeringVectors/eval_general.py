#!/usr/bin/env python3
"""General-capability benchmarks (paper Tables 1–2): ARC, MMLU subset, MathQA, RACE."""

from __future__ import annotations

import random

from run_steering_eval import _load_hf_split


def _choice_loglik(model, tokenizer, context: str, continuation: str, device) -> float:
    import torch

    ctx_ids = tokenizer(context, return_tensors="pt").input_ids
    cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    cont_len = cont_ids.shape[1]
    return token_lp[0, -cont_len:].sum().item()


def _mc_accuracy(
    model,
    tokenizer,
    device: str,
    items: list[tuple[str, list[str], int]],
) -> float:
    hits = 0
    for context, choices, gold in items:
        scores = [_choice_loglik(model, tokenizer, context, f" {c}", device) for c in choices]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        hits += int(pred == gold)
    return hits / len(items) if items else 0.0


def eval_arc_challenge(model, tokenizer, device: str, limit: int = 0) -> dict[str, float]:
    data = _load_hf_split(
        [
            (("ai2_arc", "ARC-Challenge"), {"split": "validation"}),
            (("allenai/ai2_arc", "ARC-Challenge"), {"split": "validation"}),
        ]
    )
    if limit and limit < len(data):
        data = data.select(range(limit))
    items: list[tuple[str, list[str], int]] = []
    for row in data:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {lab: i for i, lab in enumerate(labels)}
        gold = label_map[row["answerKey"]]
        ctx = f"Question: {row['question']}\nAnswer:"
        items.append((ctx, texts, gold))
    return {"arc_challenge": _mc_accuracy(model, tokenizer, device, items)}


def eval_mmlu_subset(
    model, tokenizer, device: str, n_subjects: int = 8, per_subject: int = 50, seed: int = 1234
) -> dict[str, float]:
    data = _load_hf_split([(("cais/mmlu", "all"), {"split": "test"})])
    subjects = sorted(set(data["subject"]))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    subjects = subjects[:n_subjects]
    hits, total = 0, 0
    for subject in subjects:
        rows = data.filter(lambda r, s=subject: r["subject"] == s)
        if per_subject and per_subject < len(rows):
            rows = rows.select(range(per_subject))
        for row in rows:
            ctx = f"Question: {row['question']}\nAnswer:"
            choices = row["choices"]
            gold = int(row["answer"])
            scores = [_choice_loglik(model, tokenizer, ctx, f" {c}", device) for c in choices]
            pred = max(range(len(scores)), key=lambda i: scores[i])
            hits += int(pred == gold)
            total += 1
    return {"mmlu_subset": hits / total if total else 0.0, "mmlu_questions": float(total)}


def eval_mathqa(model, tokenizer, device: str, limit: int = 200) -> dict[str, float]:
    data = _load_hf_split([(("math_qa",), {"split": "validation"})])
    if limit and limit < len(data):
        data = data.select(range(limit))
    items: list[tuple[str, list[str], int]] = []
    for row in data:
        opts = [part.split(")", 1)[-1].strip() for part in row["options"].split(",")]
        letter = row["correct"].strip()[0].lower()
        gold = ord(letter) - ord("a")
        ctx = f"Question: {row['problem']}\nAnswer:"
        items.append((ctx, opts, gold))
    return {"mathqa": _mc_accuracy(model, tokenizer, device, items)}


def eval_race(model, tokenizer, device: str, limit: int = 200) -> dict[str, float]:
    data = _load_hf_split([(("race", "high"), {"split": "validation"})])
    if limit and limit < len(data):
        data = data.select(range(limit))
    items: list[tuple[str, list[str], int]] = []
    for row in data:
        ctx = f"Article: {row['article']}\nQuestion: {row['question']}\nAnswer:"
        choices = row["options"]
        gold = list("ABCD").index(row["answer"].strip().upper())
        items.append((ctx, choices, gold))
    return {"race": _mc_accuracy(model, tokenizer, device, items)}


def eval_all_general(
    model,
    tokenizer,
    device: str,
    *,
    arc_limit: int = 0,
    mmlu_subjects: int = 8,
    mmlu_per_subject: int = 50,
    mathqa_limit: int = 200,
    race_limit: int = 200,
    seed: int = 1234,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(eval_arc_challenge(model, tokenizer, device, arc_limit))
    out.update(eval_mmlu_subset(model, tokenizer, device, mmlu_subjects, mmlu_per_subject, seed))
    out.update(eval_mathqa(model, tokenizer, device, mathqa_limit))
    out.update(eval_race(model, tokenizer, device, race_limit))
    return out
