import torch
import numpy as np
import scipy
import nltk
import typing
import re
import os

import evaluate as _hf_evaluate
evaluate_load = _hf_evaluate.load

from .bleu.bleu import Bleu

MATCH_METRICS = ['api_exact_match', 'exact_match', 'bleu', 'rougeL']
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# UTILS
# =========================
def format_score(x):
    return round(float(x), 3)


# =========================
# METRICS
# =========================
class Metric:

    @staticmethod
    def exact_match(gens, refs, is_specificity=False, is_api=False):
        if not is_specificity:
            return format_score(
                float(np.prod([1 if sorted(g) == sorted(r) else 0 for g, r in zip(gens, refs)]))
            )

        if len(gens) == 0:
            return 0.0

        score = 0
        n = len(gens)

        for i in range(n):
            g = gens[i]
            r = refs[i]

            if is_api:
                if set(g) == set(r) and len(g) > 0:
                    score += 1
            else:
                if str(g).strip() != "" and str(g).strip() == str(r).strip():
                    score += 1

        return format_score(score / max(n, 1))


    @staticmethod
    def bleu_score(predictions, references):
        metric = Bleu()
        score = metric.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )["bleu"]
        return format_score(score)


    @staticmethod
    def rouge_score(predictions, references):
        rouge_path = os.path.join(_THIS_DIR, "rouge")
        metric = evaluate_load(rouge_path)

        score = metric.compute(
            predictions=predictions,
            references=references
        )["rougeL"]

        return format_score(score)


    @staticmethod
    def api_exact_match_for_efficacy(gens, refs):
        """
        FIXED: robust + safe version
        """
        if not gens or not refs:
            return 0.0

        gen_set = set(gens[0]) if isinstance(gens[0], (list, set)) else set(gens)
        ref_set = set(refs[0][0]) if isinstance(refs[0], list) else set(refs)

        return format_score(float(gen_set == ref_set))


# =========================
# GENERATION
# =========================
def batch_generate(model, tok, prompts, max_length, sample_generate=False):

    if isinstance(prompts, str):
        prompts = [prompts]

    messages = [
        [{"role": "user", "content": p}]
        for p in prompts
    ]

    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    gen_args = {
        "input_ids": inputs,
        "max_new_tokens": max_length,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }

    with torch.no_grad():
        output = model.generate(**gen_args)

    return tok.batch_decode(output, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(**gen_args)

    return tok.batch_decode(
        outputs[:, prompt_tok["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


# =========================
# TEST GENERATION
# =========================
def test_generation_quality(model, tok, prefixes, max_out_len, tokenizer_for_fluency=None):
    texts = batch_generate(model, tok, prefixes, max_out_len, True)

    return {
        "ngram_entropy": n_gram_entropy(texts, tokenizer_for_fluency),
        "generated_texts": texts
    }


# =========================
# ENTROPY
# =========================
def n_gram_entropy(texts, agg="arith", tokenizer_for_fluency=None):
    values = [compute_n_gram_entropy(t, tokenizer_for_fluency=tokenizer_for_fluency) for t in texts]
    return float(np.mean(values))


def compute_n_gram_entropy(sentence, ns=None, weights=None, tokenizer_for_fluency=None):
    ns = ns or [2, 3]
    weights = weights or [2/3, 4/3]

    entropies = []

    for n in ns:
        fdist = compute_freq(sentence, n, tokenizer_for_fluency)
        freqs = np.array(list(fdist.values()))

        if len(freqs) == 0:
            continue

        freqs = freqs / freqs.sum()
        entropies.append(np.sum(-freqs * np.log2(freqs)))

    return float(np.mean(entropies)) if entropies else 0.0


def compute_freq(sentence, n=2, tokenizer_for_fluency=None):
    tokens = tokenizer_for_fluency.encode(sentence) if tokenizer_for_fluency else nltk.word_tokenize(sentence)
    return nltk.FreqDist(nltk.ngrams(tokens, n))


# =========================
# CLEANING
# =========================
def clean_pred(pred):
    return "\n".join([l for l in pred.split("\n") if not l.strip().startswith("#")])


def extract_first_statement(pred, remove_space=True):
    lines = pred.split("\n")

    def norm(x):
        x = x.split("#")[0]
        return re.sub(r"\s+", " ", x.strip())

    stmt = norm(lines[0]) if lines else ""

    return stmt


def extract_first_func(code):
    lines = code.split("\n")
    while lines and not lines[0].lstrip().startswith("def "):
        lines.pop(0)
    return "\n".join(lines) if lines else code


# =========================
# API EXTRACTION
# =========================
def extract_apis_in_first_stmt(pred, ref_dict, alias_dict):
    stmt = extract_first_statement(pred, False)

    apis = set()

    for m in re.finditer(r"([\w\.]+)\s*\(", stmt):
        api = m.group(1)

        if len(api.split(".")) == 2 and api.split(".")[0] in ref_dict:
            api = f"{ref_dict[api.split('.')[0]]}.{api.split('.')[1]}"

        if api in alias_dict:
            api = alias_dict[api]

        apis.add(api)

    return list(apis)