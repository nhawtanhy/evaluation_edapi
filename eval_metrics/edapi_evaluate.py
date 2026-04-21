"""
EDAPI evaluation: compute edit quality metrics (SAHR RAG version).
Clean rewrite based on friend's implementation.
"""

import typing
import re
import numpy as np
import torch
import logging

from .evaluate_utils import (
    batch_generate,
    Metric,
    test_generation_quality,
    MATCH_METRICS,
    clean_pred,
    extract_first_statement,
    extract_first_func,
    extract_apis_in_first_stmt,
)

LOG = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
STAGE_1_TOP_K = 50
STAGE_2_TOP_K = 3


# =========================
# SAHR UTILITIES
# =========================
def smart_tokenize(text):
    if not text:
        return []

    text = (
        text.replace("pd.", "pandas.")
            .replace("np.", "numpy.")
            .replace("tf.", "tensorflow.")
    )

    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[\._\(\)\[\]=/"\':]', ' ', text)

    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())

    noise = {
        'def', 'self', 'class', 'return', 'if', 'else', 'for', 'in',
        'while', 'assert', 'import'
    }

    return [t for t in tokens if t not in noise]


def detect_libs(code_lower):
    signatures = {
        "pandas": ["pd.", "dataframe", "series", "read_csv", "iloc", "loc"],
        "numpy": ["np.", "array", "linspace", "zeros", "matmul"],
        "torch": ["torch.", "tensor", "nn.module", "cuda", "device", "grad"],
        "tensorflow": ["tf.", "keras", "placeholder", "session", "layer"],
    }

    libs = set()
    for lib, sigs in signatures.items():
        if any(sig in code_lower for sig in sigs):
            libs.add(lib)
    return libs


def extract_all_identifiers(code):
    code_lower = code.lower()

    words = set(re.findall(r'\b\w+\b', code_lower))
    sub_words = set()

    for w in words:
        if '_' in w:
            sub_words.update(w.split('_'))

    calls = set(re.findall(r'\.(\w+)\s*\(', code_lower))

    return words | sub_words | calls


def build_augmented_prompt(probing_input, retrieved_docs):
    if not retrieved_docs:
        return probing_input

    blocks = []

    for i, doc in enumerate(retrieved_docs[:3], 1):
        example = doc.get("usage_examples", [""])[0] if doc.get("usage_examples") else ""

        tag = " (MOST RELEVANT)" if i == 1 else ""

        blocks.append(f"""
[{i}]{tag}
Deprecated: {doc.get('deprecated_api', '')}
Replacement: {doc.get('replacement_api', '')}
Description: {doc.get('description', '')}
Example:
{example}
""".strip())

    context = "\n\n".join(blocks)

    return f"""
You are an expert Python developer.

Task: Replace deprecated APIs with modern equivalents.

Reference:
{context}

Rules:
- Do NOT use deprecated APIs
- Output ONLY valid Python code
- No explanations

Now complete:

{probing_input.rstrip()}
""".strip()


# =========================
# MAIN EVAL FUNCTION
# =========================
def compute_edit_quality(
    model,
    tok,
    record: typing.Dict,
    bm25,
    kb,
    test_generation: bool = False,
):
    intent = record['prompt']
    rewritten_intent = record['rephrase_prompt']
    rewritten_target = record['rephrase_target_new']
    target_snippet = record['target_new']

    neighborhoods = record['specificity']
    reference_dict = record['reference_dict']
    alias_dict = record['alias_dict']
    rephrase_reference_dict = record['rephrase_reference_dict']

    new_api = record['new_api']
    portability = record.get("portability", "")

    if 'replace_prompt' in record:
        intent = record['replace_prompt']
        rewritten_intent = record['replace_rephrase_prompt']

    # =========================
    # RAG FUNCTION
    # =========================
    def get_rag_prompt(text):
        if not text:
            return "", [], False, 0.0

        tokens = smart_tokenize(text)
        candidates = bm25.get_top_n(tokens, kb, n=STAGE_1_TOP_K)

        text_lower = text.lower()
        feats = extract_all_identifiers(text)
        libs = detect_libs(text_lower)

        scored = []

        for doc in candidates:
            method = doc.get('method_name', '')
            lib = doc.get('lib_name', '')

            score = 1.0

            if lib in libs:
                score *= 5.0
            elif libs:
                score *= 0.1

            if method in feats:
                score *= 10.0
                if f".{method}" in text_lower:
                    score *= 2.0
            else:
                score *= 0.2

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in scored[:STAGE_2_TOP_K]]

        retrieved = [d.get("replacement_api", "").lower() for d in top_docs]

        expected = ""
        if new_api and isinstance(new_api[0], list):
            expected = new_api[0][0].lower()

        is_hit = any(expected in r or r in expected for r in retrieved)
        mrr = 1.0 / (retrieved.index(expected) + 1) if is_hit else 0.0

        prompt = build_augmented_prompt(text, top_docs)

        return prompt, retrieved, is_hit, mrr

    # =========================
    # PROMPT BUILDING
    # =========================
    rag_intent, _, hit, mrr = get_rag_prompt(intent)
    rag_rephrase, _, _, _ = get_rag_prompt(rewritten_intent)

    if mrr == 0:
        rag_intent = intent
        rag_rephrase = rewritten_intent

    prompts = [rag_intent, rag_rephrase]

    port_prompt = None
    if portability:
        raw = portability.get('replace_prompt', portability.get('prompt', ''))
        port_prompt, _, _, _ = get_rag_prompt(raw)
        prompts.append(port_prompt)

    # =========================
    # GENERATION
    # =========================
    gen_raw = batch_generate(model, tok, prompts, max_length=256)
    preds = [clean_pred(p) for p in gen_raw]

    # safety
    while len(preds) < 3:
        preds.append("")

    if 'replace_prompt' in record:
        preds[0] = extract_first_func(record["replace_prompt"] + preds[0])[len(record["prompt"]):]
        preds[1] = extract_first_func(record["replace_rephrase_prompt"] + preds[1])[len(record["rephrase_prompt"]):]

    gen_strs = [extract_first_statement(p, False) for p in preds]

    gen_api_0 = extract_apis_in_first_stmt(preds[0], reference_dict, alias_dict)
    gen_api_1 = extract_apis_in_first_stmt(preds[1], rephrase_reference_dict, alias_dict)

    gen_api_2 = []
    if portability:
        gen_api_2 = extract_apis_in_first_stmt(
            preds[2],
            portability.get("reference_dict", {}),
            portability.get("alias_dict", {})
        )

    # =========================
    # METRICS
    # =========================
    ret = {
        "efficacy": {},
        "generalization": {},
        "portability": {},
        "specificity": {},
        "retrieval_accuracy": float(hit),
        "mrr": mrr,
        "debug_info": {
            "prediction": gen_strs[0] if gen_strs else "",
            "target": target_snippet,
        }
    }

    metrics_funcs = [
        Metric.api_exact_match_for_efficacy,
        Metric.exact_match,
        Metric.bleu_score,
        Metric.rouge_score,
    ]

    for i, func in enumerate(metrics_funcs):
        name = MATCH_METRICS[i]

        # efficacy
        ret["efficacy"][name] = (
            func(gen_api_0, new_api)
            if i == 0 else func([gen_strs[0]], [target_snippet])
        )

        # generalization
        ret["generalization"][name] = (
            func(gen_api_1, new_api)
            if i == 0 else func([gen_strs[1]], [rewritten_target])
        )

        # portability
        if not portability or len(gen_strs) < 3:
            ret["portability"][name] = 0
        else:
            ret["portability"][name] = (
                func(gen_api_2, new_api)
                if i == 0 else func([gen_strs[2]], [portability.get("target_new", "")])
            )

    # =========================
    # SPECIFICITY
    # =========================
    spec_preds = []
    for p in neighborhoods['prompts']:
        spec_preds += batch_generate(model, tok, p, max_length=50)

    spec_clean = [extract_first_statement(clean_pred(p), False) for p in spec_preds]
    spec_apis = [extract_apis_in_first_stmt(p, {}, alias_dict) for p in spec_clean]

    ret["specificity"] = {
        MATCH_METRICS[0]: np.mean(Metric.exact_match(spec_apis, neighborhoods['pred-api'], True, True)),
        MATCH_METRICS[1]: np.mean(Metric.exact_match(spec_clean, neighborhoods['ground_truth'], True)),
        MATCH_METRICS[2]: Metric.bleu_score(spec_clean, neighborhoods['ground_truth']),
        MATCH_METRICS[3]: Metric.rouge_score(spec_clean, neighborhoods['ground_truth']),
    }

    # =========================
    # OPTIONAL TEST GEN
    # =========================
    if test_generation:
        try:
            ret.update(
                test_generation_quality(
                    model, tok,
                    [intent, rewritten_intent],
                    max_out_len=50
                )
            )
        except Exception:
            pass

    return ret