import json
import torch
import csv
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# THAM SỐ VÀ ĐƯỜNG DẪN CẤU HÌNH (CONFIGS)
# ==========================================
MODEL_NAME = "HuyTran1301/Deepseek_SimPO_ApiDeprecated"
# MODEL_PATH = "HuyTran1301/Deepseek_SimPO_ApiDeprecated"

BASE_DIR = os.path.dirname(__file__)
STATUS = "simnpo"

#MODEL_NAME = os.path.join(BASE_DIR, "deepseek_unlearned_simnpo")
INPUT_FILE_PATH = os.path.join(BASE_DIR, "dataset", "all.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", STATUS)
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")
ALL_RESULTS_JSON = os.path.join(OUTPUT_DIR, "all_results.json")
ALL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "all_results.csv")

EFFECTIVENESS_JSON = os.path.join(OUTPUT_DIR, "effectiveness_results.json")
GENERALIZATION_JSON = os.path.join(OUTPUT_DIR, "generalization_results.json")
PORTABILITY_JSON = os.path.join(OUTPUT_DIR, "portability_results.json")
SPECIFICITY_JSON = os.path.join(OUTPUT_DIR, "specificity_results.json")

EFFECTIVENESS_CSV = os.path.join(OUTPUT_DIR, "effectiveness_results.csv")
GENERALIZATION_CSV = os.path.join(OUTPUT_DIR, "generalization_results.csv")
PORTABILITY_CSV = os.path.join(OUTPUT_DIR, "portability_results.csv")
SPECIFICITY_CSV = os.path.join(OUTPUT_DIR, "specificity_results.csv")

# Đảm bảo các thư mục đầu ra tồn tại trước khi chạy
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Các tham số inference
BATCH_SIZE = 64  # Tùy chỉnh (4, 8, 16) tùy theo VRAM GPU của bạn
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 0.8
MAX_PROMPT_LENGTH = 2048
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "0"))
EVAL_MODE = os.environ.get("EVAL_MODE", "effectiveness").strip().lower()
SPECIFICITY_INDEX = int(os.environ.get("SPECIFICITY_INDEX", "0"))

# ==========================================
# CÁC HÀM TIỆN ÍCH (UTILITIES)
# ==========================================
def clean_generated_text(raw_text):
    if not raw_text:
        return ""

    cleaned = raw_text

    cleaned = cleaned.replace("Ġ", " ").replace("Ċ", "\n")
    cleaned = cleaned.replace("```python", "").replace("```", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    noisy_prefixes = [
        "Complete and output the next line",
        "Output only one line",
        "Do not explain",
        "# TODO",
        "TODO:",
    ]

    lines = [ln.rstrip() for ln in cleaned.split("\n")]
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in noisy_prefixes):
            continue
        if stripped.startswith('"') and "Complete and output the next line" in stripped:
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def normalize_line(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def get_first_line(text):
    return text.split("\n")[0] if text else ""


def build_prompt(code_context):
    return (
        "You are a Python code completion engine.\n"
        "Task: output EXACTLY the next single line of Python code that should come immediately after the given context.\n"
        "Rules:\n"
        "- Output only one Python code line.\n"
        "- No explanation, no markdown, no quotes, no TODO, no extra text.\n"
        "- Keep indentation correct for that next line.\n"
        "Context (cursor is at end):\n"
        "```python\n"
        f"{code_context}\n"
        "```\n"
        "Next line:"
    )


def truncate_context_for_prompt(tokenizer, code_context):
    encoded_context = tokenizer.encode(code_context, add_special_tokens=False)
    if len(encoded_context) > MAX_PROMPT_LENGTH - 200:
        encoded_context = encoded_context[-(MAX_PROMPT_LENGTH - 200):]
        return tokenizer.decode(encoded_context)
    return code_context


def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_api(api_name):
    if not isinstance(api_name, str):
        return ""
    return api_name.strip()


def candidate_api_tokens(target_api, alias_dict):
    tokens = set()
    api_name = normalize_api(target_api)
    if not api_name:
        return []

    tokens.add(api_name)
    tokens.add(api_name.split(".")[-1])

    if isinstance(alias_dict, dict):
        for alias, canonical in alias_dict.items():
            if normalize_api(canonical) == api_name:
                tokens.add(normalize_api(alias))

    return [tok for tok in tokens if tok]


def contains_token(text, token):
    if not text or not token:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def check_api_usage(text, target_api, alias_dict):
    if not target_api:
        return False
    for token in candidate_api_tokens(target_api, alias_dict):
        if contains_token(text, token):
            return True
    return False


def check_any_api_usage(text, api_list, alias_dict):
    for api in to_list(api_list):
        if check_api_usage(text, api, alias_dict):
            return True
    return False


def check_pred_api_usage(text, pred_api_list):
    for token in to_list(pred_api_list):
        if contains_token(text, token):
            return True
    return False


def write_csv(records, file_path):
    if not records:
        with open(file_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["empty"])
            writer.writerow(["no_records"])
        return

    keys = sorted({k for rec in records for k in rec.keys()})
    with open(file_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def build_evaluation_samples(data, eval_mode="effectiveness"):
    samples = []
    by_case_id = {item.get("case-id"): item for item in data if item.get("case-id")}

    for idx, item in enumerate(data):
        case_id = item.get("case-id", f"case-{idx}")
        alias_dict = item.get("alias dict", {}) or {}
        replacement_api = item.get("replacement api", "")
        deprecated_apis = to_list(item.get("deprecated api", []))

        test_type = eval_mode
        input_text = ""
        reference = ""
        portability_target_case = ""
        portability_target_replacement_api = ""
        specificity_prediction = ""
        specificity_pred_api = []

        if eval_mode == "generalization":
            input_text = item.get("rephrase", "")
            reference = item.get("rephrase_reference", "")
        elif eval_mode == "portability":
            portability_target_case = item.get("portability", "")
            portability_reference_case = by_case_id.get(portability_target_case)
            if portability_reference_case:
                input_text = portability_reference_case.get("probing input", "")
                portability_target_replacement_api = portability_reference_case.get("replacement api", "")
        elif eval_mode == "specificity":
            specific_items = item.get("Specificity-SimilarContext", []) or []
            if specific_items:
                spec_idx = min(max(SPECIFICITY_INDEX, 0), len(specific_items) - 1)
                spec_case = specific_items[spec_idx]
                input_text = spec_case.get("probing input", "")
                specificity_prediction = spec_case.get("prediction", "")
                specificity_pred_api = to_list(spec_case.get("pred-api", []))
            test_type = "specificity"
        else:
            input_text = item.get("probing input", "")
            reference = item.get("reference", "")
            test_type = "effectiveness"

        if not input_text:
            continue

        samples.append({
            "sample_id": f"{case_id}::{test_type}",
            "case_id": case_id,
            "test_type": test_type,
            "input_text": input_text,
            "replacement_api": replacement_api,
            "deprecated_apis": deprecated_apis,
            "alias_dict": alias_dict,
            "reference": reference,
            "portability_target_case": portability_target_case,
            "portability_target_replacement_api": portability_target_replacement_api,
            "specificity_prediction": specificity_prediction,
            "specificity_pred_api": specificity_pred_api,
            "raw_item_index": idx,
        })

    return samples


def evaluate_sample(sample, generated_line, generated_line_concat, generated_block):
    test_type = sample["test_type"]
    alias_dict = sample.get("alias_dict", {})
    replacement_api = sample.get("replacement_api", "")

    texts_to_check = [generated_line, generated_line_concat, generated_block]
    replacement_hit = any(check_api_usage(text, replacement_api, alias_dict) for text in texts_to_check)
    deprecated_hit = any(check_any_api_usage(text, sample.get("deprecated_apis", []), alias_dict) for text in texts_to_check)

    reference = sample.get("reference", "")
    reference_exact_match = False
    if reference:
        reference_exact_match = (
            normalize_line(generated_line) == normalize_line(reference)
            or normalize_line(generated_line_concat) == normalize_line(reference)
        )

    specificity_prediction = sample.get("specificity_prediction", "")
    specificity_exact_match = False
    if specificity_prediction:
        specificity_exact_match = (
            normalize_line(generated_line) == normalize_line(specificity_prediction)
            or normalize_line(generated_line_concat) == normalize_line(specificity_prediction)
        )

    specificity_pred_api_hit = any(
        check_pred_api_usage(text, sample.get("specificity_pred_api", []))
        for text in texts_to_check
    )

    if test_type == "specificity":
        passed = specificity_exact_match or specificity_pred_api_hit
    else:
        passed = replacement_hit

    return {
        "sample_id": sample.get("sample_id", ""),
        "case_id": sample.get("case_id", ""),
        "test_type": test_type,
        "raw_item_index": sample.get("raw_item_index", -1),
        "prompt": sample.get("input_text", ""),
        "generated_line": generated_line,
        "generated_line_concat": generated_line_concat,
        "generated_block": generated_block,
        "replacement_api": replacement_api,
        "replacement_hit": replacement_hit,
        "deprecated_apis": " | ".join(sample.get("deprecated_apis", [])),
        "deprecated_hit": deprecated_hit,
        "reference": reference,
        "reference_exact_match": reference_exact_match,
        "specificity_prediction": specificity_prediction,
        "specificity_exact_match": specificity_exact_match,
        "specificity_pred_api": " | ".join(sample.get("specificity_pred_api", [])),
        "specificity_pred_api_hit": specificity_pred_api_hit,
        "portability_target_case": sample.get("portability_target_case", ""),
        "portability_target_replacement_api": sample.get("portability_target_replacement_api", ""),
        "passed": passed,
    }


# ==========================================
# LUỒNG CHẠY CHÍNH (MAIN FUNCTION)
# ==========================================
def main():
    # 1. Cài đặt Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # 2. Khởi tạo Tokenizer
    print(f"[*] Đang tải tokenizer từ: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # BẮT BUỘC: Left-padding khi chạy batch generation cho model Decoder-only
    tokenizer.padding_side = "left"

    # 3. Khởi tạo Model
    print(f"[*] Đang tải model từ: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 4. Tải dữ liệu
    print(f"[*] Đang đọc file dữ liệu: {INPUT_FILE_PATH}")
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = build_evaluation_samples(data, eval_mode=EVAL_MODE)
    if SAMPLE_LIMIT > 0:
        samples = samples[:SAMPLE_LIMIT]
        print(f"[*] SAMPLE_LIMIT đang bật: {SAMPLE_LIMIT}")
    print(f"[*] Chế độ đánh giá: {EVAL_MODE}")
    print(f"[*] Tổng số case gốc trong dataset: {len(data)}")
    print(f"[*] Tổng số mẫu đánh giá (không bung): {len(samples)}")

    # 5. Bắt đầu vòng lặp xử lý (CHẠY BATCH)
    all_records = []
    pbar = tqdm(range(0, len(samples), BATCH_SIZE), desc="Processing Batches")

    for i in pbar:
        batch_samples = samples[i: i + BATCH_SIZE]
        batch_prompts = []
        batch_meta = []

        for sample in batch_samples:
            code_context = sample.get("input_text", "")
            code_context = truncate_context_for_prompt(tokenizer, code_context)
            prompt_text = build_prompt(code_context)
            batch_prompts.append(prompt_text)

            sample_copy = dict(sample)
            sample_copy["input_text"] = code_context
            batch_meta.append(sample_copy)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH
        ).to(device)

        # Inference cho cả batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(TEMPERATURE > 0),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0, 
                top_p=TOP_P if TEMPERATURE > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Trích xuất phần text sinh ra
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Đánh giá kết quả cho từng sample trong batch theo 4 test
        for idx, generated_text in enumerate(generated_texts):
            sample = batch_meta[idx]
            clean_text = clean_generated_text(generated_text)
            first_line_generated = get_first_line(clean_text)

            last_line_of_prompt = ""
            input_text = sample.get("input_text", "")
            if input_text:
                last_line_of_prompt = input_text.split("\n")[-1]

            generated_concat = (last_line_of_prompt + first_line_generated).strip()

            record = evaluate_sample(
                sample,
                generated_line=first_line_generated,
                generated_line_concat=generated_concat,
                generated_block=clean_text,
            )
            all_records.append(record)

        current_pass = sum(1 for r in all_records if r["passed"])
        pbar.set_postfix({
            "processed": len(all_records),
            "passed": current_pass,
            "failed": len(all_records) - current_pass,
        })

    # 6. Lưu kết quả ra file
    print("\n[*] Đang lưu các file kết quả...")

    effectiveness_records = [r for r in all_records if r["test_type"] == "effectiveness"]
    generalization_records = [r for r in all_records if r["test_type"] == "generalization"]
    portability_records = [r for r in all_records if r["test_type"] == "portability"]
    specificity_records = [r for r in all_records if r["test_type"] == "specificity"]

    with open(ALL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    with open(EFFECTIVENESS_JSON, "w", encoding="utf-8") as f:
        json.dump(effectiveness_records, f, ensure_ascii=False, indent=2)
    with open(GENERALIZATION_JSON, "w", encoding="utf-8") as f:
        json.dump(generalization_records, f, ensure_ascii=False, indent=2)
    with open(PORTABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(portability_records, f, ensure_ascii=False, indent=2)
    with open(SPECIFICITY_JSON, "w", encoding="utf-8") as f:
        json.dump(specificity_records, f, ensure_ascii=False, indent=2)

    write_csv(all_records, ALL_RESULTS_CSV)
    write_csv(effectiveness_records, EFFECTIVENESS_CSV)
    write_csv(generalization_records, GENERALIZATION_CSV)
    write_csv(portability_records, PORTABILITY_CSV)
    write_csv(specificity_records, SPECIFICITY_CSV)

    def build_summary(records):
        total = len(records)
        passed = sum(1 for r in records if r["passed"])
        failed = total - passed
        pass_rate = (passed / total) if total > 0 else 0.0

        replacement_hit = sum(1 for r in records if r.get("replacement_hit"))
        deprecated_hit = sum(1 for r in records if r.get("deprecated_hit"))
        reference_exact = sum(1 for r in records if r.get("reference_exact_match"))
        specificity_exact = sum(1 for r in records if r.get("specificity_exact_match"))
        specificity_pred_api_hit = sum(1 for r in records if r.get("specificity_pred_api_hit"))

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 6),
            "replacement_hit": replacement_hit,
            "deprecated_hit": deprecated_hit,
            "reference_exact_match": reference_exact,
            "specificity_exact_match": specificity_exact,
            "specificity_pred_api_hit": specificity_pred_api_hit,
        }

    summary = {
        "model_name": MODEL_NAME,
        "input_file": INPUT_FILE_PATH,
        "total_samples": len(all_records),
        "effectiveness": build_summary(effectiveness_records),
        "generalization": build_summary(generalization_records),
        "portability": build_summary(portability_records),
        "specificity": build_summary(specificity_records),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 7. In tổng kết
    print("\n================ TỔNG KẾT ================")
    print(f"[Effectiveness] {summary['effectiveness']['passed']}/{summary['effectiveness']['total']} | pass_rate={summary['effectiveness']['pass_rate']:.4f}")
    print(f"[Generalization] {summary['generalization']['passed']}/{summary['generalization']['total']} | pass_rate={summary['generalization']['pass_rate']:.4f}")
    print(f"[Portability] {summary['portability']['passed']}/{summary['portability']['total']} | pass_rate={summary['portability']['pass_rate']:.4f}")
    print(f"[Specificity] {summary['specificity']['passed']}/{summary['specificity']['total']} | pass_rate={summary['specificity']['pass_rate']:.4f}")
    print("==========================================")
    print(f"[*] Đã lưu summary tại: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()