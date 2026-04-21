import os
from pyexpat import model
import sys
import json
import torch
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi 

# Setup paths to import local eval_metrics module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# These imports assume compute_edit_quality is updated in your local file
from eval_metrics.edapi_evaluate import compute_edit_quality
from eval_metrics.evaluate_utils import MATCH_METRICS

# ==========================================
# RAG CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "Knowledge_base.json"

# SAHR Hyperparameters
# We use 50 candidates for Stage 1 (Recall) and 3 for the final prompt (Precision)
STAGE_1_TOP_K = 50
STAGE_2_TOP_K = 3

# ==========================================
# SAHR UTILITY FUNCTIONS
# ==========================================
def smart_tokenize(text):
    """Enhanced tokenization for library alignment (pd -> pandas, etc)."""
    if not text: return []
    text = text.replace("pd.", "pandas.").replace("np.", "numpy.").replace("tf.", "tensorflow.")
    # Split CamelCase (e.g., DataFrame -> Data Frame)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Tokenize on symbols, dots, and underscores
    text = re.sub(r'[\._\(\)\[\]=/"\':]', ' ', text)
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    # Noise Reduction for boilerplate
    noise = {'def', 'self', 'class', 'return', 'if', 'else', 'for', 'in', 'while', 'assert', 'import'}
    return [t for t in tokens if t not in noise]

def prepare_requests(data):
    """Maps raw dataset keys to the format required by the evaluation engine."""
    requests = []
    for d in data:
        req = {
            'case_id': d['case-id'],
            'prompt': d['probing input'],
            'target_new': d['reference'],
            'rephrase_prompt': d['rephrase'],
            'rephrase_target_new': d['rephrase_reference'],
            'reference_dict': d['reference dict'],
            'alias_dict': d['alias dict'],
            # Merged reference dicts for rephrase consistency
            'rephrase_reference_dict': {**d['reference dict'], **d.get('rephrase_reference_dict', {})},
            'new_api': [[d['replacement api']]],
            'specificity': {
                'prompts': [item['probing input'] for item in d.get('Specificity-SimilarContext', [])],
                'ground_truth': [item['prediction'] for item in d.get('Specificity-SimilarContext', [])],
                'pred-api': [item['pred-api'] for item in d.get('Specificity-SimilarContext', [])],
            },
            'portability': d.get('portability', ""),
            'target_api': d['replacement api'],
            'deprecated_api': d['deprecated api'], # Critical for RAG accuracy
        }
        requests.append(req)
    return requests

def main():
    # Model Loading (Priority: Environment Variable -> Local Path -> HF Hub)
    model_name = os.environ.get("MODEL_NAME", "/workspace/NCKH/prod_model")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "all.json")
    
    # 1. INITIALIZE RAG INDEX (SAHR Style with Signal Boosting)
    if not KB_PATH.exists():
        print(f"Error: Knowledge Base not found at {KB_PATH}")
        return

    print(f"[*] Initializing SAHR Retrieval Index from: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    tokenized_corpus = []
    for entry in kb:
        full_path = entry.get('deprecated_api', '').lower()
        # Pre-cache parts for multipliers
        entry['method_name'] = full_path.split('.')[-1]
        entry['lib_name'] = full_path.split('.')[0]
        
        # --- SIGNAL BOOSTING ---
        # We repeat the method name 20x to ensure the search focuses on the 
        # specific API call rather than the surrounding documentation text.
        boosted_signal = (
            f"{entry['deprecated_api']} " * 5 + 
            f"{entry['method_name']} " * 20 + 
            entry.get('description', '')
        )
        tokenized_corpus.append(smart_tokenize(boosted_signal))
    
    bm25 = BM25Okapi(tokenized_corpus)
    print("✓ RAG Index Ready.")

    # 2. LOAD MODEL (Unlearned Config)
    print(f"[*] Loading Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. PREPARE DATA
    num_test_samples = 20 # Set to None for full run
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if num_test_samples:
        data = data[:num_test_samples]
    
    requests = prepare_requests(data)
    case_lookup = {r['case_id']: r for r in requests}
    
    # 4. EVALUATION LOOP
    all_metrics = []
    print(f"[*] Starting Evaluation on {len(requests)} samples...")
    
    for idx, request in tqdm(enumerate(requests), total=len(requests)):
        if not request["case_id"]: continue
            
        request = request.copy()
        # Resolve portability references
        if request["portability"] != "":
            port_id = request["portability"]
            request["portability"] = case_lookup.get(port_id, "")
                
        start_time = time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        try:
            # INTEGRATION: compute_edit_quality handles the SAHR multi-stage reranking
            metric_result = compute_edit_quality(
                model=model, 
                tok=tokenizer, 
                record=request, 
                bm25=bm25, 
                kb=kb
            )
            
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
            
            all_metrics.append({
                'case_id': request['case_id'],
                'target_api': request['target_api'],
                'time': round(time() - start_time, 3),
                'max_memory': round(mem_mb, 2),
                'post': metric_result,
            })
        except Exception as e:
            print(f"\n[!] Error in case {request['case_id']}: {e}")
            continue
        
    # 5. AGGREGATE & SAVE
    print("\n[*] Computing Final Mean Metrics...")
    mean_metrics = {}
    for m_type in ['efficacy', 'generalization', 'portability', 'specificity']:
        mean_metrics[m_type] = {}
        for m_name in MATCH_METRICS:
            vals = [item['post'][m_type][m_name] for item in all_metrics 
                    if m_type in item['post'] and m_name in item['post'][m_type]]
            
            if vals:
                # Format: (Mean %, StdDev %)
                mean_metrics[m_type][m_name] = (
                    round(float(np.mean(vals)) * 100, 2), 
                    round(float(np.std(vals)) * 100, 2)
                )
            else:
                mean_metrics[m_type][m_name] = (0.0, 0.0)
    
    # Save Outputs
    res_path = os.path.join(script_dir, "results_unlearned_rag.json")
    mean_path = os.path.join(script_dir, "mean_results_unlearned_rag.json")
    
    with open(res_path, "w", encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
    with open(mean_path, "w", encoding='utf-8') as f:
        json.dump(mean_metrics, f, ensure_ascii=False, indent=4)
        
    print(f"\n✓ Process Complete.")
    print(f"Individual: {res_path}")
    print(f"Aggregated: {mean_path}")

if __name__ == "__main__":
    main()