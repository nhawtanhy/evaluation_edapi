import os
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

from eval_metrics.edapi_evaluate import compute_edit_quality
from eval_metrics.evaluate_utils import MATCH_METRICS

# ==========================================
# RAG CONFIGURATION (Matches Module)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "Knowledge_base.json"
print("Exists:", KB_PATH.exists())
STAGE_1_TOP_K = 50
STAGE_2_TOP_K = 3

# ==========================================
# SAHR UTILITY FUNCTIONS (For Indexer)
# ==========================================
def smart_tokenize(text):
    if not text: return []
    text = text.replace("pd.", "pandas.").replace("np.", "numpy.").replace("tf.", "tensorflow.")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[\._\(\)\[\]=/"\':]', ' ', text)
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    noise = {'def', 'self', 'class', 'return', 'if', 'else', 'for', 'in', 'while', 'assert', 'import'}
    return [t for t in tokens if t not in noise]

def prepare_requests(data):
    """Convert raw dataset format to request format expected by evaluate metrics."""
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
            'rephrase_reference_dict': {**d['reference dict'], **d['rephrase_reference_dict']},
            'new_api': [[d['replacement api']]],
            'specificity': {
                'prompts': [item['probing input'] for item in d['Specificity-SimilarContext']],
                'ground_truth': [item['prediction'] for item in d['Specificity-SimilarContext']],
                'pred-api': [item['pred-api'] for item in d['Specificity-SimilarContext']],
            },
            'portability': d['portability'],
            'target_api': d['replacement api'],
            'deprecated_api': d['deprecated api'],
        }
        requests.append(req)
    return requests

def main():
    # UPDATED: Path to your unlearned model
    model_name = os.environ.get(
        "MODEL_NAME", 
        "HuyTran1301/Deepseek_PROD_ApiDeprecated"
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "all.json")
    
    # 1. INITIALIZE RAG INDEX (SAHR Style)
    print(f"[*] Initializing SAHR Retrieval Index...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    tokenized_corpus = []
    for entry in kb:
        full_path = entry.get('deprecated_api', '').lower()
        entry['method_name'] = full_path.split('.')[-1]
        entry['lib_name'] = full_path.split('.')[0]
        
        # Signal Boosting: Repeat method name 20x
        boosted_signal = (f"{entry['deprecated_api']} " * 5 + 
                          f"{entry['method_name']} " * 20 + 
                          entry.get('description', ''))
        tokenized_corpus.append(smart_tokenize(boosted_signal))
    
    bm25 = BM25Okapi(tokenized_corpus)
    print("✓ RAG Index Ready.")

    # 2. LOAD UNLEARNED MODEL
    print(f"Loading Model from: {model_name}")

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
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    # 3. PREPARE DATA
    num_test_samples = 20 # Set to None to run all
    with open(dataset_path, "r") as f:
        data = json.load(f)
    if num_test_samples:
        data = data[:num_test_samples]
    requests = prepare_requests(data)
    case_lookup = {r['case_id']: r for r in requests}
    
    # 4. EVALUATION LOOP
    all_metrics = []
    print(f"Starting evaluation on {len(requests)} samples...")
    
    for idx, request in tqdm(enumerate(requests), total=len(requests)):
        if request["case_id"] == '': continue
            
        request = request.copy()
        if request["portability"] != "":
            port_id = request["portability"]
            request["portability"] = case_lookup.get(port_id, "")
                
        start = time()
        device = model.device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        try:
            # Pass RAG objects to the imported quality function
            metric_result = compute_edit_quality(
                model, tokenizer, request, 
                bm25=bm25, 
                kb=kb
            )
            
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
            
            all_metrics.append({
                'case_id': request['case_id'],
                'target_api': request['target_api'],
                'time': round(time() - start, 3),
                'max_memory': round(mem_mb, 2),
                'post': metric_result,
            })
        except Exception as e:
            print(f"Error evaluating case {request['case_id']}: {e}")
            continue
        
    # 5. AGGREGATE & SAVE
    print("\nComputing mean metrics...")
    mean_metrics = {}
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        mean_metrics[metric_name] = {}
        for match_metric in MATCH_METRICS:
            vals = [item['post'][metric_name][match_metric] for item in all_metrics 
                    if metric_name in item.get('post', {}) and match_metric in item['post'][metric_name]]
            
            if vals:
                mean_metrics[metric_name][match_metric] = (
                    round(float(np.mean(vals)) * 100, 2), 
                    round(float(np.std(vals)) * 100, 2)
                )
            else:
                mean_metrics[metric_name][match_metric] = (0, 0)
    
    # UPDATED: Output paths for the unlearned model run
    results_path = os.path.join(script_dir, "results_unlearned_rag.json")
    mean_results_path = os.path.join(script_dir, "mean_results_unlearned_rag.json")
    
    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
    with open(mean_results_path, "w", encoding='utf-8') as f:
        json.dump(mean_metrics, f, ensure_ascii=False, indent=4)
        
    print(f"\nDone! Unlearned results saved to:")
    print(f"- {results_path}")
    print(f"- {mean_results_path}")

if __name__ == "__main__":
    main()