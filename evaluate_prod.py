import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths to import local eval_metrics module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from eval_metrics.edapi_evaluate import compute_edit_quality
from eval_metrics.evaluate_utils import MATCH_METRICS

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
            'probing_predictions': d['probing predictions'][0][0],
            'api_predicted': d['probing predictions'][0][1],
            'deprecated_api': d['deprecated api'],
            'expected_call': d['expected call'],
        }
        requests.append(req)
    return requests

def main():
    model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "all.json")
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Số lượng data test. Đổi thành None nếu muốn chạy toàn bộ dataset.
    num_test_samples = 10

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    if num_test_samples is not None:
        data = data[:num_test_samples]
        print(f"-> Using first {num_test_samples} samples for quick testing.")
        
    requests = prepare_requests(data)
    print(f"Total requests: {len(requests)}")
    
    # Resolve portability cases
    case_lookup = {r['case_id']: r for r in requests}
    
    all_metrics = []
    
    print("Starting evaluation...")
    for idx, request in tqdm(enumerate(requests), total=len(requests)):
        if request["case_id"] == '':
            continue
            
        request = request.copy()
        if request["portability"] != "":
            port_id = request["portability"]
            if port_id in case_lookup:
                request["portability"] = case_lookup[port_id]
            else:
                request["portability"] = ""
                
        start = time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        # Test generation without any edit
        try:
            metric_result = compute_edit_quality(
                model, tokenizer, request, test_generation=False
            )
            
            mem_mb = 0
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
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
        
    print("Computing mean metrics...")
    mean_metrics = {}
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        mean_metrics[metric_name] = {}
        for match_metric in MATCH_METRICS:
            vals = [item['post'][metric_name][match_metric] for item in all_metrics 
                    if metric_name in item.get('post', {}) and match_metric in item['post'][metric_name]]
            
            if vals:
                mean_metrics[metric_name][match_metric] = (
                    round(float(np.mean(vals)) * 100, 2),
                    round(float(np.std(vals)) * 100, 2),
                )
            else:
                mean_metrics[metric_name][match_metric] = (0, 0)
                
    time_arr = [m["time"] for m in all_metrics]
    mem_arr = [m["max_memory"] for m in all_metrics]
    
    if time_arr:
        mean_metrics["time"] = (
            round(float(np.mean(time_arr)), 3),
            round(float(np.std(time_arr)), 3),
        )
    if mem_arr:
        mean_metrics["max_memory"] = (
            round(float(np.mean(mem_arr)), 3),
            round(float(np.std(mem_arr)), 3),
        )
    
    results_path = os.path.join(script_dir, "results.json")
    mean_results_path = os.path.join(script_dir, "mean_results.json")
    
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
    with open(mean_results_path, "w") as f:
        json.dump(mean_metrics, f, ensure_ascii=False, indent=4)
        
    print(f"Done! Results saved to:")
    print(f"- {results_path}")
    print(f"- {mean_results_path}")

if __name__ == "__main__":
    main()