import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"

# Load
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# 🔥 CRITICAL FIX
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Simple prompt
prompt = "Hello, today is a beautiful day and"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,      # deterministic
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=== OUTPUT ===")
print(result)