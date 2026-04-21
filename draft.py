import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

# IMPORTANT FIX
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

prompt = "Write a simple sentence:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,          # 👈 FIX 1 (important)
        top_p=0.9,               # 👈 FIX 2
        temperature=0.7,         # 👈 FIX 3
        repetition_penalty=1.1,  # 👈 FIX 4 (VERY useful)
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))