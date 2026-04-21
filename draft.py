import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

# IMPORTANT (from your config)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# ===== CHAT FORMAT (CRITICAL FIX) =====
messages = [
    {"role": "user", "content": "Write a simple sentence."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))