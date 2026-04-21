import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompt = "Write a simple sentence:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))