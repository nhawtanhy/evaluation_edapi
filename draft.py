import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuyTran1301/Deepseek_PROD_ApiDeprecated"

# =========================
# Load tokenizer + model
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, # Use float32 for stability
    trust_remote_code=True
).cuda()
model.eval()

# Fix padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =========================
# IMPORTANT: use correct instruction format
# (your model is EDAPI/instruction tuned, NOT raw chat)
# =========================
prompt = "import pandas as pd\nimport numpy as np\n\n# Create a dataframe\ndf = pd."

# =========================
# Tokenize
# =========================
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

# Ensure attention mask exists (important fix)
if "attention_mask" not in inputs:
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

# =========================
# Generate
# =========================
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
        do_sample=False,        # Turn off sampling
        repetition_penalty=1.2, # Turn off penalty
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# =========================
# Decode ONLY generated part
# =========================
generated = outputs[0][inputs["input_ids"].shape[1]:]

print("=== OUTPUT ===")
print(tokenizer.decode(generated, skip_special_tokens=True).strip())