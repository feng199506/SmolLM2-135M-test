from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ============================================================
# 1. 本地模型路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "lm_output" / "final"


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)


# ============================================================
# 3. 加载模型
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)

model.eval()


# ============================================================
# 4. 打印信息
# ============================================================

print("=" * 60)
print("MODEL INFO")
print("=" * 60)

print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length    :", len(tokenizer))

print("Model vocab_size    :", model.config.vocab_size)
print("Model device        :", model.device)

print("Embedding shape     :", model.get_input_embeddings().weight.shape)
print("LM head shape       :", model.get_output_embeddings().weight.shape)


# ============================================================
# 5. Tokenizer 测试
# ============================================================

prompt = "列表的基本概念是什么？"

input_ids = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
)["input_ids"]

print("\n" + "=" * 60)
print("TOKENIZER TEST")
print("=" * 60)

print("Prompt:")
print(prompt)

print("\nInput IDs:")
print(input_ids[0].tolist())

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))

print("\nDecode:")
print(tokenizer.decode(
    input_ids[0],
    skip_special_tokens=False,
))


# ============================================================
# 6. Base Model 直接生成
# ============================================================

model_inputs = {
    "input_ids": input_ids.to(model.device),
    "attention_mask": torch.ones_like(input_ids).to(model.device),
}

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=False,
)


# ============================================================
# 7. 只取生成部分
# ============================================================

output_ids = generated_ids[
    0
][len(input_ids[0]):].tolist()


# ============================================================
# 8. 输出 token
# ============================================================

print("\n" + "=" * 60)
print("GENERATED TOKENS")
print("=" * 60)

print("Generated IDs:")
print(output_ids[:100])

print("\nGenerated tokens:")
print(tokenizer.convert_ids_to_tokens(output_ids[:100]))

print("\nRaw decode:")
print(tokenizer.decode(
    output_ids,
    skip_special_tokens=False,
))

print("\nClean decode:")
print(tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
).strip())
