from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 本地模型路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# 默认测 SFT；也可改成 grpo_output / dpo_output
MODEL_DIR = BASE_DIR / "sft_output"


# ============================================================
# 2. 加载本地 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)


# ============================================================
# 3. 加载本地模型
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

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))
print("Chat template:", tokenizer.chat_template is not None)

print("Model loaded")
print("Model device:", model.device)
print("Model vocab_size:", model.config.vocab_size)

if tokenizer.chat_template is None:
    raise ValueError(
        f"{MODEL_DIR} has no chat_template. "
        "Use sft_output / dpo_output / grpo_output."
    )


# ============================================================
# 5. 准备模型输入
# ============================================================

prompt = "列表的基本概念是什么？"

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("\n" + "=" * 60)
print("CHAT TEMPLATE")
print("=" * 60)
print(text)
model_inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=False,
)

output_ids = generated_ids[
    0
][len(model_inputs.input_ids[0]):].tolist()
print("\n" + "=" * 60)
print("PROMPT:")
print(prompt)
print("\nGenerated token IDs:")
print(output_ids[:100])

print("\nGenerated tokens:")
print(tokenizer.convert_ids_to_tokens(output_ids[:100]))

print("\nDecoded raw:")
print(tokenizer.decode(
    output_ids,
    skip_special_tokens=False,
))

print("\nDecoded clean:")
content = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
).strip()

print("CONTENT:")
print(content)
