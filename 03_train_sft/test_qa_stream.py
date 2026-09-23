from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


# ============================================================
# 1. 本地模型路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
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

print("Model loaded")
print("Model device:", model.device)
print("Model vocab_size:", model.config.vocab_size)


# ============================================================
# 5. 准备模型输入
# ============================================================

prompt = "什么是列表？"

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)


# ============================================================
# 6. Streamer
# ============================================================

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

print("\n" + "=" * 60)
print("PROMPT:")
print(prompt)
print("=" * 60)
print()

_ = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=False,
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id,
)
