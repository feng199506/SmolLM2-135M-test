from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 本地模型路径
# ============================================================

# MODEL_DIR = Path("../configs").resolve()
MODEL_DIR = r"..\sft_output"

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
    enable_thinking=False,
)

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

content = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
).strip()

print("\n" + "=" * 60)
print("PROMPT:")
print(prompt)
print("CONTENT:")
print(content)