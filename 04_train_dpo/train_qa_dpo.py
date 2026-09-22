import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


# ============================================================
# 1. 路径
# ============================================================

# SFT 训练完成后的模型
MODEL_PATH = r"..\sft_output"

# DPO 数据
DATA_PATH = r".\data"

# DPO 输出
OUTPUT_PATH = r"..\dpo_output"


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))
print("Chat template:", tokenizer.chat_template is not None)


# ============================================================
# 3. 加载模型
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype="auto",
)

print("Model loaded")
print("Model vocab_size:", model.config.vocab_size)


# ============================================================
# 4. 检查 tokenizer / model vocab
# ============================================================

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != {model.config.vocab_size}"
)

print("Vocab check: OK")


# ============================================================
# 5. 加载 DPO 数据
# ============================================================

data_dir = Path(DATA_PATH)

json_files = list(data_dir.glob("*.json"))
jsonl_files = list(data_dir.glob("*.jsonl"))

data_files = json_files + jsonl_files

print()
print("Found DPO data files:")

for file in data_files:
    print("  ", file)

if not data_files:
    raise FileNotFoundError(
        f"No .json or .jsonl files found in: {DATA_PATH}"
    )


dataset = load_dataset(
    "json",
    data_files=[str(f) for f in data_files],
    split="train",
)

print()
print("Dataset size:", len(dataset))
print("Dataset columns:", dataset.column_names)


# ============================================================
# 6. 检查 DPO 数据格式
# ============================================================

required_columns = [
    "prompt",
    "chosen",
    "rejected",
]

for column in required_columns:
    if column not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain '{column}' column."
        )


# ============================================================
# 7. 查看一条 DPO 数据
# ============================================================

print()
print("=" * 80)
print("DPO SAMPLE")
print("=" * 80)

print("PROMPT:")
print(dataset[0]["prompt"])

print()
print("CHOSEN:")
print(dataset[0]["chosen"])

print()
print("REJECTED:")
print(dataset[0]["rejected"])


# ============================================================
# 8. 检查 chat template
# ============================================================

if tokenizer.chat_template is None:
    raise ValueError(
        "Tokenizer does not have a chat_template."
    )


# ============================================================
# 9. DPO 训练参数
# ============================================================

training_args = DPOConfig(
    output_dir=OUTPUT_PATH,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=5e-6,

    beta=0.1,

    max_length=1024,

    logging_steps=10,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    bf16=True,

    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False,
)


# ============================================================
# 10. DPO Trainer
# ============================================================

trainer = DPOTrainer(
    model=model,

    processing_class=tokenizer,

    train_dataset=dataset,

    args=training_args,
)


# ============================================================
# 11. 开始 DPO
# ============================================================

print()
print("=" * 80)
print("START DPO TRAINING")
print("=" * 80)

trainer.train()


# ============================================================
# 12. 保存 DPO 模型
# ============================================================

trainer.save_model(OUTPUT_PATH)

tokenizer.save_pretrained(OUTPUT_PATH)


# ============================================================
# 13. 完成
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("DPO training finished")
print("Saved to:", OUTPUT_PATH)

