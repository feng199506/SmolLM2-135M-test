import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ============================================================
# 1. 路径
# ============================================================

MODEL_PATH = r"..\lm_output\final"

DATA_PATH = r".\data"

OUTPUT_PATH = r"..\sft_output"


# ============================================================
# 2. Chat Template
# ============================================================

CHAT_TEMPLATE = (
    "{% for message in messages %}"

    "{% if message['role'] == 'assistant' %}"

    "{{ '<|im_start|>assistant\\n' }}"

    "{% generation %}"
    "{{ message['content'] }}"
    "{{ '<|im_end|>\\n' }}"
    "{% endgeneration %}"

    "{% else %}"

    "{{ '<|im_start|>' + message['role'] + '\\n' "
    "+ message['content'] + '<|im_end|>\\n' }}"

    "{% endif %}"

    "{% endfor %}"

    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)
# ============================================================
# 3. 加载 tokenizer
# ============================================================

print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)


print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length   :", len(tokenizer))


# ============================================================
# 4. 设置 Chat Template
# ============================================================

tokenizer.chat_template = CHAT_TEMPLATE


print()
print("Chat template configured.")


# ============================================================
# 5. Special Tokens
# ============================================================

im_start_id = tokenizer.convert_tokens_to_ids(
    "<|im_start|>"
)

im_end_id = tokenizer.convert_tokens_to_ids(
    "<|im_end|>"
)

eot_id = tokenizer.convert_tokens_to_ids(
    "<|endoftext|>"
)


print()
print("=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)

print("<|endoftext|> :", eot_id)
print("<|im_start|> :", im_start_id)
print("<|im_end|>   :", im_end_id)

print("EOS token     :", tokenizer.eos_token)
print("EOS token ID  :", tokenizer.eos_token_id)

print("PAD token     :", tokenizer.pad_token)
print("PAD token ID  :", tokenizer.pad_token_id)


# ============================================================
# 6. 强制 EOS = im_end
# ============================================================

if im_end_id is None:
    raise ValueError(
        "<|im_end|> does not exist in tokenizer."
    )

tokenizer.eos_token = "<|im_end|>"
tokenizer.eos_token_id = im_end_id


# ============================================================
# 7. PAD
# ============================================================

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

print()
print("Final EOS:", tokenizer.eos_token)
print("Final EOS ID:", tokenizer.eos_token_id)

print("Final PAD:", tokenizer.pad_token)
print("Final PAD ID:", tokenizer.pad_token_id)


# ============================================================
# 8. 加载模型
# ============================================================

print()
print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype="bfloat16",
)

if not __import__("torch").cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

model = model.to("cuda")


print("Model device:", model.device)
print(
    "Model dtype:",
    next(model.parameters()).dtype,
)

print(
    "Model vocab_size:",
    model.config.vocab_size,
)


# ============================================================
# 9. tokenizer / model vocab
# ============================================================

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != "
    f"{model.config.vocab_size}"
)


# ============================================================
# 10. 模型 generation config
# ============================================================

model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "generation_config"):

    model.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )


# ============================================================
# 11. 加载数据
# ============================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

data_dir = Path(DATA_PATH)

data_files = (
    list(data_dir.glob("*.json"))
    + list(data_dir.glob("*.jsonl"))
)

if not data_files:

    raise FileNotFoundError(
        f"No .json/.jsonl files found in {DATA_PATH}"
    )


for file in data_files:

    print("  ", file)


dataset = load_dataset(
    "json",
    data_files=[
        str(file)
        for file in data_files
    ],
    split="train",
)

# dataset = dataset.select(range(100))
# for item in dataset:
#     print(item)
print()
print("Dataset size:", len(dataset))
print("Columns:", dataset.column_names)


# ============================================================
# 12. 检查 messages
# ============================================================

if "messages" not in dataset.column_names:

    raise ValueError(
        "Dataset must contain 'messages'."
    )


for i in range(min(10, len(dataset))):

    messages = dataset[i]["messages"]

    if not isinstance(messages, list):

        raise ValueError(
            f"Sample {i}: messages is not list."
        )

    for message in messages:

        if message["role"] not in (
            "system",
            "user",
            "assistant",
        ):

            raise ValueError(
                f"Invalid role: "
                f"{message['role']}"
            )

        if "content" not in message:

            raise ValueError(
                f"Sample {i}: "
                "message has no content."
            )


# ============================================================
# 13. 训练配置
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_PATH,

    # num_train_epochs=5,
    # num_train_epochs=50,
    num_train_epochs=20,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    max_length=1024,

    logging_steps=10,

    save_strategy="steps",

    save_steps=1000,

    save_total_limit=2,

    bf16=True,

    tf32=True,

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    assistant_only_loss=True,

    report_to="none",

    remove_unused_columns=False,

    optim="adamw_torch_fused",

)


# ============================================================
# 14. Trainer
# ============================================================

print()
print("=" * 80)
print("CREATING SFT TRAINER")
print("=" * 80)

trainer = SFTTrainer(

    model=model,

    processing_class=tokenizer,

    train_dataset=dataset,

    args=training_args,

)


# ============================================================
# 15. 开始训练
# ============================================================

print()
print("=" * 80)
print("START QA SFT TRAINING")
print("=" * 80)

trainer.train()


# ============================================================
# 16. 保存模型
# ============================================================

print()
print("=" * 80)
print("SAVING MODEL")
print("=" * 80)

trainer.save_model(
    OUTPUT_PATH
)


# ============================================================
# 17. 保存 tokenizer
# ============================================================

tokenizer.save_pretrained(
    OUTPUT_PATH
)


# ============================================================
# 18. 最终信息
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Saved to:", OUTPUT_PATH)

print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))

print("EOS:", tokenizer.eos_token)
print("EOS ID:", tokenizer.eos_token_id)

print("PAD:", tokenizer.pad_token)
print("PAD ID:", tokenizer.pad_token_id)

print(
    "Chat template:",
    tokenizer.chat_template is not None,
)