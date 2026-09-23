import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import torch

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

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "lm_output" / "final"

DATA_PATH = Path(__file__).resolve().parent / "data"

OUTPUT_PATH = BASE_DIR / "sft_output"


print("=" * 80)
print("PATH CONFIGURATION")
print("=" * 80)

print("BASE_DIR   :", BASE_DIR)
print("MODEL_PATH :", MODEL_PATH)
print("DATA_PATH  :", DATA_PATH)
print("OUTPUT_PATH:", OUTPUT_PATH)


# ============================================================
# 2. 加载 tokenizer
# ============================================================

print()
print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
)

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length    :", len(tokenizer))


# ============================================================
# 3. Special Tokens
# ============================================================

print()
print("=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)

print("BOS token    :", tokenizer.bos_token)
print("BOS token ID :", tokenizer.bos_token_id)

print("EOS token    :", tokenizer.eos_token)
print("EOS token ID :", tokenizer.eos_token_id)

print("PAD token    :", tokenizer.pad_token)
print("PAD token ID :", tokenizer.pad_token_id)


# ============================================================
# 4. 查找已有 special tokens
# ============================================================

im_start_id = tokenizer.convert_tokens_to_ids(
    "<|im_start|>"
)

im_end_id = tokenizer.convert_tokens_to_ids(
    "<|im_end|>"
)

endoftext_id = tokenizer.convert_tokens_to_ids(
    "<|endoftext|>"
)


print()
print("=" * 80)
print("SPECIAL TOKEN IDS")
print("=" * 80)

print(
    "<|im_start|>:",
    im_start_id
)

print(
    "<|im_end|>:",
    im_end_id
)

print(
    "<|endoftext|>:",
    endoftext_id
)


# ============================================================
# 5. 构造 Chat Template
# ============================================================
#
# 你的 tokenizer 没有 chat_template。
#
# 所以这里必须自己提供。
#
# 但是：
#
# 1. 不使用 {% generation %}
# 2. 不使用 assistant_only_loss
# 3. 不修改 EOS
#
# ============================================================

if (
    im_start_id is not None
    and im_end_id is not None
):

    CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{{ message['content'] }}"
        "{{ '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )

    tokenizer.chat_template = CHAT_TEMPLATE

    print()
    print(
        "Using <|im_start|>/<|im_end|> "
        "chat template."
    )

else:

    raise ValueError(
        """
Tokenizer does not contain:

<|im_start|>
<|im_end|>

Therefore a role-based chat template cannot
be safely constructed.

Your tokenizer must contain these special
tokens if the dataset format is:

messages = [
    {"role": "user", ...},
    {"role": "assistant", ...}
]
"""
    )


# ============================================================
# 6. PAD
# ============================================================

# 不修改 EOS。
#
# 只有 PAD 不存在时才使用 EOS 作为 PAD。

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

    print()
    print(
        "PAD token was missing."
    )

    print(
        "PAD = EOS"
    )


print()
print("=" * 80)
print("FINAL TOKEN CONFIG")
print("=" * 80)

print(
    "EOS:",
    tokenizer.eos_token,
)

print(
    "EOS ID:",
    tokenizer.eos_token_id,
)

print(
    "PAD:",
    tokenizer.pad_token,
)

print(
    "PAD ID:",
    tokenizer.pad_token_id,
)


# ============================================================
# 7. 加载模型
# ============================================================

print()
print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)


# ============================================================
# 8. CUDA
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available."
    )


model = model.to("cuda")


print(
    "Model device:",
    model.device
)

print(
    "Model dtype:",
    next(model.parameters()).dtype
)

print(
    "Model vocab_size:",
    model.config.vocab_size
)


# ============================================================
# 9. vocab check
# ============================================================

print()
print("=" * 80)
print("VOCAB CHECK")
print("=" * 80)

print(
    "Tokenizer length:",
    len(tokenizer)
)

print(
    "Model vocab_size:",
    model.config.vocab_size
)


assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != "
    f"{model.config.vocab_size}"
)

print("Vocab check: OK")


# ============================================================
# 10. use_cache
# ============================================================

model.config.use_cache = False

print()
print(
    "use_cache:",
    model.config.use_cache
)


# ============================================================
# 11. Generation Config
# ============================================================

model.config.eos_token_id = (
    tokenizer.eos_token_id
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)


if hasattr(
    model,
    "generation_config",
):

    model.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )


# ============================================================
# 12. 参数统计
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print("=" * 80)
print("MODEL PARAMETERS")
print("=" * 80)

print(
    f"Total parameters    : "
    f"{total_params:,} "
    f"({total_params / 1e6:.2f} M)"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,} "
    f"({trainable_params / 1e6:.2f} M)"
)


# ============================================================
# 13. 加载 Dataset
# ============================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

data_dir = Path(DATA_PATH)

json_files = sorted(
    data_dir.glob("*.json")
)

jsonl_files = sorted(
    data_dir.glob("*.jsonl")
)

data_files = (
    json_files +
    jsonl_files
)


if not data_files:

    raise FileNotFoundError(
        f"No .json/.jsonl files found in "
        f"{DATA_PATH}"
    )


print("Found data files:")

for file in data_files:

    print(
        "  ",
        file
    )


dataset = load_dataset(
    "json",
    data_files=[
        str(file)
        for file in data_files
    ],
    split="train",
)


print()
print(
    "Dataset size:",
    len(dataset)
)

print(
    "Columns:",
    dataset.column_names
)


# ============================================================
# 14. messages 检查
# ============================================================

if "messages" not in dataset.column_names:

    raise ValueError(
        "Dataset must contain 'messages'."
    )


for i in range(
    min(10, len(dataset))
):

    messages = dataset[i]["messages"]

    if not isinstance(
        messages,
        list
    ):

        raise ValueError(
            f"Sample {i}: "
            "messages is not list."
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


print()
print("Dataset validation: OK")


# ============================================================
# 15. Chat Template Test
# ============================================================

print()
print("=" * 80)
print("CHAT TEMPLATE TEST")
print("=" * 80)


test_text = tokenizer.apply_chat_template(
    dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)


print(
    test_text[:2000]
)


# ============================================================
# 16. Tokenization Test
# ============================================================

test_ids = tokenizer(
    test_text,
    add_special_tokens=False,
)["input_ids"]


print()
print("=" * 80)
print("TOKENIZATION TEST")
print("=" * 80)

print(
    "Token count:",
    len(test_ids)
)


print()
print("First 100 tokens:")

for i, token_id in enumerate(
    test_ids[:100]
):

    token = tokenizer.convert_ids_to_tokens(
        token_id
    )

    print(
        f"{i:4d} "
        f"{token_id:6d} "
        f"{repr(token)}"
    )


# ============================================================
# 17. Training Plan
# ============================================================

per_device_bs = 1

grad_accum = 8

effective_batch_size = (
    per_device_bs *
    grad_accum
)

num_examples = len(dataset)

num_epochs = 20


steps_per_epoch = (
    num_examples +
    effective_batch_size -
    1
) // effective_batch_size


total_steps = (
    steps_per_epoch *
    num_epochs
)


warmup_steps = max(
    1,
    int(
        total_steps * 0.03
    ),
)


print()
print("=" * 80)
print("TRAINING PLAN")
print("=" * 80)

print(
    "Examples:",
    num_examples
)

print(
    "Epochs:",
    num_epochs
)

print(
    "Batch:",
    per_device_bs
)

print(
    "Gradient accumulation:",
    grad_accum
)

print(
    "Effective batch:",
    effective_batch_size
)

print(
    "Steps / epoch:",
    steps_per_epoch
)

print(
    "Total steps:",
    total_steps
)

print(
    "Warmup steps:",
    warmup_steps
)


# ============================================================
# 18. SFTConfig
# ============================================================

training_args = SFTConfig(

    output_dir=str(
        OUTPUT_PATH
    ),

    num_train_epochs=num_epochs,

    per_device_train_batch_size=(
        per_device_bs
    ),

    gradient_accumulation_steps=(
        grad_accum
    ),

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

    # ========================================================
    # 关键：
    # 不使用 assistant_only_loss
    # ========================================================

    assistant_only_loss=False,

    remove_unused_columns=False,

    optim="adamw_torch_fused",

    warmup_steps=warmup_steps,

    report_to="none",
)


# ============================================================
# 19. Trainer
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
# 20. 开始训练
# ============================================================

print()
print("=" * 80)
print("START SMOLLM2 SFT")
print("=" * 80)


trainer.train()


# ============================================================
# 21. 保存模型
# ============================================================

print()
print("=" * 80)
print("SAVING MODEL")
print("=" * 80)


trainer.save_model(
    str(OUTPUT_PATH)
)


# ============================================================
# 22. 保存 tokenizer
# ============================================================

tokenizer.save_pretrained(
    str(OUTPUT_PATH)
)


# ============================================================
# 23. 完成
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "Saved to:",
    OUTPUT_PATH
)

print(
    "Tokenizer vocab_size:",
    tokenizer.vocab_size
)

print(
    "Tokenizer length:",
    len(tokenizer)
)

print(
    "EOS:",
    tokenizer.eos_token
)

print(
    "EOS ID:",
    tokenizer.eos_token_id
)

print(
    "PAD:",
    tokenizer.pad_token
)

print(
    "PAD ID:",
    tokenizer.pad_token_id
)

print(
    "Chat template:",
    tokenizer.chat_template is not None
)

print()
print("=" * 80)
print("SMOLLM2 SFT COMPLETE")
print("=" * 80)