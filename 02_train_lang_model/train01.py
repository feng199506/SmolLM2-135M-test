from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

def main():
    # ============================================================
    # 1. 路径
    # ============================================================

    BASE_DIR = Path(__file__).resolve().parent.parent
    print(BASE_DIR)
    TOKENIZER_PATH = BASE_DIR / "tokenizer"
    CONFIG_PATH = BASE_DIR / "configs" / "config.json"
    CORPUS_PATH = BASE_DIR / "corpus"
    OUTPUT_PATH = BASE_DIR / "lm_output"


    # ============================================================
    # 2. 加载自己的 tokenizer
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        trust_remote_code=True,
    )

    print("=" * 80)
    print("TOKENIZER")
    print("=" * 80)

    print("tokenizer.vocab_size :", tokenizer.vocab_size)
    print("len(tokenizer)       :", len(tokenizer))
    print("pad_token            :", tokenizer.pad_token)
    print("eos_token            :", tokenizer.eos_token)
    print("bos_token            :", tokenizer.bos_token)
    print("unk_token            :", tokenizer.unk_token)

    # 非常重要：
    # 使用实际 tokenizer 大小，而不是 tokenizer.vocab_size
    VOCAB_SIZE = len(tokenizer)

    print("model vocab_size     :", VOCAB_SIZE)


    # ============================================================
    # 3. 加载 Qwen3-0.6B 配置
    # ============================================================

    config = AutoConfig.from_pretrained(
        str(CONFIG_PATH),
        trust_remote_code=True,
    )

    # 使用自己的 tokenizer vocabulary
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    print()
    print("=" * 80)
    print("MODEL CONFIG")
    print("=" * 80)

    print("vocab_size          :", config.vocab_size)
    print("hidden_size         :", config.hidden_size)
    print("num_hidden_layers   :", config.num_hidden_layers)
    print("num_attention_heads :", config.num_attention_heads)
    print("num_key_value_heads :", config.num_key_value_heads)
    print("intermediate_size   :", config.intermediate_size)


    # ============================================================
    # 4. 从零初始化模型
    # ============================================================

    print()
    print("=" * 80)
    print("INITIALIZING MODEL FROM SCRATCH")
    print("=" * 80)

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    print(model)

    print()
    print("Model parameters:")
    print(f"{sum(p.numel() for p in model.parameters()):,}")


    # ============================================================
    # 5. 检查 embedding vocabulary
    # ============================================================

    input_embeddings = model.get_input_embeddings()

    print()
    print("=" * 80)
    print("VOCAB CHECK")
    print("=" * 80)

    print("Tokenizer vocab :", len(tokenizer))
    print("Config vocab    :", config.vocab_size)
    print("Embedding vocab :", input_embeddings.num_embeddings)

    assert len(tokenizer) == config.vocab_size
    assert len(tokenizer) == input_embeddings.num_embeddings
    assert (
            tokenizer.pad_token_id is None
            or 0 <= tokenizer.pad_token_id < len(tokenizer)
    )

    assert (
            tokenizer.bos_token_id is None
            or 0 <= tokenizer.bos_token_id < len(tokenizer)
    )

    assert (
            tokenizer.eos_token_id is None
            or 0 <= tokenizer.eos_token_id < len(tokenizer)
    )

    print("VOCAB SIZE CHECK: OK")


    # ============================================================
    # 6. 检查 tokenizer chat_template
    # ============================================================

    print()
    print("=" * 80)
    print("CHAT TEMPLATE")
    print("=" * 80)

    if tokenizer.chat_template is not None:
        print("chat_template: OK")
    else:
        print("chat_template: None")


    # ============================================================
    # 7. 加载 corpus
    # ============================================================

    print()
    print("=" * 80)
    print("LOADING CORPUS")
    print("=" * 80)

    text_files = list(CORPUS_PATH.glob("*.txt"))

    if not text_files:
        raise RuntimeError(
            f"No .txt files found in {CORPUS_PATH}"
        )

    print(f"Found {len(text_files)} text files")

    dataset = load_dataset(
        "text",
        data_files={
            "train": [str(x) for x in text_files]
        },
    )

    print(dataset)


    # ============================================================
    # 8. Tokenize
    # ============================================================

    MAX_LENGTH = 2048


    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=False,
        )


    print()
    print("=" * 80)
    print("TOKENIZING")
    print("=" * 80)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        desc="Tokenizing corpus",
    )

    print(tokenized_dataset)


    # ============================================================
    # 9. Packing
    #
    # 把大量短 sequence 拼成连续的 MAX_LENGTH block
    # ============================================================

    def group_texts(examples):

        # 将 batch 中所有 token 拼起来
        concatenated = {
            key: sum(examples[key], [])
            for key in examples.keys()
        }

        total_length = len(concatenated["input_ids"])

        # 丢掉最后不足 MAX_LENGTH 的部分
        total_length = (
            total_length // MAX_LENGTH
        ) * MAX_LENGTH

        result = {
            key: [
                t[i: i + MAX_LENGTH]
                for i in range(0, total_length, MAX_LENGTH)
            ]
            for key, t in concatenated.items()
        }

        # Causal LM labels = input_ids
        result["labels"] = result["input_ids"].copy()

        return result


    print()
    print("=" * 80)
    print("PACKING")
    print("=" * 80)

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc=f"Packing into {MAX_LENGTH} token blocks",
    )

    print(lm_dataset)

    print(
        "Training examples:",
        len(lm_dataset["train"])
    )


    # ============================================================
    # 10. Data collator
    # ============================================================

    data_collator = default_data_collator


    # ============================================================
    # 11. Training configuration
    # ============================================================

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )
    num_examples = len(lm_dataset["train"])

    per_device_bs = 1
    grad_accum = 16
    num_epochs = 1

    # 每个 optimizer step 实际消耗的训练样本数
    effective_batch_size = per_device_bs * grad_accum

    # 每个 epoch 的 optimizer steps
    steps_per_epoch = (
                              num_examples + effective_batch_size - 1
                      ) // effective_batch_size

    # 总 optimizer steps
    total_steps = steps_per_epoch * num_epochs

    # warmup = 总训练步数的 3%
    warmup_steps = max(1, int(total_steps * 0.03))

    print()
    print("=" * 80)
    print("TRAINING STEPS")
    print("=" * 80)
    print("Training examples   :", num_examples)
    print("Effective batch size:", effective_batch_size)
    print("Steps per epoch     :", steps_per_epoch)
    print("Total steps         :", total_steps)
    print("Warmup steps        :", warmup_steps)
    training_args = TrainingArguments(

        output_dir=str(OUTPUT_PATH),

        # -------------------------
        # batch
        # -------------------------

        per_device_train_batch_size=per_device_bs,

        gradient_accumulation_steps=grad_accum,

        # -------------------------
        # training
        # -------------------------

        num_train_epochs=num_epochs,

        # 如果以后使用 max_steps，
        # 可以设置：
        #
        # max_steps=10000,

        # -------------------------
        # optimizer
        # -------------------------

        learning_rate=3e-4,

        weight_decay=0.1,

        adam_beta1=0.9,

        adam_beta2=0.95,

        adam_epsilon=1e-8,

        # -------------------------
        # scheduler
        # -------------------------

        lr_scheduler_type="cosine",

        warmup_steps=warmup_steps,

        # -------------------------
        # precision
        # -------------------------

        fp16=torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported(),

        bf16=torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),

        # -------------------------
        # logging
        # -------------------------

        logging_steps=10,

        logging_first_step=True,

        # -------------------------
        # save
        # -------------------------

        save_strategy="steps",

        save_steps=1000,

        save_total_limit=2,

        # -------------------------
        # reporting
        # -------------------------

        report_to="none",

        # -------------------------
        # dataloader
        # -------------------------

        dataloader_num_workers=0,

        # -------------------------
        # performance
        # -------------------------

        gradient_checkpointing=True,

        # -------------------------
        # seed
        # -------------------------

        seed=42,

        data_seed=42,

        # -------------------------
        # remove unused columns
        # -------------------------

        remove_unused_columns=False,
    )


    # ============================================================
    # 12. Trainer
    # ============================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )


    # ============================================================
    # 13. 开始训练
    # ============================================================

    print()
    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)

    print(f"Device      : {model.device}")
    print(f"Vocab size  : {config.vocab_size}")
    print(f"Max length  : {MAX_LENGTH}")
    print(
        f"Batch size  : "
        f"{training_args.per_device_train_batch_size}"
    )
    print(
        f"Grad accum  : "
        f"{training_args.gradient_accumulation_steps}"
    )
    print(
        f"Effective BS: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(
        f"Learning rate: "
        f"{training_args.learning_rate}"
    )


    trainer.train()


    # ============================================================
    # 14. 保存最终模型
    # ============================================================

    FINAL_PATH = OUTPUT_PATH / "final"

    print()
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    trainer.save_model(str(FINAL_PATH))
    tokenizer.save_pretrained(str(FINAL_PATH))

    print(f"Model saved to: {FINAL_PATH}")


    # ============================================================
    # 15. 简单测试
    # ============================================================

    print()
    print("=" * 80)
    print("TEST GENERATION")
    print("=" * 80)

    test_prompt = "人工智能是一种"

    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False,
    )

    print(result)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
