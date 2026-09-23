import os

# ============================================================
# CUDA
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 避免 tokenizer 多进程相关问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from pathlib import Path
from typing import List

import numpy as np
import torch

from datasets import load_dataset

from rouge_score import rouge_scorer

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
)

from trl import (
    GRPOTrainer,
    GRPOConfig,
)


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# SFT 完成后的模型
# ------------------------------------------------------------

MODEL_PATH = (
    BASE_DIR
    / "sft_output"
)
# MODEL_PATH = (
#     BASE_DIR
#     / "dpo_output"
# )

# ------------------------------------------------------------
# GRPO 数据
# ------------------------------------------------------------

DATA_PATH = (
    BASE_DIR
    / "05_train_grpo"
    / "data"
    / "grpo_train.jsonl"
)

# ------------------------------------------------------------
# GRPO 输出
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "grpo_output"
)


# ============================================================
# 2. 超参数
# ============================================================

NUM_GENERATIONS = 4

MAX_COMPLETION_LENGTH = 256

LEARNING_RATE = 1e-6

NUM_EPOCHS = 1

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8


# ============================================================
# Reward 权重
# ============================================================

ROUGE_WEIGHT = 0.20

SEMANTIC_WEIGHT = 0.50

KEYWORD_WEIGHT = 0.30


# ============================================================
# 3. 打印环境
# ============================================================

print("=" * 80)
print("Qwen3-0.6B GRPO Training")
print("=" * 80)

print(f"MODEL_PATH  : {MODEL_PATH}")
print(f"DATA_PATH   : {DATA_PATH}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")

print()

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"模型不存在：{MODEL_PATH}"
    )

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"GRPO数据不存在：{DATA_PATH}"
    )

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA 不可用，请检查 PyTorch CUDA 环境。"
    )

print(
    f"GPU         : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"CUDA        : "
    f"{torch.version.cuda}"
)

print()


# ============================================================
# 4. Tokenizer
# ============================================================

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

print(
    f"vocab_size       = {len(tokenizer)}"
)

print(
    f"pad_token        = {tokenizer.pad_token}"
)

print(
    f"pad_token_id     = {tokenizer.pad_token_id}"
)

print(
    f"eos_token        = {tokenizer.eos_token}"
)

print(
    f"eos_token_id     = {tokenizer.eos_token_id}"
)

print()


# ============================================================
# 5. Dataset
# ============================================================

print("=" * 80)
print("Loading dataset")
print("=" * 80)

dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train",
)

print(dataset)

print()

print("First sample:")

print(dataset[0])

print()


# ============================================================
# 6. ROUGE-L
# ============================================================

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
)


# ============================================================
# 7. Semantic Similarity Model
# ============================================================
#
# 默认使用 multilingual MiniLM。
#
# 如果你的环境不能联网，需要提前下载到本地，
# 然后把下面 MODEL 名称改成本地路径。
#
# ============================================================

SEMANTIC_MODEL_PATH = BASE_DIR / "sentence_transformer_configs"

print("=" * 80)
print("Loading semantic model")
print("=" * 80)

semantic_model = SentenceTransformer(
    str(SEMANTIC_MODEL_PATH),
    local_files_only=True
)

semantic_model.eval()

print(
    f"Semantic model: "
    f"{SEMANTIC_MODEL_PATH}"
)

print()


# ============================================================
# 8. completion 提取
# ============================================================

def extract_completion_text(
    completion,
):

    if completion is None:
        return ""

    # --------------------------------------------------------
    # Chat format
    #
    # [
    #   {
    #       "role": "assistant",
    #       "content": "..."
    #   }
    # ]
    # --------------------------------------------------------

    if isinstance(
        completion,
        list,
    ):

        if len(completion) == 0:
            return ""

        # 找最后一个 content
        for item in reversed(completion):

            if isinstance(
                item,
                dict,
            ):

                if "content" in item:

                    return str(
                        item["content"]
                    )

        return str(
            completion[-1]
        )

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    return str(completion)


# ============================================================
# 9. 文本归一化
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    # 去掉常见 markdown
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    return text


# ============================================================
# 10. Semantic Similarity
# ============================================================

def calculate_semantic_similarity(
    responses: List[str],
    references: List[str],
):

    if not responses:
        return []

    # --------------------------------------------------------
    # embedding
    # --------------------------------------------------------

    response_embeddings = (
        semantic_model.encode(
            responses,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    reference_embeddings = (
        semantic_model.encode(
            references,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    # --------------------------------------------------------
    # cosine similarity
    #
    # 因为已经 normalize_embeddings=True
    # 所以 cosine similarity = dot product
    # --------------------------------------------------------

    similarities = np.sum(
        response_embeddings
        * reference_embeddings,
        axis=1,
    )

    # --------------------------------------------------------
    # [-1, 1] → [0, 1]
    # --------------------------------------------------------

    similarities = (
        similarities + 1.0
    ) / 2.0

    similarities = np.clip(
        similarities,
        0.0,
        1.0,
    )

    return [
        float(x)
        for x in similarities
    ]


# ============================================================
# 11. 自动提取关键词
# ============================================================
#
# 第一版不要求你人工给每条 QA 写 keywords。
#
# 使用标准答案中的：
#
# 中文：
#   2~8 字连续片段
#
# 英文：
#   word
#
# 再过滤太常见的词。
#
# ============================================================

STOPWORDS = {
    "的是",
    "就是",
    "可以",
    "一种",
    "一个",
    "这个",
    "那个",
    "通过",
    "进行",
    "用于",
    "具有",
    "以及",
    "因此",
    "因为",
    "所以",
    "能够",
    "如果",
    "通常",
    "主要",
    "相关",
    "其中",
    "方法",
    "问题",
    "系统",
    "数据",
    "模型",
}


def extract_keywords(
    reference,
):

    reference = normalize_text(
        reference
    )

    keywords = []

    # --------------------------------------------------------
    # 中文连续片段
    # --------------------------------------------------------

    chinese_chars = []

    for char in reference:

        if (
            "\u4e00"
            <= char
            <= "\u9fff"
        ):

            chinese_chars.append(
                char
            )

        else:

            if len(chinese_chars) >= 2:

                text = "".join(
                    chinese_chars
                )

                # 2~6 gram
                max_n = min(
                    6,
                    len(text),
                )

                for n in range(
                    2,
                    max_n + 1,
                ):

                    for i in range(
                        len(text) - n + 1
                    ):

                        phrase = (
                            text[
                                i:i+n
                            ]
                        )

                        if phrase in STOPWORDS:
                            continue

                        keywords.append(
                            phrase
                        )

            chinese_chars = []

    # --------------------------------------------------------
    # 英文/数字 token
    # --------------------------------------------------------

    import re

    english_tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9_\-]{1,30}",
        reference,
    )

    keywords.extend(
        english_tokens
    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    unique = []

    seen = set()

    for keyword in keywords:

        keyword = keyword.strip()

        if not keyword:
            continue

        if keyword in seen:
            continue

        seen.add(keyword)

        unique.append(
            keyword
        )

    # --------------------------------------------------------
    # 太多关键词会导致 reward 不稳定
    # --------------------------------------------------------

    # 优先较长的关键词
    unique.sort(
        key=len,
        reverse=True,
    )

    return unique[:20]


# ============================================================
# 12. Keyword Reward
# ============================================================

def calculate_keyword_score(
    response,
    reference,
):

    response = normalize_text(
        response
    )

    reference = normalize_text(
        reference
    )

    if not response:
        return 0.0

    keywords = extract_keywords(
        reference
    )

    if not keywords:
        return 0.0

    matched = 0

    for keyword in keywords:

        if keyword in response:

            matched += 1

    score = (
        matched
        / len(keywords)
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# 13. ROUGE Reward
# ============================================================

def calculate_rouge_score(
    response,
    reference,
):

    response = normalize_text(
        response
    )

    reference = normalize_text(
        reference
    )

    if not response:
        return 0.0

    if not reference:
        return 0.0

    score = rouge.score(
        reference,
        response,
    )

    return float(
        score[
            "rougeL"
        ].fmeasure
    )


# ============================================================
# 14. 主 Reward Function
# ============================================================

def combined_reward(
    completions,
    solution,
    **kwargs,
):

    responses = []

    references = []

    # --------------------------------------------------------
    # 提取文本
    # --------------------------------------------------------

    for completion, reference in zip(
        completions,
        solution,
    ):

        response = extract_completion_text(
            completion
        )

        response = normalize_text(
            response
        )

        reference = normalize_text(
            reference
        )

        responses.append(
            response
        )

        references.append(
            reference
        )

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    semantic_scores = (
        calculate_semantic_similarity(
            responses,
            references,
        )
    )

    rewards = []

    # --------------------------------------------------------
    # 每个 completion
    # --------------------------------------------------------

    for i in range(
        len(responses)
    ):

        response = responses[i]

        reference = references[i]

        # --------------------------------------------
        # ROUGE-L
        # --------------------------------------------

        rouge_score = (
            calculate_rouge_score(
                response,
                reference,
            )
        )

        # --------------------------------------------
        # Semantic
        # --------------------------------------------

        semantic_score = (
            semantic_scores[i]
        )

        # --------------------------------------------
        # Keyword
        # --------------------------------------------

        keyword_score = (
            calculate_keyword_score(
                response,
                reference,
            )
        )

        # --------------------------------------------
        # 最终 reward
        # --------------------------------------------

        reward = (

            ROUGE_WEIGHT
            * rouge_score

            +

            SEMANTIC_WEIGHT
            * semantic_score

            +

            KEYWORD_WEIGHT
            * keyword_score
        )

        reward = float(
            np.clip(
                reward,
                0.0,
                1.0,
            )
        )

        rewards.append(
            reward
        )

    return rewards


# ============================================================
# 15. GRPO Config
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)

# 每个 optimizer step 实际消耗的训练样本数
effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
num_examples = len(dataset)

# 每个 epoch 的 optimizer steps
steps_per_epoch = (
    num_examples + effective_batch_size - 1
) // effective_batch_size

# 总 optimizer steps（必须与 NUM_EPOCHS 一致）
total_steps = steps_per_epoch * NUM_EPOCHS

# warmup = 总训练步数的 3%
warmup_steps = max(1, int(total_steps * 0.03))

training_args = GRPOConfig(

    # ========================================================
    # output
    # ========================================================

    output_dir=str(
        OUTPUT_DIR
    ),

    # ========================================================
    # batch
    # ========================================================

    per_device_train_batch_size=(
        BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    # ========================================================
    # GRPO
    # ========================================================

    num_generations=(
        NUM_GENERATIONS
    ),

    # ========================================================
    # Generation
    # ========================================================

    max_completion_length=(
        MAX_COMPLETION_LENGTH
    ),

    temperature=0.9,

    top_p=1.0,

    # ========================================================
    # Learning rate
    # ========================================================

    learning_rate=(
        LEARNING_RATE
    ),

    weight_decay=0.01,

    warmup_steps=warmup_steps,

    # ========================================================
    # Training
    # ========================================================

    num_train_epochs=(
        NUM_EPOCHS
    ),

    # ========================================================
    # Precision
    # ========================================================

    bf16=use_bf16,

    fp16=(
        torch.cuda.is_available()
        and not use_bf16
    ),

    # ========================================================
    # Logging
    # ========================================================

    logging_steps=1,

    report_to="none",

    # ========================================================
    # Saving
    # ========================================================

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    # ========================================================
    # Gradient
    # ========================================================

    gradient_checkpointing=True,

    # ========================================================
    # Seed
    # ========================================================

    seed=42,

    # ========================================================
    # Reward weights
    #
    # combined_reward 是一个 reward function
    # 所以这里不需要多个 weight。
    #
    # ========================================================
)


# ============================================================
# 16. GRPO Trainer
# ============================================================

print("=" * 80)
print("Creating GRPO Trainer")
print("=" * 80)

trainer = GRPOTrainer(

    model=str(
        MODEL_PATH
    ),

    reward_funcs=[
        combined_reward
    ],

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,
)


# ============================================================
# 17. Training
# ============================================================

print()
print("=" * 80)
print("START GRPO TRAINING")
print("=" * 80)
print()

trainer.train()


# ============================================================
# 18. 保存模型
# ============================================================

print()
print("=" * 80)
print("Saving model")
print("=" * 80)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

print()
print("=" * 80)
print("GRPO TRAINING FINISHED")
print("=" * 80)

print(
    f"Model saved to:\n"
    f"{OUTPUT_DIR}"
)
