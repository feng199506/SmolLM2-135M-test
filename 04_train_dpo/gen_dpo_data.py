import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import json
import re
import time

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "sft_output"

DATA_PATH = BASE_DIR / "03_train_sft" / "data"

OUTPUT_DIR = BASE_DIR / "04_train_dpo" / "data"

OUTPUT_PATH = OUTPUT_DIR / "dpo_train.jsonl"


# ============================================================
# 2. 参数
# ============================================================

MAX_NEW_TOKENS = 256

MAX_SAMPLES = None

PRINT_EVERY = 1


# ============================================================
# 3. 时间
# ============================================================

def format_seconds(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m"


# ============================================================
# 4. tokenizer
# ============================================================

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
)


print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))
print(
    "Chat template:",
    tokenizer.chat_template is not None,
)


if tokenizer.chat_template is None:

    raise ValueError(
        "SFT tokenizer has no chat_template."
    )


# ============================================================
# 5. Special tokens
# ============================================================

print()
print("=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)

print("EOS:", tokenizer.eos_token)
print("EOS ID:", tokenizer.eos_token_id)

print("PAD:", tokenizer.pad_token)
print("PAD ID:", tokenizer.pad_token_id)

print(
    "<|im_start|>:",
    tokenizer.convert_tokens_to_ids(
        "<|im_start|>"
    )
)

print(
    "<|im_end|>:",
    tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )
)


# ============================================================
# 6. model
# ============================================================

print()
print("=" * 80)
print("Loading SFT model")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    torch_dtype="bfloat16",
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available."
    )

model = model.to("cuda")
model.eval()


print("Model device:", model.device)

print(
    "Model dtype:",
    next(model.parameters()).dtype,
)

print(
    "CUDA device:",
    torch.cuda.get_device_name(0),
)

print(
    "CUDA memory:",
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print(
    "Model vocab_size:",
    model.config.vocab_size,
)


assert len(tokenizer) == model.config.vocab_size


# ============================================================
# 7. generation config
# ============================================================

model.config.eos_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id

model.generation_config.eos_token_id = (
    tokenizer.eos_token_id
)

model.generation_config.pad_token_id = (
    tokenizer.pad_token_id
)


# ============================================================
# 8. dataset
# ============================================================

print()
print("=" * 80)
print("Loading SFT dataset")
print("=" * 80)

data_files = (
    list(Path(DATA_PATH).glob("*.json"))
    + list(Path(DATA_PATH).glob("*.jsonl"))
)

if not data_files:

    raise FileNotFoundError(
        f"No data files found in {DATA_PATH}"
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


print()
print("Dataset size:", len(dataset))
print("Columns:", dataset.column_names)


if "messages" not in dataset.column_names:

    raise ValueError(
        "Dataset must contain messages."
    )


# ============================================================
# 9. question
# ============================================================

def get_user_question(messages):

    for message in reversed(messages):

        if message["role"] == "user":

            return message["content"].strip()

    return ""


# ============================================================
# 10. chosen
# ============================================================

def get_reference_answer(messages):

    for message in reversed(messages):

        if message["role"] == "assistant":

            return message["content"].strip()

    return ""


# ============================================================
# 11. generate rejected
# ============================================================

@torch.inference_mode()
def generate_rejected(
    question,
    index,
    total,
):

    start = time.time()

    print()
    print("-" * 80)

    print(
        f"[{index}/{total}] GENERATION START",
        flush=True,
    )

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]


    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )


    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }


    input_length = inputs["input_ids"].shape[1]


    print(
        f"[{index}/{total}] Input tokens: "
        f"{input_length}",
        flush=True,
    )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    generation_start = time.time()


    print(
        f"[{index}/{total}] "
        "Calling generate()...",
        flush=True,
    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        num_beams=1,

        use_cache=True,

        eos_token_id=tokenizer.eos_token_id,

        pad_token_id=tokenizer.pad_token_id,

    )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    generation_time = (
        time.time()
        - generation_start
    )


    generated_tokens = outputs[
        0,
        input_length:
    ]


    output_length = (
        generated_tokens.shape[0]
    )


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


    total_time = time.time() - start


    print(
        f"[{index}/{total}] "
        f"Output tokens: {output_length}",
        flush=True,
    )


    print(
        f"[{index}/{total}] "
        f"Generation time: "
        f"{format_seconds(generation_time)}",
        flush=True,
    )


    if output_length > 0:

        print(
            f"[{index}/{total}] "
            f"Speed: "
            f"{output_length / generation_time:.2f} tok/s",
            flush=True,
        )


    print(
        f"[{index}/{total}] "
        f"Total time: "
        f"{format_seconds(total_time)}",
        flush=True,
    )


    return answer.strip()


# ============================================================
# 12. clean
# ============================================================

def clean_answer(answer):

    if not answer:

        return ""


    answer = answer.strip()


    prefixes = [

        "assistant:",

        "Assistant:",

        "assistant：",

        "Assistant：",

    ]


    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = (
                answer[
                    len(prefix):
                ]
                .strip()
            )


    stop_markers = [

        "\nuser",

        "\nUser",

        "\n用户",

        "\n### User",

        "\n<|im_start|>user",

        "<|im_start|>user",

    ]


    for marker in stop_markers:

        if marker in answer:

            answer = (
                answer
                .split(marker)[0]
                .strip()
            )


    for code in range(1, 5):

        answer = answer.replace(
            chr(code),
            "",
        )


    return answer.strip()


# ============================================================
# 13. valid rejected
# ============================================================

def valid_rejected(answer):

    if not answer:

        return False


    if len(answer) < 5:

        return False


    if answer.count("�") >= 2:

        return False


    bad_control_count = 0


    for ch in answer:

        code = ord(ch)

        if (
            code < 32
            and ch not in "\n\r\t"
        ):

            bad_control_count += 1


    if bad_control_count >= 2:

        return False


    if re.search(
        r"(.)\1{7,}",
        answer,
    ):

        return False


    return True


# ============================================================
# 14. output
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 15. generate DPO
# ============================================================

print()
print("=" * 80)
print("GENERATING DPO DATA")
print("=" * 80)


total = len(dataset)


if MAX_SAMPLES is not None:

    total = min(
        total,
        MAX_SAMPLES,
    )


print(
    "Total samples:",
    total,
)


written = 0
skipped = 0

start_all = time.time()


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:


    for index in range(total):

        item_start = time.time()


        messages = dataset[index]["messages"]


        question = get_user_question(
            messages
        )


        chosen = get_reference_answer(
            messages
        )


        chosen = chosen.strip()


        print()
        print("=" * 80)

        print(
            f"[{index + 1}/{total}] PROCESSING",
            flush=True,
        )

        print(
            "Question:",
            question[:300],
            flush=True,
        )


        # ====================================================
        # generate rejected
        # ====================================================

        rejected = generate_rejected(

            question,

            index + 1,

            total,

        )


        print()
        print("RAW REJECTED:")

        print(
            repr(rejected)
        )


        rejected = clean_answer(
            rejected
        )


        print()
        print("CLEANED REJECTED:")

        print(
            repr(rejected)
        )


        # ====================================================
        # validation
        # ====================================================

        if not question:

            print(
                "SKIP: empty question"
            )

            skipped += 1
            continue


        if not chosen:

            print(
                "SKIP: empty chosen"
            )

            skipped += 1
            continue


        if not valid_rejected(
            rejected
        ):

            print(
                "SKIP: invalid rejected"
            )

            skipped += 1
            continue


        if rejected == chosen:

            print(
                "SKIP: rejected == chosen"
            )

            skipped += 1
            continue


        # ====================================================
        # DPO item
        # ====================================================

        dpo_item = {

            "prompt": [

                {
                    "role": "user",
                    "content": question,
                }

            ],

            "chosen": [

                {
                    "role": "assistant",
                    "content": chosen,
                }

            ],

            "rejected": [

                {
                    "role": "assistant",
                    "content": rejected,
                }

            ],

        }


        f.write(
            json.dumps(
                dpo_item,
                ensure_ascii=False,
            )
            + "\n"
        )


        f.flush()


        written += 1


        # ====================================================
        # statistics
        # ====================================================

        elapsed = (
            time.time()
            - start_all
        )


        processed = index + 1


        avg_time = (
            elapsed
            / processed
        )


        remaining = (
            total
            - processed
        )


        eta = (
            avg_time
            * remaining
        )


        item_time = (
            time.time()
            - item_start
        )


        print()
        print(
            f"[{processed}/{total}] WRITTEN"
        )

        print(
            f"Progress : "
            f"{processed}/{total} "
            f"({processed / total * 100:.2f}%)"
        )

        print(
            f"Current  : "
            f"{format_seconds(item_time)}"
        )

        print(
            f"Average  : "
            f"{format_seconds(avg_time)} / sample"
        )

        print(
            f"Elapsed  : "
            f"{format_seconds(elapsed)}"
        )

        print(
            f"ETA      : "
            f"{format_seconds(eta)}"
        )

        print(
            f"Written  : {written}"
        )

        print(
            f"Skipped  : {skipped}"
        )

        print(
            f"GPU      : "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )


# ============================================================
# 16. DONE
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Input samples :", total)
print("Written       :", written)
print("Skipped       :", skipped)

print()
print("DPO dataset:")
print(OUTPUT_PATH)