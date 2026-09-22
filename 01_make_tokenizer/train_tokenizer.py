from pathlib import Path
import json

from tokenizers import (
    Tokenizer,
    AddedToken,
)

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import (
    ByteLevel,
    Digits,
    Sequence,
)

from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from transformers import GPT2TokenizerFast


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_DIR = BASE_DIR / "corpus"
OUTPUT_DIR = BASE_DIR / "tokenizer"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 固定 Special Tokens
# ============================================================

SPECIAL_TOKENS = [
    "<|endoftext|>",       # 0
    "<|im_start|>",        # 1
    "<|im_end|>",          # 2
    "<repo_name>",         # 3
    "<reponame>",          # 4
    "<file_sep>",          # 5
    "<filename>",          # 6
    "<gh_stars>",          # 7
    "<issue_start>",       # 8
    "<issue_comment>",     # 9
    "<issue_closed>",      # 10
    "<jupyter_start>",     # 11
    "<jupyter_text>",      # 12
    "<jupyter_code>",      # 13
    "<jupyter_output>",    # 14
    "<jupyter_script>",    # 15
    "<empty_output>",      # 16
]

EXPECTED_SPECIAL_IDS = {
    token: i
    for i, token in enumerate(SPECIAL_TOKENS)
}


# ============================================================
# 3. 目标 vocab
# ============================================================

TARGET_VOCAB_SIZE = 49152

print("=" * 70)
print("Target vocab size:", TARGET_VOCAB_SIZE)
print("Special tokens:", len(SPECIAL_TOKENS))
print("=" * 70)


# ============================================================
# 4. 找语料
# ============================================================

corpus_files = []

for p in CORPUS_DIR.rglob("*"):
    if p.is_file():
        corpus_files.append(str(p))

if not corpus_files:
    raise RuntimeError(
        f"没有找到语料文件：{CORPUS_DIR}"
    )

print(f"Found {len(corpus_files)} corpus files")


# ============================================================
# 5. 创建 BPE Tokenizer
# ============================================================

tokenizer = Tokenizer(
    BPE(
        vocab={},
        merges=[],
        fuse_unk=False,
        byte_fallback=False,
    )
)


# ============================================================
# 6. PreTokenizer
# ============================================================

tokenizer.pre_tokenizer = Sequence(
    [
        Digits(
            individual_digits=True
        ),
        ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=True,
        ),
    ]
)


# ============================================================
# 7. Decoder
# ============================================================

tokenizer.decoder = ByteLevelDecoder(
    add_prefix_space=True,
    trim_offsets=True,
    use_regex=True,
)


# ============================================================
# 8. Trainer
# ============================================================

trainer = BpeTrainer(
    vocab_size=TARGET_VOCAB_SIZE,
    special_tokens=[
        AddedToken(
            token,
            normalized=False,
            special=True,
            single_word=False,
            lstrip=False,
            rstrip=False,
        )
        for token in SPECIAL_TOKENS
    ],
    min_frequency=2,
    show_progress=True,
)


# ============================================================
# 9. 训练
# ============================================================

print()
print("=" * 70)
print("Training tokenizer...")
print("=" * 70)

tokenizer.train(
    files=corpus_files,
    trainer=trainer,
)


# ============================================================
# 10. 检查 Special Token ID
# ============================================================

print()
print("=" * 70)
print("Checking special token IDs")
print("=" * 70)

for token, expected_id in EXPECTED_SPECIAL_IDS.items():

    actual_id = tokenizer.token_to_id(token)

    print(
        f"{actual_id:5d}  {token}"
    )

    if actual_id != expected_id:
        raise RuntimeError(
            f"Special token ID 错误："
            f"{token}: actual={actual_id}, "
            f"expected={expected_id}"
        )


# ============================================================
# 11. 检查 vocab size
# ============================================================

actual_vocab_size = tokenizer.get_vocab_size()

print()
print(f"Final vocab size: {actual_vocab_size}")


# ============================================================
# 12. 检查所有 special token
# ============================================================

vocab = tokenizer.get_vocab()

print()
print("=" * 70)
print("Special token verification")
print("=" * 70)

for token, expected_id in EXPECTED_SPECIAL_IDS.items():

    if token not in vocab:
        raise RuntimeError(
            f"Missing special token: {token}"
        )

    if vocab[token] != expected_id:
        raise RuntimeError(
            f"Wrong ID for {token}: "
            f"{vocab[token]} != {expected_id}"
        )


# ============================================================
# 13. 检查普通 vocab 是否从 17 开始
# ============================================================

normal_items = [
    (token, idx)
    for token, idx in vocab.items()
    if token not in EXPECTED_SPECIAL_IDS
]

if not normal_items:
    raise RuntimeError(
        "没有普通 vocab"
    )

min_normal_id = min(
    idx for _, idx in normal_items
)

max_normal_id = max(
    idx for _, idx in normal_items
)

print()
print("Normal vocab ID range:")
print(
    min_normal_id,
    "->",
    max_normal_id
)

if min_normal_id != 17:
    raise RuntimeError(
        f"普通 vocab 没有从 17 开始，"
        f"实际为 {min_normal_id}"
    )


# ============================================================
# 14. 保存 tokenizer.json
# ============================================================

TOKENIZER_JSON = OUTPUT_DIR / "tokenizer.json"

tokenizer.save(
    str(TOKENIZER_JSON)
)

print()
print("Saved:")
print(TOKENIZER_JSON)


# ============================================================
# 14.1 检查 tokenizer.json 的 post_processor
# ============================================================

with open(
    TOKENIZER_JSON,
    "r",
    encoding="utf-8",
) as f:

    tokenizer_json_data = json.load(f)


post_processor = tokenizer_json_data.get(
    "post_processor"
)

print()
print("post_processor:", post_processor)

if post_processor is not None:
    raise RuntimeError(
        "tokenizer.json 的 post_processor 必须为 null"
    )


# ============================================================
# 14.2 保存 vocab.json
# ============================================================

VOCAB_JSON = OUTPUT_DIR / "vocab.json"

with open(
    VOCAB_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        vocab,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("Saved:")
print(VOCAB_JSON)


# ============================================================
# 14.3 保存 merges.txt
# ============================================================

MERGES_TXT = OUTPUT_DIR / "merges.txt"

merges = tokenizer_json_data["model"]["merges"]

with open(
    MERGES_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write("#version: 0.2\n")

    for merge in merges:

        # 新版本 tokenizer.json 可能是：
        # ["a", "b"]
        #
        # 也可能是：
        # "a b"

        if isinstance(merge, list):
            f.write(" ".join(merge) + "\n")
        else:
            f.write(merge + "\n")

print()
print("Saved:")
print(MERGES_TXT)

print()
print("Number of BPE merges:", len(merges))

# ============================================================
# 15. 生成 tokenizer_config.json
# ============================================================

tokenizer_config = {
    "add_prefix_space": False,

    "added_tokens_decoder": {
        str(i): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for i, token in enumerate(SPECIAL_TOKENS)
    },

    "additional_special_tokens": SPECIAL_TOKENS,

    "bos_token": "<|endoftext|>",

    "clean_up_tokenization_spaces": False,

    "eos_token": "<|endoftext|>",

    "model_max_length": 8192,

    "tokenizer_class": "GPT2Tokenizer",

    "unk_token": "<|endoftext|>",

    "vocab_size": actual_vocab_size,
}


TOKENIZER_CONFIG_JSON = (
    OUTPUT_DIR / "tokenizer_config.json"
)

with open(
    TOKENIZER_CONFIG_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        tokenizer_config,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("Saved:")
print(TOKENIZER_CONFIG_JSON)


# ============================================================
# 16. 保存 special_tokens_map.json
# ============================================================

special_tokens_map = {
    "additional_special_tokens": SPECIAL_TOKENS,

    "bos_token": {
        "content": "<|endoftext|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },

    "eos_token": {
        "content": "<|endoftext|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },

    "unk_token": {
        "content": "<|endoftext|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}


SPECIAL_MAP_JSON = (
    OUTPUT_DIR / "special_tokens_map.json"
)

with open(
    SPECIAL_MAP_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        special_tokens_map,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("Saved:")
print(SPECIAL_MAP_JSON)

# ============================================================
# 17. 用 Transformers GPT2TokenizerFast 加载测试
# ============================================================

print()
print("=" * 70)
print("Testing GPT2TokenizerFast")
print("=" * 70)

hf_tokenizer = GPT2TokenizerFast(
    tokenizer_file=str(TOKENIZER_JSON),

    bos_token="<|endoftext|>",

    eos_token="<|endoftext|>",

    unk_token="<|endoftext|>",

    additional_special_tokens=[
        token
        for token in SPECIAL_TOKENS
        if token != "<|endoftext|>"
    ],
)

hf_tokenizer.model_max_length = 8192


# ============================================================
# 18. 最终检查
# ============================================================

print()
print("=" * 70)
print("Final verification")
print("=" * 70)

print(
    "len(tokenizer):",
    len(hf_tokenizer)
)

print(
    "vocab_size:",
    hf_tokenizer.vocab_size
)

print(
    "bos_token:",
    hf_tokenizer.bos_token,
    hf_tokenizer.bos_token_id,
)

print(
    "eos_token:",
    hf_tokenizer.eos_token,
    hf_tokenizer.eos_token_id,
)

print(
    "unk_token:",
    hf_tokenizer.unk_token,
    hf_tokenizer.unk_token_id,
)


# ============================================================
# 18.1 检查 HF Special Token ID
# ============================================================

for token, expected_id in EXPECTED_SPECIAL_IDS.items():

    actual_id = (
        hf_tokenizer.convert_tokens_to_ids(token)
    )

    if actual_id != expected_id:
        raise RuntimeError(
            f"HF tokenizer ID 错误："
            f"{token}: "
            f"{actual_id} != {expected_id}"
        )


# ============================================================
# 18.2 检查 HF vocab size
# ============================================================

if len(hf_tokenizer) != actual_vocab_size:
    raise RuntimeError(
        f"HF tokenizer vocab size 错误："
        f"{len(hf_tokenizer)} != "
        f"{actual_vocab_size}"
    )


# ============================================================
# 18.3 检查 vocab.json
# ============================================================

with open(
    VOCAB_JSON,
    "r",
    encoding="utf-8",
) as f:

    saved_vocab = json.load(f)


if len(saved_vocab) != actual_vocab_size:
    raise RuntimeError(
        f"vocab.json 数量错误："
        f"{len(saved_vocab)} != "
        f"{actual_vocab_size}"
    )


for token, expected_id in EXPECTED_SPECIAL_IDS.items():

    if saved_vocab.get(token) != expected_id:
        raise RuntimeError(
            f"vocab.json 中 {token} ID 错误："
            f"{saved_vocab.get(token)} != "
            f"{expected_id}"
        )


# ============================================================
# 19. 编解码测试
# ============================================================

test_texts = [
    "Hello world!",
    "你好，世界！",
    "Python def hello():",
    "1234567890",
    "<|im_start|>user",
    "<|im_end|>",
]


print()
print("=" * 70)
print("Encode / Decode test")
print("=" * 70)


for text in test_texts:

    ids = hf_tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = hf_tokenizer.decode(
        ids,
        skip_special_tokens=False,
    )

    print()
    print("TEXT:")
    print(text)

    print("IDS:")
    print(ids)

    print("DECODE:")
    print(decoded)


# ============================================================
# 20. 最终输出
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("Tokenizer directory:")
print(OUTPUT_DIR)

print()
print("Files:")

print("  tokenizer.json")
print("  vocab.json")
print("  merges.txt")
print("  tokenizer_config.json")
print("  special_tokens_map.json")

print()
print("Final vocab size:", actual_vocab_size)
print("Final merges:", len(merges))