# Model_02_test

Small-scale LLM training pipeline: **Tokenizer → Pretrain (LM) → SFT → DPO → GRPO**.

Architecture config uses `LlamaForCausalLM` (compact size; see `configs/config.json`). Chat format uses `<|im_start|>` / `<|im_end|>`.

中文说明见 [README_zh.md](README_zh.md).

---

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (recommended)
- Dependencies: see `requirements.txt`

```bash
# Example (adjust CUDA index for your PyTorch build)
pip install -r requirements.txt
```

> Note: `torch==...+cu126` style wheels usually need the official PyTorch CUDA index, not plain PyPI.

---

## Project Layout

```text
Model_02_test/
├── corpus/                         # Pretrain text (.txt)
├── configs/                        # Model config (Llama-style)
├── tokenizer/                      # Trained tokenizer output
├── sentence_transformer_configs/   # Local embedding model for GRPO rewards
├── 01_make_tokenizer/              # Train BPE tokenizer
├── 02_train_lang_model/            # Causal LM from scratch
├── 03_train_sft/                   # Supervised fine-tuning (QA)
├── 04_train_dpo/                   # Preference data + DPO
├── 05_train_grpo/                  # GRPO with custom rewards
├── lm_output/                      # Pretrain checkpoints (+ final/)
├── sft_output/                     # SFT model
├── dpo_output/                     # DPO model
└── grpo_output/                    # GRPO model
```

Large model folders (`lm_output`, `sft_output`, `dpo_output`, `grpo_output`, `tokenizer`, `*.safetensors`) are gitignored.

---

## Pipeline (run in order)

Paths in training scripts are resolved via `Path(__file__)`, so you can run from the repo root.

### 1) Tokenizer

Put plain-text corpora under `corpus/`, then:

```bash
python 01_make_tokenizer/train_tokenizer.py
```

Output: `tokenizer/` (`tokenizer.json`, `vocab.json`, `merges.txt`, configs).

Target vocab size in code is `49152`; actual size depends on corpus richness (`min_frequency=2`).

### 2) Language model pretraining

```bash
python 02_train_lang_model/train01.py
```

- Loads `tokenizer/` + `configs/config.json`
- Packs corpus into length-`2048` blocks
- Saves to `lm_output/final/`

Quick smoke test:

```bash
python 02_train_lang_model/test.py
```

> **Important:** Pretrain quality dominates everything downstream. A tiny `corpus/` only yields a few optimizer steps and a weak base model.

### 3) SFT (instruction / QA)

Prepare chat JSONL under `03_train_sft/data/` (field `messages` with `user` / `assistant`), e.g. generate with:

```bash
python 03_train_sft/gen_data2.py
# or
python 03_train_sft/gen_data.py
```

Train:

```bash
python 03_train_sft/train_qa_sft.py
```

- Base: `lm_output/final`
- Output: `sft_output/`
- Uses TRL `SFTTrainer` with `assistant_only_loss=True` and a chat template that marks assistant spans with `{% generation %}`

Test:

```bash
python 03_train_sft/test_qa.py
python 03_train_sft/test_qa_stream.py
```

### 4) DPO

Generate preference data from SFT model + SFT QA data:

```bash
python 04_train_dpo/gen_dpo_data.py
```

- Writes `04_train_dpo/data/dpo_train.jsonl`
- Columns: `prompt` / `chosen` / `rejected` (chat lists)
- Rejected answers are **sampled** (temperature / top-p) and filtered; weak bases may skip many rows

Train:

```bash
python 04_train_dpo/train_qa_dpo.py
```

- Base: `sft_output`
- Output: `dpo_output/`

### 5) GRPO

Build GRPO dataset from SFT JSONL:

```bash
python 05_train_grpo/gen_grpo_data.py
```

- Writes `05_train_grpo/data/grpo_train.jsonl`
- Fields: `prompt` (chat list), `solution` (reference answer)

Train:

```bash
python 05_train_grpo/train_grpo.py
```

- Default base: `sft_output` (switch to `dpo_output` in the script if desired)
- Needs local `sentence_transformer_configs/` for semantic reward
- Combined reward: ROUGE-L + semantic similarity + keyword overlap
- Output: `grpo_output/`

---

## Data Formats

**SFT** (`*.jsonl`):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO** (`*.jsonl`):

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

**GRPO** (`*.jsonl`):

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "solution": "..."
}
```

---

## Tips

1. Expand `corpus/` substantially before serious pretraining; then retrain tokenizer and LM if needed.
2. Keep tokenizer vocab size aligned with `config.vocab_size` / model embeddings.
3. Prefer testing chat models under `sft_output` / `dpo_output` / `grpo_output` (they carry `chat_template`); `lm_output/final` is a raw LM.
4. Set `CUDA_VISIBLE_DEVICES` inside scripts (default `"0"`).
5. After improving the base/SFT model, regenerate DPO data before DPO training.

---

## License / Notes

This repo is a learning / experiment pipeline. Model weights and large artifacts are not meant to be committed.
