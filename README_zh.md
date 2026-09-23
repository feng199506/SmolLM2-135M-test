# Model_02_test

小型大语言模型训练流水线：**Tokenizer → 预训练 (LM) → SFT → DPO → GRPO**。

模型配置为 `LlamaForCausalLM` 小规模结构（见 `configs/config.json`）。对话格式使用 `<|im_start|>` / `<|im_end|>`。

English docs: [README.md](README.md).

---

## 环境要求

- Python 3.10+
- NVIDIA GPU + CUDA（推荐）
- 依赖见 `requirements.txt`

```bash
# 示例（PyTorch CUDA 版通常需配合官方 index）
pip install -r requirements.txt
```

> 注意：`torch==...+cu126` 这类 wheel 一般不能直接从普通 PyPI 安装，需使用 PyTorch 官方 CUDA 源。

---

## 目录结构

```text
Model_02_test/
├── corpus/                         # 预训练语料（.txt）
├── configs/                        # 模型结构配置（Llama 风格）
├── tokenizer/                      # 训练得到的分词器
├── sentence_transformer_configs/   # GRPO 语义奖励用的本地向量模型
├── 01_make_tokenizer/              # 训练 BPE Tokenizer
├── 02_train_lang_model/            # 从零训练因果语言模型
├── 03_train_sft/                   # 监督微调（问答）
├── 04_train_dpo/                   # 偏好数据生成 + DPO
├── 05_train_grpo/                  # GRPO（自定义奖励）
├── lm_output/                      # 预训练产物（含 final/）
├── sft_output/                     # SFT 模型
├── dpo_output/                     # DPO 模型
└── grpo_output/                    # GRPO 模型
```

大体积产物（`lm_output`、`sft_output`、`dpo_output`、`grpo_output`、`tokenizer`、`*.safetensors`）已在 `.gitignore` 中忽略。

---

## 训练流程（按顺序）

训练脚本路径基于 `Path(__file__)` 解析，可在仓库根目录直接运行。

### 1）训练 Tokenizer

将纯文本语料放入 `corpus/`，然后：

```bash
python 01_make_tokenizer/train_tokenizer.py
```

输出目录：`tokenizer/`（含 `tokenizer.json`、`vocab.json`、`merges.txt` 等）。

代码中目标词表大小为 `49152`；实际大小取决于语料规模与 `min_frequency=2`。

### 2）语言模型预训练

```bash
python 02_train_lang_model/train01.py
```

- 加载 `tokenizer/` 与 `configs/config.json`
- 将语料 pack 成长度 `2048` 的训练块
- 保存到 `lm_output/final/`

快速测试：

```bash
python 02_train_lang_model/test.py
```

> **重要：** 预训练质量决定后续所有阶段上限。`corpus/` 过小时，有效训练步数极少，底座会很弱。

### 3）SFT（指令 / 问答微调）

在 `03_train_sft/data/` 准备 chat 格式 JSONL（字段 `messages`，含 `user` / `assistant`），可用：

```bash
python 03_train_sft/gen_data2.py
# 或
python 03_train_sft/gen_data.py
```

训练：

```bash
python 03_train_sft/train_qa_sft.py
```

- 底座：`lm_output/final`
- 输出：`sft_output/`
- 使用 TRL `SFTTrainer`，开启 `assistant_only_loss=True`，chat template 用 `{% generation %}` 标记助手回答区间

测试：

```bash
python 03_train_sft/test_qa.py
python 03_train_sft/test_qa_stream.py
```

### 4）DPO

用 SFT 模型 + SFT 问答数据生成偏好对：

```bash
python 04_train_dpo/gen_dpo_data.py
```

- 输出：`04_train_dpo/data/dpo_train.jsonl`
- 字段：`prompt` / `chosen` / `rejected`（对话列表）
- rejected 使用**采样生成**（temperature / top-p）并做质量过滤；底座较弱时可能大量样本被跳过

训练：

```bash
python 04_train_dpo/train_qa_dpo.py
```

- 底座：`sft_output`
- 输出：`dpo_output/`

### 5）GRPO

从 SFT JSONL 生成 GRPO 数据：

```bash
python 05_train_grpo/gen_grpo_data.py
```

- 输出：`05_train_grpo/data/grpo_train.jsonl`
- 字段：`prompt`（对话列表）、`solution`（标准答案）

训练：

```bash
python 05_train_grpo/train_grpo.py
```

- 默认底座：`sft_output`（可在脚本中改为 `dpo_output`）
- 需要本地 `sentence_transformer_configs/` 计算语义相似度奖励
- 综合奖励：ROUGE-L + 语义相似度 + 关键词覆盖
- 输出：`grpo_output/`

---

## 数据格式

**SFT**（`*.jsonl`）：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO**（`*.jsonl`）：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

**GRPO**（`*.jsonl`）：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "solution": "..."
}
```

---

## 使用建议

1. 正式训练前请大幅扩充 `corpus/`，必要时重新训练 tokenizer 与预训练模型。
2. 保持 tokenizer 词表大小与 `config.vocab_size` / embedding 一致。
3. 对话能力测试请使用 `sft_output` / `dpo_output` / `grpo_output`（带 `chat_template`）；`lm_output/final` 是裸语言模型。
4. 脚本内默认 `CUDA_VISIBLE_DEVICES="0"`，可按机器修改。
5. 底座或 SFT 提升后，请重新生成 DPO 数据再训 DPO。

---

## 说明

本仓库用于学习与实验。模型权重及大型产物默认不纳入版本库。
