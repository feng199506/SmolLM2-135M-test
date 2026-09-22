import json
from pathlib import Path


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILES = [
    BASE_DIR / "03_train_sft" / "data" / "qa_train_1100.jsonl",
]

OUTPUT_FILE = (
    BASE_DIR
    / "05_train_grpo"
    / "data"
    / "grpo_train.jsonl"
)


# ============================================================
# 2. 清理文本
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    return text.strip()


# ============================================================
# 3. 从 messages 中提取最后一个 user / assistant
# ============================================================

def extract_qa(messages):

    question = None
    answer = None

    for message in messages:

        role = message.get("role")
        content = clean_text(
            message.get("content", "")
        )

        if role == "user":
            question = content

        elif role == "assistant":
            answer = content

    return question, answer


# ============================================================
# 4. 转换单条数据
# ============================================================

def convert_item(item):

    if "messages" not in item:
        return None

    question, answer = extract_qa(
        item["messages"]
    )

    if not question:
        return None

    if not answer:
        return None

    return {
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "solution": answer,
    }


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("GRPO Dataset Generator")
    print("=" * 70)

    rows = []

    total = 0
    invalid = 0

    # ========================================================
    # 读取所有输入文件
    # ========================================================

    for input_file in INPUT_FILES:

        print()
        print(f"读取：{input_file}")

        if not input_file.exists():

            print(
                f"[WARNING] 文件不存在：{input_file}"
            )

            continue

        with open(
            input_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                total += 1

                try:

                    item = json.loads(line)

                except json.JSONDecodeError as e:

                    print(
                        f"[WARNING] "
                        f"{input_file.name}:{line_no} "
                        f"JSON错误：{e}"
                    )

                    invalid += 1

                    continue

                result = convert_item(item)

                if result is None:

                    invalid += 1

                    continue

                rows.append(result)

    # ========================================================
    # 去重
    # ========================================================

    unique_rows = []

    seen_questions = set()

    duplicate = 0

    for row in rows:

        question = row["prompt"][0]["content"]

        if question in seen_questions:

            duplicate += 1

            continue

        seen_questions.add(question)

        unique_rows.append(row)

    rows = unique_rows

    # ========================================================
    # 输出目录
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 写入
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # 统计
    # ========================================================

    print()
    print("=" * 70)
    print("生成完成")
    print("=" * 70)

    print(f"原始样本：{total}")
    print(f"无效样本：{invalid}")
    print(f"重复问题：{duplicate}")
    print(f"最终样本：{len(rows)}")

    print()
    print(f"输出文件：{OUTPUT_FILE}")

    # ========================================================
    # 显示样例
    # ========================================================

    if rows:

        print()
        print("=" * 70)
        print("第一条样本")
        print("=" * 70)

        print(
            json.dumps(
                rows[0],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()