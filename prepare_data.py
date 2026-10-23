"""讀取中文情緒 CSV，將情緒名稱轉為整數標籤。
儲存標籤對照表，並以分層切分產生訓練與驗證 JSONL。"""


import pandas as pd

import json

from collections import Counter

from pathlib import Path

from sklearn.model_selection import train_test_split



# 原始情緒資料 CSV 路徑。
INPUT_FILE = Path("/content/homework1/data.csv")

# 訓練資料、驗證資料及標籤對照表的輸出位置。
OUTPUT_DIR = Path("/content/homework1/data/bert")

# 保留 20% 資料作為驗證集。
VALID_RATIO = 0.2

# 固定切分亂數，方便重現資料分組。
SEED = 42



def load_reviews(path: Path):
    """讀取 text、emotion 欄位，略過空白句子並檢查情緒標籤。
    儲存情緒編號對照表，回傳句子列表與整數標籤列表。"""

    texts = []
    labels = []

    # 記錄略過的空白句子資料列編號。
    skipped_empty_rows = []

    # 保留空欄位為空字串，供後續檢查。
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

    # 確認 CSV 包含句子與情緒欄位。
    required_columns = {"text", "emotion"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV 必須包含 text 和 emotion，"
            f"實際欄位：{df.columns.tolist()}"
        )


    # 清除情緒名稱前後空白。
    df["emotion"] = df["emotion"].astype(str).str.strip()

    # 情緒標籤不得為空白。
    if (df["emotion"] == "").any():
        raise ValueError("資料中有空白的 emotion，請先檢查 CSV。")

    # 依實際情緒名稱排序，建立從 0 開始的整數編號。
    emotion_names = sorted(df["emotion"].unique().tolist())

    label2id = {
        emotion: index
        for index, emotion in enumerate(emotion_names)
    }

    print("情緒標籤對照：", label2id)

    # 逐筆讀取資料；編號包含一列標題，並非含跨行欄位時的實體行號。
    for line_number, (_, row) in enumerate(df.iterrows(), start=2):

        text = str(row["text"]).strip()
    

        # 空白句子沒有可學習的內容，記錄後略過。
        if not text:
            skipped_empty_rows.append(line_number)
            continue
    
        emotion = row["emotion"]

        # 使用同一份對照表將情緒名稱轉為整數。
        label = label2id[emotion]

        texts.append(text)
        labels.append(label)

    if not texts:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    if skipped_empty_rows:
        print(
            "已略過空評論："
            f"{len(skipped_empty_rows)} 筆；"
            f"原始行號={skipped_empty_rows}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 同時儲存名稱到編號、編號到名稱，供訓練與推論使用。
    mapping = {
        "label2id": label2id,
        "id2label": {
            index: emotion
            for emotion, index in label2id.items()
        },
    }

    mapping_path = OUTPUT_DIR / "label_mapping.json"

    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)

    print(f"標籤對照表：{mapping_path}")

    return texts, labels


def save_jsonl(texts, labels, path: Path):
    """將句子與整數標籤逐筆寫入 JSONL，欄位為 sentences、labels。"""

    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):

            row = {
                "sentences": text,
                "labels": label,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_label_distribution(name, labels):
    """顯示各情緒編號的資料筆數及比例。"""

    counts = Counter(labels)
    total = len(labels)

    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total

        print(
            f"  label={label}: "
            f"{count} 筆 "
            f"({ratio:.2%})"
        )



def main():


    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_FILE.resolve()}"
        )


    # 讀取情緒 CSV，取得句子與整數標籤。
    texts, labels = load_reviews(INPUT_FILE)

    print("=" * 72)
    print_label_distribution("全部資料", labels)


    (
        # 以情緒標籤分層切分，維持訓練與驗證集的類別比例。
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"


    save_jsonl(
        # 以情緒標籤分層切分，維持訓練與驗證集的類別比例。
        train_texts,
        train_labels,
        train_path,
    )

    save_jsonl(
        valid_texts,
        valid_labels,
        valid_path,
    )


    print_label_distribution(
        "Training",
        train_labels,
    )

    print_label_distribution(
        "Validation",
        valid_labels,
    )

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
