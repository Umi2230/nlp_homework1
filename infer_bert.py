"""載入微調後的 BERT 情緒分類模型與 Tokenizer。
預測 TEST_TEXTS 中各句子的情緒，輸出類別名稱及信心分數。
情緒名稱由模型設定中的標籤對照取得。"""

from pathlib import Path
from pprint import pprint

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)



# 載入訓練程式匯出的最佳模型資料夾。
MODEL_DIR = Path(
    "/content/homework1/output/bert_emotion_clf"
)

# 待預測句子，可依測試需求調整。
TEST_TEXTS = [
    "我每天都能跟朋友見面，我好開心！",
    "最好的朋友要去澳洲一年，好捨不得",
    "好累，明天還要上課",
    "剛剛收到工作錄取的通知書！",
    "今年的秋天好像比較晚來。",
    "你還好嗎?",
    "好噁心我要吐了",
    "我不是說不要了嗎?你怎麼講不聽!"
]



def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )


    # 使用與微調模型相同的 Tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )


    # 載入模型權重及情緒標籤對照。
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )


    # CUDA 可用時使用第一張 GPU，否則使用 CPU。
    device = (
        0
        if torch.cuda.is_available()
        else -1
    )

    print(
        "Inference device："
        + (
            torch.cuda.get_device_name(0)
            if device == 0
            else "CPU"
        )
    )


    # 建立情緒分類 Pipeline。
    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


    # 批次預測句子，並截斷超過最大長度的內容。
    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=512,
    )


    # 逐句輸出情緒與信心分數；簡式輸出保留四位小數。
    for text, result in zip(
        TEST_TEXTS,
        results,
    ):
        print(
            f"{text} => {result['label']} "
            f"({result['score']:.4f})"
        )

        print("=" * 72)
        print(f"文字：{text}")
        print(
            f"預測：{result['label']}"
        )
        print(
            f"信心分數：{result['score']:.6f}"
        )

    print("=" * 72)

    print("Raw pipeline output：")
    # 顯示原始預測結果，保留完整的分數精度。
    pprint(results)


if __name__ == "__main__":
    main()
