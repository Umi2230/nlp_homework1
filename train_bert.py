"""微調 google-bert/bert-base-chinese，建立中文八類情緒分類模型。
讀取訓練與驗證 JSONL，使用動態補齊與梯度累積進行訓練。
以驗證損失選擇最佳模型，並儲存模型、Tokenizer 與訓練紀錄。"""


import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)



# 使用的中文 BERT 預訓練模型。
MODEL_NAME = "google-bert/bert-base-chinese"

# prepare_data.py 產生的 JSONL 與標籤對照表位置。
DATA_DIR = Path("/content/homework1/data/bert")

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

# 最佳模型、訓練紀錄與 checkpoint 的儲存位置。
OUTPUT_DIR = Path("/content/homework1/output/bert_emotion_clf")




# 沿用資料準備階段的標籤編號，避免類別對應不一致。
with (DATA_DIR / "label_mapping.json").open(
    "r", encoding="utf-8"
) as file:
    mapping = json.load(file)

LABEL2ID = mapping["label2id"]

# JSON 物件的鍵為字串，讀回後轉為整數類別編號。
ID2LABEL = {
    int(index): emotion
    for index, emotion in mapping["id2label"].items()
}

# 由標籤對照表決定分類數，目前資料為八類。
NUM_LABELS = len(LABEL2ID)

# 輸入序列最長 512 個 Token，超出部分截斷。
MAX_SEQ_LENGTH = 512



# 固定訓練亂數種子，協助重現實驗。
SEED = 42

# 訓練最多三輪；可能因 Early Stopping 提前結束。
NUM_TRAIN_EPOCHS = 3

# 每個裝置每批訓練八筆資料。
PER_DEVICE_TRAIN_BATCH_SIZE = 8

# 每個裝置每批驗證八筆資料。
PER_DEVICE_EVAL_BATCH_SIZE = 8

# 累積兩批梯度後更新；單 GPU 的有效批量通常為 8 × 2 = 16。
GRADIENT_ACCUMULATION_STEPS = 2

# 微調學習率為 2e-5，與梯度累積步數分別設定。
LEARNING_RATE = 2e-5

# 前 50 個參數更新步數逐步提高學習率。
WARMUP_STEPS = 50

# 以權重衰減降低過度擬合的風險。
WEIGHT_DECAY = 0.01

# 暖身結束後使用線性學習率衰減。
LR_SCHEDULER_TYPE = "linear"

# 每 30 個參數更新步數進行驗證。
EVAL_STEPS = 30

# 與驗證頻率一致，每 30 個更新步數儲存 checkpoint。
SAVE_STEPS = 30

# 限制保留的 checkpoint 數量。
SAVE_TOTAL_LIMIT = 2

# 驗證損失連續三次未改善時提前停止。
EARLY_STOPPING_PATIENCE = 3

# 每 10 個更新步數記錄訓練指標。
LOGGING_STEPS = 10



# 啟用 FP16 混合精度；執行時還會確認 CUDA 可用。
USE_FP16 = True

# 本次設定不啟用 BF16。
USE_BF16 = False



def print_gpu_info():
    """顯示套件版本、CUDA 可用狀態及 GPU 資訊。"""

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(
            "CUDA capability："
            f"{torch.cuda.get_device_capability(0)}"
        )
        print(f"PyTorch CUDA：{torch.version.cuda}")

    print("=" * 72)


def compute_metrics(eval_pred):
    """將各類別的 logits 取 argmax，得到預測情緒編號。
    與正確標籤比較，計算正確率及各類別等權平均的 Macro F1。"""

    labels = eval_pred.label_ids

    predictions = eval_pred.predictions

    # 若模型回傳 tuple，取出第一個 logits 陣列。
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 從每筆資料的各類別分數中取出最高分的類別編號。
    preds = np.argmax(
        predictions,
        axis=-1,
    )

    # 各類別 F1 等權平均；無法計算的項目設為 0。
    f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    # 比較預測編號與正確編號，計算正確率。
    accuracy = float(np.mean(preds == labels))

    return {
        "accuracy": accuracy,
        "f1": f1,
    }



def main():


    print_gpu_info()


    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE

    if not train_path.is_file():
        raise FileNotFoundError(
            f"找不到 training data：{train_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    if not valid_path.is_file():
        raise FileNotFoundError(
            f"找不到 validation data：{valid_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # 讀取固定的訓練集與驗證集。
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    print(dataset)


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    def tokenize_batch(batch):
        """將句子轉為 Token，截斷超長內容；補齊由批次資料整理器處理。"""

        return tokenizer(
            batch["sentences"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,

        # 完成 Tokenization 後，移除不需傳入模型的原始句子。
        remove_columns=["sentences"],
    )


    # 每批只補齊到該批最長序列，減少多餘計算。
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )


    # 建立情緒分類模型；logits 形狀為 [批次筆數, 類別數]。
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


    # 只有 CUDA 可用且設定開啟時才使用 FP16。
    fp16_enabled = (
        USE_FP16
        and torch.cuda.is_available()
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        weight_decay=WEIGHT_DECAY,

        lr_scheduler_type=LR_SCHEDULER_TYPE,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,

        # 訓練結束時載回驗證損失最低的 checkpoint。
        load_best_model_at_end=True,

        # 使用驗證損失選擇最佳模型。
        metric_for_best_model="eval_loss",

        # 驗證損失越低越好。
        greater_is_better=False,

        fp16=fp16_enabled,
        bf16=USE_BF16,
        tf32=False,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["validation"]
        ),

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        # 依驗證損失是否持續改善，決定是否提前停止。
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )


    # 開始訓練；此呼叫未指定從 checkpoint 接續。
    trainer.train()


    print(
        "最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        "最佳 eval_loss："
        f"{trainer.state.best_metric}"
    )


    # 將已載回的最佳模型儲存到輸出資料夾。
    trainer.save_model(
        str(OUTPUT_DIR)
    )

    # 一併儲存 Tokenizer，供後續推論載入。
    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # 儲存訓練紀錄與最佳 checkpoint 資訊，供繪圖程式使用。
    trainer.save_state()

    print(
        "Final model directory："
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
