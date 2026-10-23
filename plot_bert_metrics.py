"""讀取模型資料夾中的 trainer_state.json，繪製訓練與驗證曲線。
將 Loss 存為 loss_curve.png；有 eval_f1 紀錄時另存 f1_curve.png。
圖片存於 OUTPUT_DIR，使用 Agg 後端，適用於無桌面的 Colab 環境。"""

from pathlib import Path
import json

import matplotlib
# 使用不需要桌面視窗的繪圖後端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt



# 訓練紀錄來源與 PNG 圖片的儲存資料夾。
OUTPUT_DIR = Path("/content/homework1/output/bert_emotion_clf")

STATE_FILE = OUTPUT_DIR / "trainer_state.json"

LOSS_PLOT_FILE = OUTPUT_DIR / "loss_curve.png"

F1_PLOT_FILE = OUTPUT_DIR / "f1_curve.png"



def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{STATE_FILE.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    # 讀取 Trainer 儲存的訓練狀態。
    with STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(file)

    logs = state.get(
        "log_history",
        [],
    )

    if not logs:
        raise RuntimeError(
            "trainer_state.json 中沒有 log_history。"
        )


    # 擷取各更新步數的訓練損失。
    train_loss = [
        (item["step"], item["loss"])
        for item in logs
        if "step" in item and "loss" in item
    ]

    # 擷取各驗證步數的驗證損失。
    valid_loss = [
        (item["step"], item["eval_loss"])
        for item in logs
        if "step" in item and "eval_loss" in item
    ]

    # 有損失紀錄時繪製曲線，並儲存為 loss_curve.png。
    if train_loss or valid_loss:

        plt.figure(figsize=(10, 6))

        if train_loss:
            x, y = zip(*train_loss)
            plt.plot(
                x,
                y,
                label="Training Loss",
                linewidth=1.8,
            )

        if valid_loss:
            x, y = zip(*valid_loss)
            plt.plot(
                x,
                y,
                marker="o",
                label="Validation Loss",
                linewidth=1.8,
            )

        plt.xlabel("Optimizer Step")
        plt.ylabel("Loss")
        plt.title("BERT Training and Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            LOSS_PLOT_FILE,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Loss 圖表："
            f"{LOSS_PLOT_FILE.resolve()}"
        )


    # 擷取驗證 Macro F1，對應 train_bert.py 的 f1 指標。
    eval_f1 = [
        (item["step"], item["eval_f1"])
        for item in logs
        if "step" in item and "eval_f1" in item
    ]

    # 有 F1 紀錄時繪圖，並儲存為 f1_curve.png。
    if eval_f1:

        plt.figure(figsize=(10, 6))

        x, y = zip(*eval_f1)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label="Validation F1",
        )

        plt.xlabel("Optimizer Step")
        plt.ylabel("F1")
        plt.title("BERT Validation F1")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            F1_PLOT_FILE,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "F1 圖表："
            f"{F1_PLOT_FILE.resolve()}"
        )

    print(
        f"Training loss points：{len(train_loss)}"
    )
    print(
        f"Validation loss points：{len(valid_loss)}"
    )
    print(
        f"Validation F1 points：{len(eval_f1)}"
    )


if __name__ == "__main__":
    main()
