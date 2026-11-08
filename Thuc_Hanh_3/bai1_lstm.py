import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data import prepare_sentiment
from engine import (Config, cls_metrics, env_info, evaluate_cls, get_device,
                    plot_confusion, plot_history, run_tfidf, save_json,
                    seed_all, show_cls_errors, train_model, training_summary)
from models import SentenceClassifier

ROOT = Path(__file__).resolve().parent


def train_and_evaluate(rnn_type, cfg, loaders, vocab, device, ckpt_path):
    # Dùng chung cho Bài 1, 2, 4: huấn luyện, rồi đo dev/test bằng checkpoint tốt nhất
    seed_all(cfg.seed)
    model = SentenceClassifier(len(vocab), rnn_type, cfg.emb_dim, cfg.hidden,
                               cfg.layers, cfg.dropout)
    history, best_epoch = train_model(model, loaders, cfg, device, "cls", ckpt_path)

    result = training_summary(model, history, best_epoch)
    result["history"] = history
    preds = {}
    for split in ("dev", "test"):
        _, true, pred = evaluate_cls(model, loaders[split], device)
        result[split] = cls_metrics(true, pred)
        preds[split] = (true, pred)
    return result, preds


def metrics_table(entries):
    # entries: {"tên mô hình": kết quả có khóa "test"}
    rows = []
    for name, res in entries.items():
        t = res["test"]
        row = {"model": name,
               "accuracy": t["accuracy"],
               "precision_macro": t["precision_macro"],
               "recall_macro": t["recall_macro"],
               "f1_macro": t["f1_macro"]}
        for label, value in t["f1_per_class"].items():
            row["f1_" + label] = value
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(entries):
    rows = []
    for name, res in entries.items():
        rows.append({"model": name,
                     "parameters": res["parameters"],
                     "best_epoch": res["best_epoch"],
                     "best_dev_f1": res["best_dev_score"],
                     "sec_per_epoch": round(res["seconds_per_epoch"], 1),
                     "test_f1_macro": res["test"]["f1_macro"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    device = get_device()
    out = ROOT / "results" / "bai1"
    out.mkdir(parents=True, exist_ok=True)

    loaders, vocab, data = prepare_sentiment(cfg.batch_size)
    result, preds = train_and_evaluate(
        "LSTM", cfg, loaders, vocab, device,
        ROOT / "checkpoints" / "bai1_lstm.pt")

    train_texts, train_y = data["train"]
    tfidf = run_tfidf(train_texts, train_y,
                      {"dev": data["dev"], "test": data["test"]})

    save_json({"config": asdict(cfg), "env": env_info(),
               "model": result, "tfidf": tfidf}, out / "results.json")
    plot_history({"LSTM": result["history"]}, out / "curves.png", "macro-F1")
    plot_confusion(*preds["dev"], out / "confusion_dev.png")
    plot_confusion(*preds["test"], out / "confusion_test.png")

    table = metrics_table({"LSTM": result, "TF-IDF": tfidf})
    table.to_csv(out / "test_metrics.csv", index=False)
    cost = cost_table({"LSTM": result})
    cost.to_csv(out / "cost.csv", index=False)

    print("\nKết quả trên test:")
    print(table.round(4).to_string(index=False))
    print("\nChi phí huấn luyện:")
    print(cost.round(4).to_string(index=False))
    print("\nMột số câu dự đoán sai trên dev:")
    show_cls_errors(data["dev"][0], *preds["dev"], limit=10)


if __name__ == "__main__":
    main()