import argparse
import json
from dataclasses import asdict

from bai1_lstm import (ROOT, cost_table, metrics_table,
                                 train_and_evaluate)
from data import prepare_sentiment
from engine import (Config, env_info, get_device, plot_confusion,
                    plot_history, save_json, show_cls_errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    bai1_path = ROOT / "results" / "bai1" / "results.json"
    if not bai1_path.exists():
        raise SystemExit("Chưa có kết quả Bài 1, hãy chạy bai1_lstm_sentiment.py trước.")
    bai1 = json.loads(bai1_path.read_text(encoding="utf-8"))

    cfg = Config(epochs=args.epochs)
    if bai1["config"] != asdict(cfg):
        print("Cảnh báo: cấu hình khác Bài 1, bảng so sánh sẽ không công bằng.")

    device = get_device()
    out = ROOT / "results" / "bai2"
    out.mkdir(parents=True, exist_ok=True)

    loaders, vocab, data = prepare_sentiment(cfg.batch_size)
    result, preds = train_and_evaluate(
        "GRU", cfg, loaders, vocab, device,
        ROOT / "checkpoints" / "bai2_gru.pt")

    save_json({"config": asdict(cfg), "env": env_info(), "model": result},
              out / "results.json")
    plot_history({"LSTM": bai1["model"]["history"], "GRU": result["history"]},
                 out / "curves.png", "macro-F1")
    plot_confusion(*preds["dev"], out / "confusion_dev.png")
    plot_confusion(*preds["test"], out / "confusion_test.png")

    models = {"LSTM": bai1["model"], "GRU": result}
    table = metrics_table({**models, "TF-IDF": bai1["tfidf"]})
    table.to_csv(out / "test_metrics.csv", index=False)
    cost = cost_table(models)
    cost.to_csv(out / "cost.csv", index=False)

    print("\nSo sánh trên test:")
    print(table.round(4).to_string(index=False))
    print("\nChi phí huấn luyện:")
    print(cost.round(4).to_string(index=False))
    print("\nMột số câu GRU dự đoán sai trên dev:")
    show_cls_errors(data["dev"][0], *preds["dev"], limit=10)


if __name__ == "__main__":
    main()