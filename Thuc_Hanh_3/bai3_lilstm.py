import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data import prepare_ner
from engine import (Config, env_info, evaluate_ner, get_device, ner_metrics,
                    plot_history, save_json, seed_all, show_ner_errors,
                    train_model, training_summary)
from models import BiLSTMTagger

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    device = get_device()
    out = ROOT / "results" / "bai3"
    out.mkdir(parents=True, exist_ok=True)

    loaders, vocab, tags, data = prepare_ner(cfg.batch_size)

    seed_all(cfg.seed)
    model = BiLSTMTagger(len(vocab), len(tags), cfg.emb_dim, cfg.hidden,
                         cfg.layers, cfg.dropout)
    history, best_epoch = train_model(
        model, loaders, cfg, device, "ner",
        ROOT / "checkpoints" / "bai3_bilstm.pt", tags=tags)

    result = training_summary(model, history, best_epoch)
    result["history"] = history
    _, dev_true, dev_pred = evaluate_ner(model, loaders["dev"], device, tags)
    _, test_true, test_pred = evaluate_ner(model, loaders["test"], device, tags)
    result["dev"] = ner_metrics(dev_true, dev_pred)
    result["test"] = ner_metrics(test_true, test_pred)

    save_json({"config": asdict(cfg), "env": env_info(), "model": result},
              out / "results.json")
    plot_history({"BiLSTM": history}, out / "curves.png", "entity-F1")

    per_type = pd.DataFrame(result["test"]["per_type"]).T
    per_type.to_csv(out / "test_per_type.csv")

    test = result["test"]
    print(f"\nTest (mức thực thể, strict IOB2): precision={test['precision']:.4f} "
          f"recall={test['recall']:.4f} micro-F1={test['f1_micro']:.4f}")
    print("\nF1 từng loại thực thể trên test:")
    print(per_type.round(4).to_string())
    print(f"\nSố tham số: {result['parameters']:,} | "
          f"epoch tốt nhất: {result['best_epoch']} | "
          f"{result['seconds_per_epoch']:.0f}s/epoch")
    print("\nMột số câu dự đoán sai trên dev:")
    show_ner_errors(data["dev"][0], dev_true, dev_pred, limit=10)


if __name__ == "__main__":
    main()