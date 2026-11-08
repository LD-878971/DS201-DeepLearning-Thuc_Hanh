import argparse
import json
from dataclasses import asdict, replace

from bai1_lstm import ROOT, cost_table, train_and_evaluate
from data import prepare_sentiment
from engine import Config, env_info, get_device, plot_history, save_json

LAYER_LIST = [1, 2, 5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg = Config(epochs=args.epochs, seed=args.seed)
    device = get_device()
    out = ROOT / "results" / "bai4"
    out.mkdir(parents=True, exist_ok=True)

    loaders, vocab, _ = prepare_sentiment(base_cfg.batch_size)

    entries = {}
    for n in LAYER_LIST:
        name = f"LSTM-{n}layer"
        cfg = replace(base_cfg, layers=n)
        path = out / f"{name}_seed{cfg.seed}.json"

        # Kaggle có thể ngắt giữa chừng: cấu hình nào đã chạy xong thì dùng lại
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["config"] == asdict(cfg):
                print(f"{name}: đã có kết quả, bỏ qua (xóa {path.name} để chạy lại)")
                entries[name] = saved["model"]
                continue

        print(f"\n=== {name} (seed {cfg.seed}) ===")
        result, _ = train_and_evaluate(
            "LSTM", cfg, loaders, vocab, device,
            ROOT / "checkpoints" / f"bai4_{name}_seed{cfg.seed}.pt")
        save_json({"config": asdict(cfg), "env": env_info(), "model": result}, path)
        entries[name] = result

    table = cost_table(entries)
    table["test_accuracy"] = [entries[m]["test"]["accuracy"] for m in table["model"]]
    table.to_csv(out / f"depth_table_seed{base_cfg.seed}.csv", index=False)
    plot_history({m: r["history"] for m, r in entries.items()},
                 out / f"curves_seed{base_cfg.seed}.png", "macro-F1")

    print("\nSo sánh độ sâu:")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()