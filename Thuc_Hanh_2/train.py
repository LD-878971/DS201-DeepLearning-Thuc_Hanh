# ===================== train.py =====================
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import train_one_epoch, evaluate
from models import build_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_dl, val_dl, classes = build_train_val_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=len(classes), img_size=args.img_size).to(device)
    print(f"Model: {args.model} | params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = []

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion, device)

        history.append((ep, tr_loss, va_loss, tr_acc, va_acc, va_f1))
        print(f"epoch {ep:2d}  train {tr_loss:.4f}/{tr_acc:.4f}   "
              f"val {va_loss:.4f}/{va_acc:.4f}   macro-F1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "model_name": args.model,
                "epoch": ep,
                "f1": va_f1,
            }, out_dir / "best.pt")
            print(f"  → lưu best.pt (F1={va_f1:.4f})")

    with open(out_dir / "history.csv", "w") as f:
        f.write("epoch,tr_loss,va_loss,tr_acc,va_acc,va_f1\n")
        for row in history:
            f.write(",".join(map(str, row)) + "\n")

    print(f"\nDone. Best val macro-F1 = {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["lenet", "googlenet", "resnet18", "resnet50"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    # gợi ý mặc định cho pretrained
    if args.model == "resnet50" and args.img_size == 64:
        print("Gợi ý: ResNet50 pretrained nên dùng --img_size 224 và --lr 1e-4")
    main(args)