# ===================== evaluate.py =====================
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data import build_test_loader
from engine import evaluate
from models import build_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)
    model_name = args.model or ckpt.get("model_name", "lenet")

    test_dl, test_classes = build_test_loader(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    assert classes == test_classes, "Thứ tự lớp train/test không khớp!"

    model = build_model(model_name, num_classes=len(classes), img_size=img_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)

    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")

    print(f"\n========== KẾT QUẢ TRÊN TẬP TEST ({model_name}) ==========")
    print(f"Loss          : {test_loss:.4f}")
    print(f"Accuracy      : {test_acc:.4f}")
    print(f"Precision-mac : {prec:.4f}")
    print(f"Recall-mac    : {rec:.4f}")
    print(f"F1-macro      : {test_f1:.4f}")
    print("======================================================\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix – {model_name} (Test)")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90, fontsize=7)
    plt.yticks(ticks, classes, fontsize=7)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path = Path(args.ckpt).parent / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Đã lưu {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=["lenet", "googlenet", "resnet18", "resnet50"],
                        help="Nếu không chỉ định sẽ lấy từ checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)