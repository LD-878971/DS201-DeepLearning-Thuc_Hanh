import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import classification_report as entity_report
from seqeval.metrics import f1_score as entity_f1
from seqeval.scheme import IOB2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_fscore_support)

from data import PAD_TAG, SENTIMENTS


@dataclass
class Config:
    emb_dim: int = 300
    hidden: int = 256
    layers: int = 5
    dropout: float = 0.3
    batch_size: int = 32
    lr: float = 1e-3
    clip: float = 1.0
    epochs: int = 10
    seed: int = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def env_info():
    info = {"python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ---------- Huấn luyện ----------

def compute_loss(task, logits, labels):
    if task == "ner":
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=PAD_TAG)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    true, pred = [], []
    for ids, lengths, labels in loader:
        logits = model(ids.to(device), lengths)
        true += labels.tolist()
        pred += logits.argmax(-1).cpu().tolist()
    score = f1_score(true, pred, labels=[0, 1, 2], average="macro",
                     zero_division=0)
    return score, true, pred


@torch.no_grad()
def evaluate_ner(model, loader, device, tags):
    model.eval()
    true, pred = [], []
    for ids, lengths, labels in loader:
        guess = model(ids.to(device), lengths).argmax(-1).cpu()
        for gold_row, guess_row, n in zip(labels, guess, lengths.tolist()):
            true.append([tags[i] for i in gold_row[:n].tolist()])
            pred.append([tags[i] for i in guess_row[:n].tolist()])
    score = entity_f1(true, pred, mode="strict", scheme=IOB2, zero_division=0)
    return score, true, pred


def train_model(model, loaders, cfg, device, task, ckpt_path, tags=None):
    # task: "cls" hoặc "ner"; trả về (history, best_epoch)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_score, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        loss_sum, count = 0.0, 0
        for ids, lengths, labels in loaders["train"]:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = compute_loss(task, model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            n = (labels != PAD_TAG).sum().item() if task == "ner" else labels.numel()
            loss_sum += loss.item() * n
            count += n
        seconds = time.time() - start

        if task == "ner":
            dev_score = evaluate_ner(model, loaders["dev"], device, tags)[0]
        else:
            dev_score = evaluate_cls(model, loaders["dev"], device)[0]

        history.append({"epoch": epoch, "loss": loss_sum / count,
                        "dev_score": dev_score, "seconds": seconds})
        print(f"epoch {epoch}: loss={loss_sum / count:.4f} "
              f"dev={dev_score:.4f} ({seconds:.0f}s)")

        if dev_score > best_score:
            best_score, best_epoch = dev_score, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    return history, best_epoch


def training_summary(model, history, best_epoch):
    return {
        "parameters": count_params(model),
        "best_epoch": best_epoch,
        "best_dev_score": history[best_epoch - 1]["dev_score"],
        "seconds_per_epoch": sum(h["seconds"] for h in history) / len(history),
    }


# ---------- Độ đo ----------

def cls_metrics(true, pred):
    labels = [0, 1, 2]
    p, r, f, _ = precision_recall_fscore_support(
        true, pred, labels=labels, average="macro", zero_division=0)
    per_class = f1_score(true, pred, labels=labels, average=None,
                         zero_division=0)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "f1_per_class": {name: float(v) for name, v in zip(SENTIMENTS, per_class)},
    }


def ner_metrics(true, pred):
    rep = entity_report(true, pred, mode="strict", scheme=IOB2,
                        output_dict=True, zero_division=0)
    micro = rep["micro avg"]
    per_type = {}
    for name, values in rep.items():
        if "avg" in name:
            continue
        per_type[name] = {k: float(v) for k, v in values.items()}
    return {"precision": float(micro["precision"]),
            "recall": float(micro["recall"]),
            "f1_micro": float(micro["f1-score"]),
            "per_type": per_type}


def run_tfidf(train_texts, train_y, eval_sets):
    # eval_sets: {"dev": (texts, labels), "test": (texts, labels)}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 token_pattern=r"(?u)\b\w+\b")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(vectorizer.fit_transform(train_texts), train_y)
    out = {}
    for name, (texts, labels) in eval_sets.items():
        pred = clf.predict(vectorizer.transform(texts))
        out[name] = cls_metrics(labels, pred)
    return out


# ---------- Báo cáo ----------

def _make_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    _make_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_history(histories, path, score_name):
    # histories: {"tên": history}
    _make_dir(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3))
    for name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        ax1.plot(epochs, [h["loss"] for h in history], marker="o", label=name)
        ax2.plot(epochs, [h["dev_score"] for h in history], marker="o", label=name)
    ax1.set_title("Train loss")
    ax2.set_title(f"Dev {score_name}")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(true, pred, path):
    _make_dir(path)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        true, pred, labels=[0, 1, 2], display_labels=SENTIMENTS,
        ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Chỉ in ra màn hình, không ghi vào results/ để không đưa dữ liệu lên repo công khai
def show_cls_errors(texts, true, pred, limit=5):
    shown = 0
    for text, t, p in zip(texts, true, pred):
        if t != p:
            print(f"[{SENTIMENTS[t]} -> {SENTIMENTS[p]}] {text}")
            shown += 1
            if shown == limit:
                break


def show_ner_errors(words, true, pred, limit=5):
    shown = 0
    for tokens, gold, guess in zip(words, true, pred):
        if gold == guess:
            continue
        print(" ".join(tokens))
        for tok, g, p in zip(tokens, gold, guess):
            if g != p:
                print(f"   {tok}: đúng={g}, dự đoán={p}")
        shown += 1
        if shown == limit:
            break