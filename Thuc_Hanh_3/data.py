import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent / "data"
PAD_ID, UNK_ID = 0, 1
PAD_TAG = -100
SENTIMENTS = ["negative", "neutral", "positive"]

SOURCES = [
    ("uit-vsfc/UIT-VSFC-train.json",
     "https://drive.google.com/uc?export=download&id=1eJdEStCQBf3GtbOIQfIR2iS-Wb3nlfJX",
     "e41e65672f2ade65ee8d2d394659a2bfeabbce987f0d2ad9e19998068a78fc1e"),
    ("uit-vsfc/UIT-VSFC-dev.json",
     "https://drive.google.com/uc?export=download&id=1UWbzHFeUPu4AuofOEy0Fo0SFotVZHzUe",
     "ab781a38209c17e640e077f83d82ecb9ea4f3469e63fc6ec2a232b0da6fbe823"),
    ("uit-vsfc/UIT-VSFC-test.json",
     "https://drive.google.com/uc?export=download&id=1g4imdabNbswIu91K6zKFGalkjDBBqaQq",
     "14a6c5697907084086551a17c76e908dad88f2dc3b8be720c9880c8fc1212aa6"),
    ("phoner/train_syllable.json",
     "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/train_syllable.json",
     "fb4097f6ac956a03e484319898db0dafc7283c135cd7888f803ad26a5a253419"),
    ("phoner/dev_syllable.json",
     "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/dev_syllable.json",
     "c74393fb5d1547d51ddda175830892ca62a7efa2d93291b21a35cb21bfee612e"),
    ("phoner/test_syllable.json",
     "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/test_syllable.json",
     "4b677028d6db5e1dab74d727d88b719454da60cfeaba681d9ee401fc83b04c77"),
]


# ---------- 1. Đảm bảo có dữ liệu ----------

def sha256_of(content):
    return hashlib.sha256(content).hexdigest()


def download(url, retries=3):
    error = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return resp.read()
        except OSError as e:
            error = e
    raise error


def manual_hint(rel, path):
    return (f"Không lấy được tệp {rel} (link bị chặn, trả về HTML hoặc sai nội dung). "
            f"Hãy tải tay từ nguồn gốc và đặt vào: {path}")


def ensure_data(data_dir=DATA_DIR, only=None):
    # only = "uit-vsfc" hoặc "phoner" để chỉ tải phần cần dùng
    data_dir = Path(data_dir)
    for rel, url, expected in SOURCES:
        if only and not rel.startswith(only):
            continue
        path = data_dir / rel
        if path.exists() and sha256_of(path.read_bytes()) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Đang tải", rel)
        try:
            content = download(url)
        except OSError as e:
            raise RuntimeError(manual_hint(rel, path)) from e
        if sha256_of(content) != expected:
            raise RuntimeError(manual_hint(rel, path))
        path.write_bytes(content)
    return data_dir


# ---------- 2. Đọc dữ liệu ----------

def load_vsfc(data_dir):
    # trả về {split: (danh sách câu, danh sách nhãn số)}
    out = {}
    for split in ["train", "dev", "test"]:
        path = Path(data_dir) / "uit-vsfc" / f"UIT-VSFC-{split}.json"
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        texts = [r["sentence"] for r in rows]
        labels = [SENTIMENTS.index(r["sentiment"]) for r in rows]
        out[split] = (texts, labels)
    return out


def load_phoner(data_dir):
    # trả về {split: (danh sách words, danh sách tags)}
    out = {}
    for split in ["train", "dev", "test"]:
        path = Path(data_dir) / "phoner" / f"{split}_syllable.json"
        words, tags = [], []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row["words"] or len(row["words"]) != len(row["tags"]):
                raise ValueError(f"Dòng lỗi trong {path}")
            words.append(row["words"])
            tags.append(row["tags"])
        out[split] = (words, tags)
    return out


# ---------- 3. Từ điển và mã hóa ----------

def split_words(text):
    return text.lower().split()


def build_vocab(token_lists, min_freq=2):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for token in sorted(counter):
        if counter[token] >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def to_ids(tokens, vocab):
    ids = [vocab.get(t, UNK_ID) for t in tokens]
    return ids if ids else [UNK_ID]   # câu rỗng vẫn cần 1 bước để packing


# ---------- 4. Dataset và DataLoader ----------

class SeqDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate_cls(batch):
    seqs = [torch.tensor(ids) for ids, _ in batch]
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    y = torch.tensor([label for _, label in batch])
    return x, lengths, y


def collate_ner(batch):
    seqs = [torch.tensor(ids) for ids, _ in batch]
    tag_seqs = [torch.tensor(tags) for _, tags in batch]
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    y = pad_sequence(tag_seqs, batch_first=True, padding_value=PAD_TAG)
    return x, lengths, y


def make_loader(dataset, collate_fn, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn)


def prepare_sentiment(batch_size=32, data_dir=DATA_DIR):
    ensure_data(data_dir, only="uit-vsfc")
    data = load_vsfc(data_dir)
    sizes = [len(data[s][0]) for s in ("train", "dev", "test")]
    if sizes != [11426, 1583, 3166]:
        raise ValueError(f"Số mẫu UIT-VSFC không khớp: {sizes}")

    vocab = build_vocab(split_words(t) for t in data["train"][0])
    loaders = {}
    for split, (texts, labels) in data.items():
        ids = [to_ids(split_words(t), vocab) for t in texts]
        loaders[split] = make_loader(SeqDataset(ids, labels), collate_cls,
                                     batch_size, split == "train")
    return loaders, vocab, data


def prepare_ner(batch_size=32, data_dir=DATA_DIR):
    ensure_data(data_dir, only="phoner")
    data = load_phoner(data_dir)
    sizes = [len(data[s][0]) for s in ("train", "dev", "test")]
    if sizes != [5027, 2000, 3000]:
        raise ValueError(f"Số mẫu PhoNER không khớp: {sizes}")

    vocab = build_vocab(data["train"][0])   # giữ nguyên chữ hoa, dấu câu
    tags = sorted({t for row in data["train"][1] for t in row})
    if len(tags) != 21:
        raise ValueError(f"Cần 21 nhãn BIO, đang có {len(tags)}")
    tag_id = {t: i for i, t in enumerate(tags)}

    loaders = {}
    for split, (words, tag_rows) in data.items():
        ids = [to_ids(w, vocab) for w in words]
        tag_ids = [[tag_id[t] for t in row] for row in tag_rows]
        loaders[split] = make_loader(SeqDataset(ids, tag_ids), collate_ner,
                                     batch_size, split == "train")
    return loaders, vocab, tags, data
    