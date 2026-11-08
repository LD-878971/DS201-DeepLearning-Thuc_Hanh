from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int = 64):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

def build_train_val_loaders(
    data_root: str | Path,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
):
    data_root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)

    val_root = data_root / "val"
    if val_root.is_dir():
        val_ds = datasets.ImageFolder(val_root, transform=eval_tf)
    else:
        n_total = len(train_ds)
        n_val   = max(1, int(0.1 * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        val_ds.dataset.transform = eval_tf

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if hasattr(train_ds, "classes"):
        classes = train_ds.classes
    else:
        classes = train_ds.dataset.classes

    return train_dl, val_dl, classes

def build_test_loader(
    data_root: str | Path,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 2,
):
    data_root = Path(data_root)
    _, eval_tf = build_transforms(img_size)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dl, test_ds.classes