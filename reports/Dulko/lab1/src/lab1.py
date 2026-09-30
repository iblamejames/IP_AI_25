"""ЛР 1, вариант 5: простая CNN, размеченная часть STL-10, SGD."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")
SEED = 5


class LabeledSTL10(datasets.STL10):
    # Для train/test проверяем размеченные файлы, без неиспользуемой unlabeled-части.
    train_list = datasets.STL10.train_list[:2]
    url = datasets.STL10.url.replace("http:", "https:")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def preprocessing(transfer=False, training=False):
    size = 224 if transfer else 96
    mean = (0.485, 0.456, 0.406) if transfer else (0.5,) * 3
    std = (0.229, 0.224, 0.225) if transfer else (0.5,) * 3
    steps = [transforms.Resize(256), transforms.CenterCrop(size)] if transfer else [transforms.Resize((size, size))]
    if training:
        steps = ([transforms.Resize(256), transforms.RandomCrop(size)] if transfer
                 else [transforms.RandomCrop(size, padding=8)])
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def loaders(transfer, batch_size, workers):
    train = LabeledSTL10(DATA, split="train", download=True, transform=preprocessing(transfer, True))
    valid = LabeledSTL10(DATA, split="train", download=True, transform=preprocessing(transfer))
    test = LabeledSTL10(DATA, split="test", download=True, transform=preprocessing(transfer))
    # Одинаковое разделение для обеих лабораторных: 400/100 изображений каждого класса.
    rng = np.random.default_rng(SEED)
    train_ids, valid_ids = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(train.labels == label))
        train_ids.extend(ids[:400].tolist())
        valid_ids.extend(ids[400:].tolist())
    options = dict(batch_size=batch_size, num_workers=workers,
                   pin_memory=DEVICE.type == "cuda", persistent_workers=workers > 0)
    return (DataLoader(Subset(train, train_ids), shuffle=True, **options),
            DataLoader(Subset(valid, valid_ids), **options), DataLoader(test, **options), test)


def pass_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    # Замороженная DenseNet должна сохранять статистику BatchNorm из ImageNet.
    if hasattr(model, "features"):
        model.features.eval()
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, 100 * correct / count


@torch.no_grad()
def classify(model, image, transfer=False):
    model.eval()
    probabilities = model(preprocessing(transfer)(image).unsqueeze(0).to(DEVICE)).softmax(1)[0]
    label = probabilities.argmax().item()
    return CLASSES[label], probabilities[label].item()


def predict(model, image_path, output, transfer=False):
    image = Image.open(image_path).convert("RGB")
    label, score = classify(model, image, transfer)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image)
    ax.set_title(f"Prediction: {label}; softmax: {score:.1%}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"{image_path}: {label}, softmax={score:.4f}", flush=True)
    return {"image": Path(image_path).name, "class": label, "softmax": score}


def restore(model, checkpoint, transfer=False):
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    (model.classifier if transfer else model).load_state_dict(state)
    return model.to(DEVICE).eval()


def train(model, output, transfer=False, epochs=25, batch_size=64, lr=0.01, workers=2):
    output.mkdir(parents=True, exist_ok=True)
    train_ld, valid_ld, test_ld, test = loaders(transfer, batch_size, workers)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    history, best_loss, best_epoch = [], float("inf"), 0
    start = time.perf_counter()
    print(f"Device: {DEVICE}; train=4000; validation=1000; test=8000", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = pass_epoch(model, train_ld, optimizer)
        valid_loss, valid_acc = pass_epoch(model, valid_ld)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                            validation_loss=valid_loss, validation_accuracy=valid_acc))
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            torch.save((model.classifier if transfer else model).state_dict(), output / "model.pth")
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}: loss={train_loss:.4f}; val_loss={valid_loss:.4f}; "
              f"train_acc={train_acc:.2f}%; val_acc={valid_acc:.2f}%", flush=True)
    restore(model, output / "model.pth", transfer)
    test_loss, test_acc = pass_epoch(model, test_ld)
    metrics = dict(model="DenseNet121" if transfer else "CNN", seed=SEED, epochs=epochs,
                   best_epoch=best_epoch, batch_size=batch_size, lr=lr, momentum=0.9,
                   weight_decay=1e-4, optimizer="SGD", scheduler="CosineAnnealingLR",
                   train_size=4000, validation_size=1000, test_size=8000,
                   parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   test_loss=test_loss, test_accuracy=test_acc, seconds=time.perf_counter() - start,
                   device=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU",
                   torch_version=torch.__version__, history=history)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, label in [("train", "Train"), ("validation", "Validation")]:
        axes[0].plot(range(1, epochs + 1), [h[prefix + "_loss"] for h in history], label=label)
        axes[1].plot(range(1, epochs + 1), [h[prefix + "_accuracy"] for h in history], label=label)
    for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy (%)"]):
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for label, ax in enumerate(axes.flat):
        index = int(np.flatnonzero(test.labels == label)[0])
        image = Image.fromarray(test.data[index].transpose(1, 2, 0))
        predicted, score = classify(model, image, transfer)
        ax.imshow(image)
        ax.set_title(f"True: {CLASSES[label]}\nPred: {predicted} ({score:.0%})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=180)
    plt.close(fig)
    print(f"Test accuracy: {test_acc:.2f}%; test loss: {test_loss:.4f}", flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Классифицировать своё изображение вместо обучения")
    parser.add_argument("--choose", action="store_true", help="Выбрать изображение в окне")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error("epochs должно быть положительным, workers неотрицательным")
    seed_everything()
    output = ROOT / "results"
    if args.choose:
        from tkinter import Tk, filedialog
        window = Tk()
        window.withdraw()
        selected = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        window.destroy()
        if not selected:
            raise SystemExit("Изображение не выбрано")
        args.image = Path(selected)
    if args.image:
        model = restore(CNN(), output / "model.pth")
        predict(model, args.image, output / "custom_prediction.png")
    else:
        train(CNN(), output, epochs=args.epochs, workers=args.workers)
