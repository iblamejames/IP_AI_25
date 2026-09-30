"""ЛР 2, вариант 5: предобученная DenseNet121, STL-10, SGD."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lab1" / "src"))
from lab1 import CNN, classify, restore, seed_everything, train


def build_model():
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.requires_grad_(False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model


def compare(image_path):
    image = Image.open(image_path).convert("RGB")
    custom = restore(CNN(), ROOT.parent / "lab1" / "results" / "model.pth")
    pretrained = restore(build_model(), ROOT / "results" / "model.pth", True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    results = {}
    for ax, model, name, transfer in [(axes[0], custom, "CNN", False),
                                      (axes[1], pretrained, "DenseNet121", True)]:
        label, score = classify(model, image, transfer)
        results[name] = dict(image=image_path.name, **{"class": label}, softmax=score)
        ax.imshow(image)
        ax.set_title(f"{name}: {label}\nSoftmax: {score:.1%}")
        ax.axis("off")
        print(f"{name}: {label}, softmax={score:.4f}")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "comparison.png", dpi=180)
    plt.close(fig)
    (ROOT / "results" / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Сравнить обе сети на своём изображении")
    parser.add_argument("--choose", action="store_true", help="Выбрать изображение в окне")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error("epochs должно быть положительным, workers неотрицательным")
    seed_everything()
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
        compare(args.image)
    else:
        train(build_model(), ROOT / "results", transfer=True, epochs=args.epochs, workers=args.workers)
