from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, image_size


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict(image_path):
    model, class_names, image_size = load_model()
    transform = get_transform(image_size)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=3)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        results.append((class_names[idx], float(prob)))

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py ruta_imagen.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    results = predict(image_path)

    print("\nTop 3 predicciones:")
    for cls, prob in results:
        print(f"{cls}: {prob:.4f}")
