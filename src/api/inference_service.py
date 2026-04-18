from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceService:
    def __init__(self) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint["image_size"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        top_k = min(top_k, len(self.class_names))
        top_probs, top_idxs = torch.topk(probs, k=top_k)

        results = []
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            results.append({
                "label": self.class_names[idx],
                "score": round(prob, 4),
            })

        return results