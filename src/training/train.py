from pathlib import Path
import copy

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 8
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    f1 = f1_score(labels_all, preds_all, average="macro")

    return loss, f1, preds_all, labels_all


def main():
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders()
    class_names = train_dataset.classes

    print("Clases:", class_names)
    print("Device:", device)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    mlflow.set_experiment("arqueovision-style-classification")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    with mlflow.start_run():
        mlflow.log_param("model_name", "resnet18")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("frozen_backbone", True)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1_macro", val_f1, step=epoch)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1_macro={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)

        test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)
        report = classification_report(test_labels, test_preds, target_names=class_names)

        print("\n=== TEST REPORT ===")
        print(report)

        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_f1_macro", test_f1)

        report_path = MODELS_DIR / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        model_path = MODELS_DIR / "best_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
        }, model_path)

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(model_path))
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"\nMejor val_f1_macro: {best_val_f1:.4f}")
        print(f"Test f1_macro: {test_f1:.4f}")
        print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
