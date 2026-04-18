from pathlib import Path
import random
import shutil

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_images(class_dir: Path) -> list[Path]:
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * SPLITS["train"])
    n_val = int(n_total * SPLITS["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"No existe RAW_DIR: {RAW_DIR}")

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No se encontraron carpetas de clases en {RAW_DIR}")

    print(f"Clases detectadas: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        files = get_images(class_dir)

        if not files:
            print(f"[AVISO] No se encontraron imágenes en {class_dir}")
            continue

        splits = split_files(files)

        for split_name, split_files_list in splits.items():
            target_dir = PROCESSED_DIR / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files_list:
                shutil.copy2(file_path, target_dir / file_path.name)

        print(
            f"{class_dir.name}: total={len(files)} | "
            f"train={len(splits['train'])} | "
            f"val={len(splits['val'])} | "
            f"test={len(splits['test'])}"
        )

    print(f"\nDataset dividido en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
