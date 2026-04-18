# ArqueoVision

ArqueoVision is a deep-learning pipeline that classifies architectural heritage images into four historical styles: **Baroque**, **Gothic**, **Hispanic-Muslim**, and **Renaissance**. It exposes the model through a FastAPI inference service and includes a minimal web frontend for interactive prediction.

---

## What ArqueoVision does

- Trains a fine-tuned **ResNet-18** on a labelled dataset of architectural photographs.
- Tracks experiments, metrics, and artefacts with **MLflow**.
- Serves predictions via a **REST API** (FastAPI + Uvicorn) that returns the top-3 classes with confidence scores.
- Provides a browser-based **frontend** where users can drag-and-drop or upload an image and immediately see the classification results and a local log of all past analyses.

---

## Project structure

```
arqueovision/
├── data/
│   ├── raw/                  # Original images organised by class
│   │   ├── Baroque/
│   │   ├── Gothic/
│   │   ├── Hispanic-Muslim/
│   │   └── Renaissance/
│   └── processed/            # Auto-generated train / val / test splits
│       ├── train/
│       ├── val/
│       └── test/
├── frontend/
│   └── index.html            # Single-page prediction UI
├── models/
│   ├── best_model.pth        # Saved model weights (not tracked by git)
│   └── classification_report.txt
├── mlruns/                   # MLflow experiment data (not tracked by git)
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   └── inference_service.py
│   ├── inference/
│   │   └── predict.py
│   └── training/
│       ├── split_dataset.py  # Train/val/test splitter
│       └── train.py          # Training loop
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## How to train

### 1. Prepare the dataset

Place raw images inside `data/raw/<ClassName>/` (only `.jpg`, `.jpeg`, `.png`, `.webp` are picked up), then run the splitter to create the 70/15/15 train/val/test partitions:

```bash
python -m src.training.split_dataset
```

### 2. Run the training script

```bash
python -m src.training.train
```

The script will:
- Fine-tune ResNet-18 (frozen backbone, trainable head) for 8 epochs by default.
- Log parameters, per-epoch metrics, and the best checkpoint to MLflow.
- Save the best weights to `models/best_model.pth` and a classification report to `models/classification_report.txt`.

Key hyperparameters (editable at the top of `train.py`):

| Variable | Default |
|---|---|
| `EPOCHS` | `8` |
| `BATCH_SIZE` | `16` |
| `IMAGE_SIZE` | `224` |
| `LEARNING_RATE` | `1e-3` |

---

## How to launch MLflow

```bash
mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to explore runs, compare metrics, and inspect artefacts.

---

## How to start FastAPI

Make sure `models/best_model.pth` exists, then:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns API status and loaded class names |
| `POST` | `/predict-image` | Accepts a multipart image file, returns top-3 predictions |

Interactive docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## How to run with Docker Compose

### Requirements

- Docker ≥ 24
- Docker Compose plugin (`docker compose`)
- *(Optional)* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU inference

### Build and start

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | [http://localhost](http://localhost) |
| API | [http://localhost:8000](http://localhost:8000) |
| API docs | [http://localhost:8000/docs](http://localhost:8000/docs) |

The model weights are mounted from `./models` into the container at runtime — no need to rebuild the image when you retrain.

### CPU-only mode

Remove or comment out the `deploy.resources` block in `docker-compose.yml` before starting.

### Stop

```bash
docker compose down
```
