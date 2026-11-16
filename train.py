
import os
import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "mlflow"
LABELS_FILE = BASE_DIR / "labels.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 3
SEED = 42

# Set DATASET_DIR if your dataset is stored elsewhere.
possible_dataset_paths = [
    BASE_DIR / "BanglaLekha-Isolated" / "Images",
    BASE_DIR / "dataset" / "BanglaLekha-Isolated" / "Images",
    BASE_DIR / "data" / "BanglaLekha-Isolated" / "Images",
    BASE_DIR / "Images",
]

dataset_dir = Path(os.environ["DATASET_DIR"]) if os.environ.get("DATASET_DIR") else None

if dataset_dir is None:
    dataset_dir = next(
        (p for p in possible_dataset_paths if p.is_dir()),
        None
    )

if dataset_dir is None or not dataset_dir.is_dir():
    raise FileNotFoundError(
        "Could not find the dataset Images folder. "
        "Set DATASET_DIR to the folder containing the 84 class folders."
    )

if not LABELS_FILE.exists():
    raise FileNotFoundError("labels.json was not found in the project folder.")

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    labels = json.load(file)

print("Dataset:", dataset_dir)
print("Loading BanglaLekha-Isolated...")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(dataset_dir),
    validation_split=0.2,
    subset="training",
    seed=SEED,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(dataset_dir),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Class folders:", class_names)

if num_classes != 84:
    raise ValueError(
        f"Expected 84 class folders, but found {num_classes}."
    )

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# --------------------------------------------------
# CNN model
# --------------------------------------------------

def create_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# MLflow experiment tracking
# --------------------------------------------------

tracking_uri = ARTIFACT_DIR.resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Bangla_Handwritten_Character_Recognition")

experiments = [
    {"name": "Experiment_1", "learning_rate": 0.001},
    {"name": "Experiment_2", "learning_rate": 0.0005},
]

best_accuracy = -1.0

for experiment in experiments:
    print("\nStarting:", experiment["name"])

    with mlflow.start_run(run_name=experiment["name"]):
        model = create_model(experiment["learning_rate"])

        mlflow.log_params({
            "learning_rate": experiment["learning_rate"],
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "image_size": "64x64",
            "number_of_classes": num_classes,
            "optimizer": "Adam",
            "model": "CNN"
        })

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )

        val_accuracy = max(history.history["val_accuracy"])
        val_loss = min(history.history["val_loss"])

        mlflow.log_metric("best_validation_accuracy", float(val_accuracy))
        mlflow.log_metric("best_validation_loss", float(val_loss))

        print("Validation accuracy:", round(val_accuracy * 100, 2), "%")

        # Save the best-performing model.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save(MODEL_DIR / "model.keras")

            with open(BASE_DIR / "class_names.json", "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)

            mlflow.log_artifact(str(MODEL_DIR / "model.keras"))

print("\nTraining experiments completed.")
print("Best validation accuracy:", round(best_accuracy * 100, 2), "%")
print("Model saved at:", MODEL_DIR / "model.keras")
print("MLflow tracking directory:", ARTIFACT_DIR)

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
