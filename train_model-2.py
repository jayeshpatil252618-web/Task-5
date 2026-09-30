from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gestures"
MODEL_PATH = ROOT / "models" / "hand_gesture_svm.pkl"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

MAX_IMAGES_PER_CLASS = 500
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "data/gestures does not exist. Download LeapGestRecog and "
        "put its gesture class folders inside data/gestures."
    )

class_dirs = sorted([
    p for p in DATA_DIR.iterdir()
    if p.is_dir()
])

if len(class_dirs) < 2:
    raise ValueError(
        "At least two gesture class folders are required."
    )

X = []
y = []
class_names = []

for class_index, class_dir in enumerate(class_dirs):
    class_names.append(class_dir.name)

    images = [
        p for p in class_dir.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ][:MAX_IMAGES_PER_CLASS]

    print(f"{class_dir.name}: {len(images)} images")

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                X.append(extract_features(image))
                y.append(class_index)
        except Exception:
            continue

if len(X) == 0:
    raise ValueError("No valid images were found.")

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

print("Training SVM...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test,
    predictions,
    target_names=class_names,
    zero_division=0
)

print(report)

joblib.dump(
    {
        "model": model,
        "classes": class_names
    },
    MODEL_PATH
)

with open(RESULTS / "model_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(report)

with open(RESULTS / "model_info.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "algorithm": "SVM",
            "classes": class_names,
            "accuracy": float(accuracy),
            "training_images": int(len(X_train)),
            "testing_images": int(len(X_test))
        },
        f,
        indent=4
    )

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 7))
plt.imshow(cm)
plt.title("Hand Gesture Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)
plt.close()

print("=" * 55)
print("HAND GESTURE TRAINING COMPLETED")
print("=" * 55)
print(f"Classes: {len(class_names)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Model: {MODEL_PATH}")
print("=" * 55)
