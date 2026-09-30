from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
MODEL_PATH = ROOT / "models" / "cat_dog_svm.pkl"
RESULTS = ROOT / "results"

IMAGE_SIZE = (64, 64)
MAX_PER_CLASS = 1000

def hog_features(image):
    image = image.convert("L").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    gx = np.zeros_like(pixels)
    gy = np.zeros_like(pixels)

    gx[:, 1:-1] = pixels[:, 2:] - pixels[:, :-2]
    gy[1:-1, :] = pixels[2:, :] - pixels[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180)

    cell = 8
    bins = 9
    features = []

    for y in range(0, IMAGE_SIZE[1], cell):
        for x in range(0, IMAGE_SIZE[0], cell):
            mag = magnitude[y:y+cell, x:x+cell]
            ang = orientation[y:y+cell, x:x+cell]

            hist = np.zeros(bins)

            for m, a in zip(mag.ravel(), ang.ravel()):
                b = int(a // 20)
                b = min(b, bins - 1)
                hist[b] += m

            hist /= (np.linalg.norm(hist) + 1e-8)
            features.extend(hist)

    return np.array(features, dtype=np.float32)

def load_images():
    X, y = [], []

    for label, folder_name in [(0, "cats"), (1, "dogs")]:
        folder = TRAIN_DIR / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        files = [
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ][:MAX_PER_CLASS]

        if not files:
            raise ValueError(f"No images found in {folder}")

        for path in files:
            try:
                with Image.open(path) as image:
                    X.append(hog_features(image))
                    y.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y)

RESULTS.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

print("Loading images...")
X, y = load_images()

print(f"Images loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
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
print(classification_report(
    y_test,
    predictions,
    target_names=["Cat", "Dog"]
))

joblib.dump(
    {
        "model": model,
        "image_size": IMAGE_SIZE
    },
    MODEL_PATH
)

with open(RESULTS / "model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(
        classification_report(
            y_test,
            predictions,
            target_names=["Cat", "Dog"]
        )
    )

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Cat vs Dog Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)
plt.close()

print(f"Model saved: {MODEL_PATH}")
print("Training completed.")
