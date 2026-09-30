from pathlib import Path
import uuid
import joblib
import numpy as np

from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "cat_dog_svm.pkl"
UPLOAD_DIR = ROOT / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)

model_data = None

def load_model():
    global model_data

    if model_data is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run: python backend/train_model.py"
            )
        model_data = joblib.load(MODEL_PATH)

    return model_data

def hog_features(image):
    image = image.convert("L").resize((64, 64))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    gx = np.zeros_like(pixels)
    gy = np.zeros_like(pixels)

    gx[:, 1:-1] = pixels[:, 2:] - pixels[:, :-2]
    gy[1:-1, :] = pixels[2:, :] - pixels[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180

    features = []
    cell = 8
    bins = 9

    for y in range(0, 64, cell):
        for x in range(0, 64, cell):
            mag = magnitude[y:y+cell, x:x+cell]
            ang = orientation[y:y+cell, x:x+cell]

            hist = np.zeros(bins)

            for m, a in zip(mag.ravel(), ang.ravel()):
                b = min(int(a // 20), bins - 1)
                hist[b] += m

            hist /= np.linalg.norm(hist) + 1e-8
            features.extend(hist)

    return np.array(features, dtype=np.float32)

@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND, path)

@app.get("/api/health")
def health():
    return jsonify({
        "status": "UP",
        "service": "Cat Dog Classification API"
    })

@app.get("/api/model-info")
def model_info():
    try:
        data = load_model()
        return jsonify({
            "status": "trained",
            "algorithm": "Support Vector Machine",
            "classes": ["Cat", "Dog"],
            "image_size": data["image_size"]
        })
    except FileNotFoundError as e:
        return jsonify({
            "status": "not_trained",
            "error": str(e)
        }), 503

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No image selected."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp"}

    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        return jsonify({
            "error": "Only JPG, JPEG, PNG and BMP images are supported."
        }), 400

    filename = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename

    try:
        file.save(path)

        with Image.open(path) as image:
            features = hog_features(image).reshape(1, -1)

        data = load_model()
        model = data["model"]

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        label = "Dog" if prediction == 1 else "Cat"
        confidence = float(max(probabilities) * 100)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "filename": filename
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        return jsonify({
            "error": f"Unable to process image: {e}"
        }), 400

@app.get("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    print("Cat vs Dog Classifier")
    print("Open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
