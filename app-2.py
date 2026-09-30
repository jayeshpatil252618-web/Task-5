from pathlib import Path
import uuid
import joblib
import numpy as np

from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import init_db, add, all_rows, clear
from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "hand_gesture_svm.pkl"
UPLOAD_DIR = ROOT / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)

model_data = None
init_db()

def get_model():
    global model_data

    if model_data is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run: python backend/train_model.py"
            )
        model_data = joblib.load(MODEL_PATH)

    return model_data

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
        "service": "Hand Gesture Recognition API"
    })

@app.get("/api/model-info")
def model_info():
    try:
        data = get_model()
        return jsonify({
            "status": "trained",
            "algorithm": "SVM",
            "classes": data["classes"]
        })
    except FileNotFoundError as e:
        return jsonify({
            "status": "not_trained",
            "error": str(e)
        }), 503

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a gesture image."}), 400

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
            features = extract_features(image).reshape(1, -1)

        data = get_model()
        model = data["model"]
        classes = data["classes"]

        predicted_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        gesture = classes[predicted_index]
        confidence = float(np.max(probabilities) * 100)

        row_id = add(
            gesture,
            confidence,
            filename
        )

        return jsonify({
            "id": row_id,
            "prediction": gesture,
            "confidence": round(confidence, 2),
            "filename": filename,
            "image_url": f"/uploads/{filename}"
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        return jsonify({
            "error": f"Unable to classify image: {e}"
        }), 400

@app.get("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.get("/api/history")
def history():
    return jsonify(all_rows())

@app.delete("/api/history")
def delete_history():
    clear()
    return jsonify({"message": "History cleared"})

if __name__ == "__main__":
    print("=" * 50)
    print("HAND GESTURE RECOGNITION APP")
    print("Open http://127.0.0.1:5000")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
