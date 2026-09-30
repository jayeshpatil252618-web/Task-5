from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import init_db, add, all_rows, clear

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL = ROOT / "models/customer_kmeans_model.pkl"

app = Flask(__name__)
CORS(app)
init_db()

def get_model():
    if not MODEL.exists():
        raise FileNotFoundError(
            "Model not found. Run: python backend/train_model.py"
        )
    return joblib.load(MODEL)

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
        "service": "Customer Segmentation API"
    })

@app.get("/api/model-info")
def model_info():
    try:
        m = get_model()
        return jsonify({
            "status": "trained",
            "algorithm": "K-Means Clustering",
            "clusters": 5,
            "features": m["features"]
        })
    except FileNotFoundError as e:
        return jsonify({
            "status": "not_trained",
            "error": str(e)
        }), 503

@app.post("/api/predict")
def predict():
    try:
        data = request.get_json() or {}

        income = float(data["income"])
        score = float(data["spending_score"])

        if income < 0:
            return jsonify({"error": "Annual income cannot be negative."}), 400

        if score < 0 or score > 100:
            return jsonify({
                "error": "Spending score must be between 0 and 100."
            }), 400

        m = get_model()

        X = pd.DataFrame(
            [[income, score]],
            columns=m["features"]
        )

        X_scaled = m["scaler"].transform(X)
        cluster = int(m["model"].predict(X_scaled)[0])

        row_id = add(income, score, cluster)

        return jsonify({
            "id": row_id,
            "income": income,
            "spending_score": score,
            "cluster": cluster
        })

    except KeyError:
        return jsonify({
            "error": "income and spending_score are required."
        }), 400

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    except (ValueError, TypeError):
        return jsonify({
            "error": "Enter valid numeric values."
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/history")
def history():
    return jsonify(all_rows())

@app.delete("/api/history")
def delete_history():
    clear()
    return jsonify({"message": "History cleared"})

if __name__ == "__main__":
    print("Customer Segmentation App")
    print("Open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
