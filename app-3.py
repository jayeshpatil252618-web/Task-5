from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader

from database import init_db, add, all_rows, clear
from nlp import calculate_match, keyword_analysis

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPLOAD_DIR = ROOT / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)
init_db()

def extract_pdf_text(path):
    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages).strip()

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
        "service": "Resume Job Matching API"
    })

@app.post("/api/match")
def match_resume():

    if "resume" not in request.files:
        return jsonify({
            "error": "Please upload a PDF resume."
        }), 400

    resume = request.files["resume"]
    job_description = request.form.get(
        "job_description",
        ""
    ).strip()

    if not resume.filename:
        return jsonify({
            "error": "No resume selected."
        }), 400

    if not resume.filename.lower().endswith(".pdf"):
        return jsonify({
            "error": "Only PDF resumes are supported."
        }), 400

    if not job_description:
        return jsonify({
            "error": "Please enter a job description."
        }), 400

    safe_name = Path(resume.filename).name
    path = UPLOAD_DIR / safe_name

    try:
        resume.save(path)

        resume_text = extract_pdf_text(path)

        if not resume_text:
            return jsonify({
                "error": "Could not extract text from the PDF."
            }), 400

        score = calculate_match(
            resume_text,
            job_description
        )

        matched, missing = keyword_analysis(
            resume_text,
            job_description
        )

        row_id = add(
            safe_name,
            score,
            ", ".join(matched),
            ", ".join(missing)
        )

        return jsonify({
            "id": row_id,
            "filename": safe_name,
            "score": round(score, 2),
            "matched_keywords": matched,
            "missing_keywords": missing
        })

    except Exception as e:
        return jsonify({
            "error": f"Unable to process resume: {e}"
        }), 400

@app.get("/api/history")
def history():
    return jsonify(all_rows())

@app.delete("/api/history")
def delete_history():
    clear()
    return jsonify({
        "message": "History cleared"
    })

if __name__ == "__main__":
    print("=" * 50)
    print("RESUME JOB MATCHING APP")
    print("Open http://127.0.0.1:5000")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
