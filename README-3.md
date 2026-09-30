# Task-05 Full-Stack Resume Job Matching

A full-stack NLP/ML application that matches a resume against job descriptions and calculates a text-similarity score.

## Features
- Upload PDF resume
- Enter job description
- Extract resume text
- TF-IDF vectorization
- Cosine similarity
- Match score
- Keyword extraction
- Missing keyword detection
- Prediction/history storage using SQLite
- REST API
- Responsive frontend

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- ML/NLP: TF-IDF + Cosine Similarity
- PDF processing: PyPDF2
- Database: SQLite

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start:

```powershell
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How it works

1. Upload a resume PDF.
2. Enter a job description.
3. Backend extracts resume text.
4. TF-IDF converts resume and job description into vectors.
5. Cosine similarity calculates the match score.
6. Important job keywords are compared with resume keywords.
7. The result is displayed in the web application.

This score is a text-similarity indicator, not a hiring decision or guarantee of interview selection.
