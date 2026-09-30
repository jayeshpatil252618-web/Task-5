# Task-04 Full-Stack Hand Gesture Recognition

A full-stack Machine Learning application that recognizes hand gestures from uploaded images.

This follows the Prodigy InfoTech ML Task-04: develop a hand gesture recognition model from image/video data for human-computer interaction.

## Dataset

Kaggle LeapGestRecog:
https://www.kaggle.com/gti-upm/leapgestrecog

After downloading and extracting the dataset, copy the gesture image folders into:

```text
data/gestures/
```

The training script automatically discovers the gesture folders.

## Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- ML: Support Vector Machine (SVM)
- Image processing: Pillow
- Features: HOG-style gradient features
- Database: SQLite
- Visualization: Matplotlib

## Features

- Upload a hand gesture image
- Extract image features
- SVM multi-class classification
- Display predicted gesture and confidence
- Prediction history
- REST API
- Confusion matrix
- Training metrics

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the LeapGestRecog gesture folders under:

```text
data/gestures/
```

Then train:

```powershell
python backend/train_model.py
```

Start:

```powershell
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Dataset folder example

```text
data/
└── gestures/
    ├── 00_palm/
    ├── 01_l/
    ├── 02_fist/
    ├── 03_fist_moved/
    ├── 04_thumb/
    ├── 05_index/
    ├── 06_ok/
    ├── 07_palm_moved/
    ├── 08_c/
    └── 09_down/
```

The actual dataset folder names may vary slightly. The program uses the discovered folder names as class labels.

## API

GET `/api/health`

GET `/api/model-info`

POST `/api/predict` with multipart field `image`

GET `/api/history`

DELETE `/api/history`
