# Task-03 Full-Stack Cat vs Dog Image Classification

Full-stack machine-learning web application for classifying an uploaded image as **Cat** or **Dog** using an SVM classifier.

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: SVM with HOG image features
- Image Processing: Pillow
- Model: scikit-learn
- Storage: local uploads/results

## Dataset
Use a cats-vs-dogs image dataset. Organize it like:

```text
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Put the training images into `data/train/cats` and `data/train/dogs`.

## Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python backend/train_model.py
```

The model will be saved to:

```text
models/cat_dog_svm.pkl
```

## Start Web App

```powershell
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How it works

1. User uploads a cat/dog image.
2. Backend reads the image.
3. Image is resized and converted into HOG features.
4. SVM predicts Cat or Dog.
5. Result and confidence are returned to the frontend.

If the training dataset is not present, the project still starts, but prediction requires a trained model.
