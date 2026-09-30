# Task-02 Full-Stack Customer Segmentation

A full-stack Machine Learning application that uses K-Means clustering to group retail customers according to annual income and spending score.

## Dataset
Kaggle Customer Segmentation:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Download `Mall_Customers.csv` and place it in:
`data/Mall_Customers.csv`

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask REST API
- Machine Learning: K-Means Clustering
- Database: SQLite
- Visualization: Matplotlib

## Features
- Customer segmentation using K-Means
- Elbow Method
- Interactive customer segmentation form
- Cluster prediction API
- Prediction history stored in SQLite
- Customer cluster visualization
- Responsive web interface

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `Mall_Customers.csv` inside `data/`.

Train the model:

```powershell
python backend/train_model.py
```

Start the application:

```powershell
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API

GET `/api/health`

GET `/api/model-info`

POST `/api/predict`

Example:
```json
{
  "income": 60,
  "spending_score": 70
}
```

GET `/api/history`

DELETE `/api/history`
