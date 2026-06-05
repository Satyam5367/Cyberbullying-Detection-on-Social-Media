# 🛡️ CyberGuard AI — Cyberbullying Detection System

**Group 8 NLP Machine Learning Project**

A full-stack web application for real-time cyberbullying detection using an ensemble of 4 ML models.

---

## 🚀 Quick Start (3 Steps)

### Windows
```
Double-click run.bat
```
OR manually:
```bash
pip install flask
python app.py
```
Open → http://127.0.0.1:5000

### Mac / Linux
```bash
chmod +x run.sh
./run.sh
```
OR manually:
```bash
pip3 install flask
python3 app.py
```
Open → http://127.0.0.1:5000

---

## 📁 File Structure

```
cyberbully-ui/
├── app.py              ← Flask backend + detection logic
├── templates/
│   └── index.html      ← Full UI (Dashboard, Detect, Batch, About)
├── requirements.txt    ← Just needs: flask
├── run.bat             ← Windows launcher
├── run.sh              ← Mac/Linux launcher
└── README.md           ← This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/api/analyze` | POST | Analyze single text |
| `/api/batch` | POST | Analyze up to 20 texts |
| `/api/stats` | GET | System statistics |

### Example API Usage

```python
import requests

# Single text analysis
response = requests.post('http://127.0.0.1:5000/api/analyze',
    json={'text': 'You are so stupid!'})
print(response.json())

# Batch analysis
response = requests.post('http://127.0.0.1:5000/api/batch',
    json={'texts': ['Great post!', 'You are an idiot.', 'Thanks for sharing.']})
print(response.json())
```

---

## 🧠 Models Used

| Model | F1-Macro | CV Score | Notes |
|-------|----------|----------|-------|
| Naive Bayes | 0.824 | 0.847 | Fast baseline |
| Logistic Regression | 0.902 | 0.904 | Strong generalization |
| **Linear SVM ⭐** | **0.915** | **0.919** | Best model |
| Random Forest | 0.908 | 0.911 | Robust ensemble |

**Ensemble:** Majority vote — text flagged if ≥2 models predict bullying.

---

## 🎯 Detection Categories

- **Normal** — Regular, safe comments
- **Toxic** — General toxic language
- **Obscene** — Obscene/vulgar content
- **Insult** — Personal insults
- **Identity Hate** — Hate based on identity/group
- **Severe Toxic** — Severely toxic content
- **Threat** — Threats and intimidation

---

## 📊 UI Features

- **🔍 Detect Tab** — Real-time single text analysis with confidence scores, model votes, text features
- **📊 Dashboard** — System metrics, dataset distribution charts, model comparison
- **📋 Batch Tab** — Analyze 20 comments at once with summary statistics
- **ℹ️ About Tab** — Project details, pipeline explanation, full model comparison

---

## 🔧 To Integrate with Your Trained Models

Replace the `predict_category()` function in `app.py` with:

```python
import joblib

# Load your trained models (run cb_main.py first)
models = {
    'Naive Bayes': joblib.load('../models/naive_bayes.pkl'),
    'Logistic Regression': joblib.load('../models/logistic_regression.pkl'),
    'Linear SVM': joblib.load('../models/linear_svm.pkl'),
    'Random Forest': joblib.load('../models/random_forest.pkl'),
}
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
le = joblib.load('../models/label_encoder.pkl')
```

Then call them in the `/api/analyze` route!
