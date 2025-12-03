# 🚀 Career Recommendation AI System

AI-powered full-stack web application that analyzes resumes, extracts skills, and predicts the top 3 career roles using advanced machine learning & NLP.

Built with Flask + SBERT + XGBoost + LightGBM + HuggingFace NER + MySQL + Bootstrap UI.

🔥 Features
✅ AI Capabilities

Resume parsing (PDF, DOCX, TXT)
HuggingFace NER (Name / Org Extraction)
Skill extraction using NLP + smart keyword rules
ML-based role prediction using:
SBERT semantic embeddings
TF-IDF + SVD dimensionality
PCA reduction
XGBoost & LightGBM ensemble

90%+ accuracy from custom career-role dataset (mapped_roles.csv)

✅ Full-Stack Web App

Secure Login / Register with SQLAlchemy
User dashboard showing prediction history
Upload resume → instant results with detailed breakdown
“Quick Predict” (enter skills → get roles)
Admin-style model versioning & retraining support
Modern UI with gradients, animations, icons

✅ Database Features

User profile table
Prediction history table
Resume extraction saved
Skill + role mapping
Job cache table (future add-on)

🧠 Architecture

career_recommender/
│── app/
│   ├── api.py                 # Main backend routes / APIs
│   ├── models/                # SQLAlchemy models
│   ├── services/
│   │     ├── predictor.py     # ML prediction engine
│   │     ├── resume_parser.py # NLP resume parser
│   │     ├── model_manager.py # Model version control
│   ├── templates/             # HTML frontend pages
│   ├── static/                # CSS, JS, images
│   ├── __init__.py
│   └── run_app.py
│
│── models/                    # Saved ML models
│── scripts/  
│   ├── 02_train_rebuilt_model.py  # Training pipeline
│
│── uploads/                   # Resume uploads
│── data/                      # mapped_roles.csv
│── .env                       # Environment config
│── README.md                  # (you are here)
└── requirements.txt

🏗️ Tech Stack
Backend

Python (Flask)
SQLAlchemy ORM
MySQL / MariaDB
Joblib (model storage)

Machine Learning

Sentence-Transformers (SBERT)
Scikit-Learn (PCA, SVD, TF-IDF)
XGBoost + LightGBM
HuggingFace Transformers (NER)

Frontend

Bootstrap 5
JavaScript (Fetch API)
Chart.js (Dashboard graphs)

📦 Required Datasets

prediction-data.csv
canonical_skills.csv
job_skills.csv
profiles_clean.csv
new_prediction-data.csv

📦 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/career-recommender-ai.git
cd career-recommender-ai

2️⃣ Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set up MySQL

Create database:

CREATE DATABASE career_recommender;


Run schema:

USE career_recommender;
-- (Paste provided schema tables here)

5️⃣ Create .env file
SQLALCHEMY_DATABASE_URI=mysql+pymysql://root:password@localhost/career_recommender
UPLOAD_FOLDER=uploads
SECRET_KEY=your-secret

6️⃣ Run the app
python -m app.run_app


Visit → http://127.0.0.1:5000/ui/home

🎯 Machine Learning Model

Training script:

scripts/02_train_rebuilt_model.py


This script:

Loads mapped_roles.csv
Generates SBERT embeddings
Creates combined feature vectors
Trains LightGBM + XGBoost

Saves:

best model
PCA, TF-IDF, SVD

label encoder

confusion matrix

metrics.json

Model files saved to /models/.

📊 Dashboard Preview

✔ Prediction history
✔ Top-3 roles
✔ Raw matched job titles
✔ Date/time
✔ Charts for matches


📁 API Endpoints
🔹 POST /api/predict

Predict role from manual skill input

🔹 POST /api/upload_resume

Upload resume → parse → predict

🔹 GET /api/user/<id>/predictions

History of predictions

🔹 GET /api/models

List ML models

🙌 Screenshots 
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-4.png)
![alt text](image-3.png)

## 🎓 Future Enhancements

Job recommendation from LinkedIn / Indeed scraping
Salary prediction
Career roadmap generator
AI-based course recommendations
Real-time embeddings using OpenAI API (optional)

## 💡 Author

Chandra D
📍 Mysore, India
📧 preethamchandu1308@gmail.com
💼 Data Science & AI Engineer

# ⭐ If this project helped you — don’t forget to star the repo!
