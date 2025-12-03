import os

# -------------------------------------
# YOUR PROJECT ROOT FOLDER NAME
# -------------------------------------
ROOT = "career_recommender"

# Folder structure definition
folders = [
    f"{ROOT}/app",
    f"{ROOT}/app/models",
    f"{ROOT}/app/services",
    f"{ROOT}/app/templates",
    f"{ROOT}/app/static",
    f"{ROOT}/app/static/css",
    f"{ROOT}/app/static/js",
    f"{ROOT}/app/static/images",
    f"{ROOT}/scripts",
    f"{ROOT}/data",
    f"{ROOT}/data/processed",
    f"{ROOT}/data/powerbi",
    f"{ROOT}/models",
    f"{ROOT}/logs",
]

# Placeholder files to create
files = {
    f"{ROOT}/app/__init__.py": "",
    f"{ROOT}/app/api.py": "# Flask API routes will be added here\n",
    f"{ROOT}/app/run_app.py": """from flask import Flask
from app.api import api
from app.database import init_db

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'
    init_db(app)
    app.register_blueprint(api)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
""",
    f"{ROOT}/app/database.py": """from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

load_dotenv()

db = SQLAlchemy()

def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
""",
    f"{ROOT}/app/services/predictor.py": "# ML Model loading + Career prediction logic\n",
    f"{ROOT}/app/services/resume_parser.py": "# Resume NLP extraction logic will go here\n",
    f"{ROOT}/app/models/user.py": "# SQLAlchemy User model\n",
    f"{ROOT}/app/models/profile.py": "# SQLAlchemy Profile model\n",
    f"{ROOT}/app/models/prediction.py": "# SQLAlchemy Prediction model\n",
    f"{ROOT}/app/templates/home.html": "<!-- Home Page -->\n",
    f"{ROOT}/app/templates/upload_resume.html": "<!-- Resume Upload Page -->\n",
    f"{ROOT}/app/templates/result.html": "<!-- Result Page -->\n",
    f"{ROOT}/app/static/css/style.css": "/* Main CSS */\n",
    f"{ROOT}/app/static/js/main.js": "// Main JS\n",
    f"{ROOT}/scripts/train_improved_model.py": "# ML training script\n",
    f"{ROOT}/scripts/export_powerbi.py": "# PowerBI export script\n",
    f"{ROOT}/scripts/fix_csv.py": "# CSV cleaner script\n",
    f"{ROOT}/.env": "DATABASE_URL=mysql+pymysql://root:password@localhost/career_recommender\nSECRET_KEY=your-secret-key\n",
}

# -------------------------------------
# Create folders
# -------------------------------------
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# -------------------------------------
# Create placeholder files
# -------------------------------------
for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("\n🎉 Project Structure Created Successfully!")
print(f"📁 Root folder: {ROOT}")

