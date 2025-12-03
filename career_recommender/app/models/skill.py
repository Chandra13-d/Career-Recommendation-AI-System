# app/models/skill.py
from app.database import db
from datetime import datetime

class Skill(db.Model):
    __tablename__ = "skills"
    id = db.Column(db.Integer, primary_key=True)
    skill_name = db.Column(db.String(200), unique=True)
    canonical = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
