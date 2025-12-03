# app/models/profile.py
from app.database import db
from datetime import datetime

class Profile(db.Model):
    __tablename__ = "profiles"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    technical_skills = db.Column(db.Text)
    certifications = db.Column(db.Text)
    interests_domains = db.Column(db.Text)
    suggested_role = db.Column(db.String(200))
    top3_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
