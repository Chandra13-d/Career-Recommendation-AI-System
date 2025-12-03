# app/models/prediction.py
# app/models/prediction.py
from app.database import db
from datetime import datetime

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("profiles.id"), nullable=True)
    input_json = db.Column(db.Text)
    input_text = db.Column(db.Text, nullable=True)
    predicted_role = db.Column(db.String(256))
    top3_json = db.Column(db.Text)
    model_version = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
