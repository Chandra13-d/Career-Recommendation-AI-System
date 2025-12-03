# app/api.py
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint, request, jsonify, session, redirect,
    url_for, render_template
)
from werkzeug.utils import secure_filename

from app.database import db
from app.models.user import User
from app.models.profile import Profile
from app.models.prediction import Prediction

# Services
from app.services.predictor import predict_topk, is_model_ready
from app.services.resume_parser import parse_resume
from app.services.model_manager import ModelManager


# ---------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------
api = Blueprint("api", __name__)

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
model_manager = ModelManager(MODEL_DIR)

ALLOWED_EXT = {"pdf", "docx", "txt"}


# ---------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------
@api.route("/")
def health():
    return jsonify({
        "status": "ok",
        "ml_ready": is_model_ready(),
        "app": "career_recommender"
    })


# ---------------------------------------------------------------
# API REGISTER
# ---------------------------------------------------------------
@api.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}

    email = data.get("email")
    password = data.get("password")
    full_name = data.get("full_name")

    if not email or not password:
        return jsonify({"error": "Email & password required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 400

    user = User(email=email, full_name=full_name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "registered", "user_id": user.id})


# ---------------------------------------------------------------
# API LOGIN
# ---------------------------------------------------------------
@api.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}

    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user.id
    session["user_name"] = user.full_name

    return jsonify({
        "message": "login_success",
        "redirect": "/ui/home",
        "user_id": user.id,
        "user_name": user.full_name
    })


# ---------------------------------------------------------------
# UI LOGIN (Form)
# ---------------------------------------------------------------
@api.route("/ui/login", methods=["GET", "POST"])
def ui_login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email")
    password = request.form.get("password")

    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return render_template("login.html", error="Invalid login")

    session["user_id"] = user.id
    session["user_name"] = user.full_name
    return redirect(url_for("api.ui_home"))


# ---------------------------------------------------------------
# UI REGISTER (Form)
# ---------------------------------------------------------------
@api.route("/ui/register", methods=["GET", "POST"])
def ui_register():
    if request.method == "GET":
        return render_template("register.html")

    email = request.form.get("email")
    password = request.form.get("password")
    full_name = request.form.get("full_name")

    if User.query.filter_by(email=email).first():
        return render_template("register.html", error="Email already exists")

    user = User(email=email, full_name=full_name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    session["user_name"] = user.full_name

    return redirect(url_for("api.ui_home"))


# ---------------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------------
# RESUME UPLOAD + PARSE + ML PREDICT
# ---------------------------------------------------------------
@api.route("/api/upload_resume", methods=["POST"])
def upload_resume():
    if "user_id" not in session:
        return jsonify({"error": "not_logged_in"}), 401

    if "file" not in request.files:
        return jsonify({"error": "File missing"}), 400

    f = request.files["file"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    save_path = UPLOAD_FOLDER / filename
    f.save(save_path)

    parsed = parse_resume(str(save_path))

    profile_input = {
        "technical_skills": parsed.get("skills", []),
        "certifications": [],
        "interests_domains": [],
        "raw_role": parsed.get("job_title") or ""
    }

    top3 = predict_topk(profile_input, k=3)

    # Save Profile
    profile = Profile(
        user_id=session["user_id"],
        technical_skills=",".join(parsed.get("skills", [])),
        certifications="",
        interests_domains="",
        suggested_role=top3[0]["label"] if top3 else None,
        top3_json=json.dumps(top3)
    )
    db.session.add(profile)
    db.session.commit()

    # Save Prediction
    pred = Prediction(
        user_id=session["user_id"],
        profile_id=profile.id,
        input_json=json.dumps(parsed),
        input_text=parsed.get("raw_text", ""),
        predicted_role=top3[0]["label"] if top3 else None,
        top3_json=json.dumps(top3),
        model_version="v1"
    )
    db.session.add(pred)
    db.session.commit()

    return jsonify({
        "parsed": parsed,
        "top3": top3,
        "prediction_id": pred.id
    })


# ---------------------------------------------------------------
# QUICK PREDICT (FIXED!)
# ---------------------------------------------------------------
@api.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}

    # --------------- QUICK PREDICT ---------------
    if "skills" in data:
        skills_raw = data["skills"]

        if isinstance(skills_raw, str):
            skills = [s.strip().lower() for s in skills_raw.split(",") if s.strip()]
        else:
            skills = skills_raw

        profile = {
            "technical_skills": skills,
            "certifications": [],
            "interests_domains": [],
            "raw_role": ""
        }

    # --------------- API PROFILE INPUT ---------------
    else:
        profile = data.get("profile") or data

    try:
        top3 = predict_topk(profile, k=3)
        return jsonify({"top3": top3}), 200

    except Exception as e:
        print("Quick Predict ERROR:", e)
        return jsonify({"error": "prediction_failed", "detail": str(e)}), 500


# ---------------------------------------------------------------
# USER PREDICTION HISTORY
# ---------------------------------------------------------------
@api.route("/api/user/<int:user_id>/predictions")
def user_predictions(user_id):
    preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    return jsonify([
        {
            "id": p.id,
            "predicted_role": p.predicted_role,
            "top3": json.loads(p.top3_json),
            "created_at": p.created_at.isoformat(),
        }
        for p in preds
    ])


# ---------------------------------------------------------------
# UI ROUTES
# ---------------------------------------------------------------
@api.route("/ui/home")
def ui_home():
    return render_template("home.html", ml_ready=is_model_ready())


@api.route("/ui/upload")
def ui_upload():
    if "user_id" not in session:
        return redirect(url_for("api.ui_login"))
    return render_template("upload_resume.html")


@api.route("/ui/dashboard")
def ui_dashboard():
    if "user_id" not in session:
        return redirect(url_for("api.ui_login"))
    return render_template("dashboard.html")


@api.route("/ui/about")
def ui_about():
    return render_template("about.html")


@api.route("/ui/contact")
def ui_contact():
    return render_template("contact.html")


# ---------------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------------
@api.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("api.ui_login"))
