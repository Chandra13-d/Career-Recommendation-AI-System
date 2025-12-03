import os
from flask import Flask
from flask_session import Session
from dotenv import load_dotenv
from app.database import db

def create_app():
    load_dotenv()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Session config
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)

    # init DB
    db.init_app(app)

    # import models so tables are known
    from app.models import user, profile, prediction  # noqa: F401

    with app.app_context():
        db.create_all()

    # register routes blueprint
    from app.api import api
    app.register_blueprint(api)

    return app
