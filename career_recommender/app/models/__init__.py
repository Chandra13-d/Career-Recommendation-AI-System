# app/models/__init__.py
# Make sure this file exists and imports all model modules so SQLAlchemy registry sees exactly one class per path.
from .user import User
from .profile import Profile
from .prediction import Prediction
from .skill import Skill  # if you have this
