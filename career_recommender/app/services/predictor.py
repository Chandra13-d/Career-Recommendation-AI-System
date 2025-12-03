# app/services/predictor.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATA_CSV = ROOT / "data" / "mapped_roles.csv"

# Artifact filenames saved by your training script
SBERT_F = MODEL_DIR / "sbert_model.joblib"
PCA_F = MODEL_DIR / "pca.joblib"
TFIDF_F = MODEL_DIR / "tfidf.joblib"
SVD_F = MODEL_DIR / "svd.joblib"
LE_F = MODEL_DIR / "label_encoder.joblib"
# best model may be xgboost or lightgbm; we'll pick whichever exists (xgboost preferred)
BEST_XGB = MODEL_DIR / "career_best_model_xgboost.joblib"
BEST_LGB = MODEL_DIR / "career_best_model_lightgbm.joblib"
BEST_ANY = MODEL_DIR / "career_best_model_xgboost.joblib" if BEST_XGB.exists() else (BEST_LGB if BEST_LGB.exists() else None)

# Lazy-loaded globals
_sbert = None
_pca = None
_tfidf = None
_svd = None
_le = None
_model = None
_raw_lookup = None


# ----------------------------
# Utilities: load artifacts
# ----------------------------
def _load_sbert():
    global _sbert
    if _sbert is None:
        if not SBERT_F.exists():
            raise FileNotFoundError(f"Missing SBERT artifact: {SBERT_F}")
        _sbert = joblib.load(SBERT_F)
    return _sbert

def _load_pca():
    global _pca
    if _pca is None:
        if not PCA_F.exists():
            raise FileNotFoundError(f"Missing PCA artifact: {PCA_F}")
        _pca = joblib.load(PCA_F)
    return _pca

def _load_tfidf():
    global _tfidf
    if _tfidf is None:
        if not TFIDF_F.exists():
            raise FileNotFoundError(f"Missing TFIDF artifact: {TFIDF_F}")
        _tfidf = joblib.load(TFIDF_F)
    return _tfidf

def _load_svd():
    global _svd
    if _svd is None:
        if not SVD_F.exists():
            raise FileNotFoundError(f"Missing SVD artifact: {SVD_F}")
        _svd = joblib.load(SVD_F)
    return _svd

def _load_label_encoder():
    global _le
    if _le is None:
        if not LE_F.exists():
            raise FileNotFoundError(f"Missing LabelEncoder: {LE_F}")
        _le = joblib.load(LE_F)
    return _le

def _load_model():
    global _model
    if _model is None:
        # Pick the best available artifact
        if BEST_XGB.exists():
            _model = joblib.load(BEST_XGB)
        elif BEST_LGB.exists():
            _model = joblib.load(BEST_LGB)
        else:
            raise FileNotFoundError("No best model found. Expected career_best_model_xgboost.joblib or career_best_model_lightgbm.joblib in models/")
    return _model


# ----------------------------
# Raw-role lookup from CSV
# ----------------------------
def _build_raw_lookup():
    global _raw_lookup
    if _raw_lookup is None:
        if not DATA_CSV.exists():
            # fallback: empty map
            _raw_lookup = {}
            return _raw_lookup
        df = pd.read_csv(DATA_CSV, low_memory=False)
        # Expect columns: mapped_role and raw_role (if different names, try variations)
        if "mapped_role" not in df.columns and "mapped" in df.columns:
            df = df.rename(columns={c:c for c in df.columns})
        if "mapped_role" not in df.columns:
            # try common alternatives
            for c in df.columns:
                if "mapped" in c:
                    df = df.rename(columns={c: "mapped_role"})
                    break
        if "raw_role" not in df.columns:
            # try "raw" or "raw_role_title"
            for c in df.columns:
                if "raw" in c:
                    df = df.rename(columns={c: "raw_role"})
                    break

        # ensure both columns exist
        if "mapped_role" in df.columns and "raw_role" in df.columns:
            grp = df.groupby("mapped_role")["raw_role"].apply(lambda s: list(pd.Series(s).dropna().unique())).to_dict()
            _raw_lookup = grp
        else:
            # fallback: return mapping where mapped_role -> [mapped_role]
            if "mapped_role" in df.columns:
                uniques = df["mapped_role"].dropna().unique().tolist()
                _raw_lookup = {m: [m] for m in uniques}
            else:
                _raw_lookup = {}
    return _raw_lookup


# ----------------------------
# Feature pipeline (exactly like training)
# ----------------------------
def _build_profile_text(profile: Dict[str, Any]) -> str:
    """
    Combine fields into the single text string used during training.
    Accepts profile with keys: technical_skills (list), certifications (list), interests_domains (list), raw_role (str).
    """
    skills = profile.get("technical_skills") or profile.get("skills") or []
    if isinstance(skills, str):
        skills = [skills]
    certs = profile.get("certifications") or []
    if isinstance(certs, str):
        certs = [certs]
    interests = profile.get("interests_domains") or []
    if isinstance(interests, str):
        interests = [interests]
    raw_role = profile.get("raw_role") or profile.get("job_title") or ""
    parts = []
    if skills:
        parts.append(" ".join([str(x) for x in skills if str(x).strip()]))
    if certs:
        parts.append(" ".join([str(x) for x in certs if str(x).strip()]))
    if interests:
        parts.append(" ".join([str(x) for x in interests if str(x).strip()]))
    if raw_role:
        parts.append(str(raw_role))
    text = " ".join(parts).strip()
    if not text:
        # last-resort: entire profile dict
        text = " ".join([str(v) for v in profile.values() if v])
    return text


def _build_features_from_text(text: str) -> np.ndarray:
    """
    Returns a 1D numpy array of combined features (emb_pca concat tfidf_svd)
    """
    sbert = _load_sbert()
    pca = _load_pca()
    tfidf = _load_tfidf()
    svd = _load_svd()

    # SBERT embed
    emb = sbert.encode([text], show_progress_bar=False)
    emb = np.asarray(emb)
    emb_pca = pca.transform(emb)  # shape (1, 128)

    # TFIDF -> SVD
    tf = tfidf.transform([text])
    tf_svd = svd.transform(tf)  # shape (1, 50)

    X = np.hstack([emb_pca, tf_svd])  # shape (1, 178)
    return X.flatten()


def _align_proba(proba: np.ndarray, model_classes: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Align model probabilities to 0..num_classes-1 columns. (same as training's align_proba_to_classes)
    """
    n_samples = proba.shape[0]
    if proba.shape[1] == num_classes and np.array_equal(model_classes, np.arange(num_classes)):
        return proba
    aligned = np.zeros((n_samples, num_classes), dtype=float)
    # model_classes might be ints (0..n-1) or str labels — convert if possible
    for i, cls in enumerate(model_classes):
        try:
            idx = int(cls)
        except Exception:
            # if classes are strings from label encoder, we try to map via label encoder later
            idx = None
        if idx is not None and 0 <= idx < num_classes:
            aligned[:, idx] = proba[:, i]
        else:
            # fallback: put into same index i if within range
            if i < num_classes:
                aligned[:, i] = proba[:, i]
    return aligned


# ----------------------------
# Main predict_topk API
# ----------------------------
def predict_topk(profile_dict: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """
    profile_dict example:
    {
      "technical_skills": ["python","sql"],
      "certifications": ["aws"],
      "interests_domains": ["data"],
      "raw_role": "Data Analyst"
    }
    Returns: [ { "label": mapped_label, "score": prob, "raw_roles": [...] }, ... ]
    """
    # load all required artifacts lazily
    le = _load_label_encoder()
    model = _load_model()
    raw_map = _build_raw_lookup()

    # build text and features
    text = _build_profile_text(profile_dict)
    features = _build_features_from_text(text).reshape(1, -1)  # shape (1, n_features)

    # predict_proba
    try:
        proba = model.predict_proba(features)
    except Exception as e:
        # some models expect 2D input with dtype float32
        try:
            proba = model.predict_proba(np.asarray(features, dtype=np.float32))
        except Exception as e2:
            raise RuntimeError(f"Model prediction failed: {e} | {e2}")

    # align probabilities in case model returned less/more columns
    num_classes = len(le.classes_)
    model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    proba_aligned = _align_proba(proba, model_classes, num_classes)

    top_idxs = np.argsort(proba_aligned[0])[::-1][:k]

    results = []
    for idx in top_idxs:
        # if label encoder maps 0..n-1 to mapped_role strings
        try:
            mapped_label = le.inverse_transform([idx])[0]
        except Exception:
            # fallback if label encoder uses different ordering
            mapped_label = str(idx)
        score = float(proba_aligned[0, idx])
        raw_roles = raw_map.get(mapped_label, [])
        # ensure list and limit to 5
        if isinstance(raw_roles, (list, tuple)):
            raw_roles_trim = raw_roles[:5]
        else:
            raw_roles_trim = [raw_roles] if raw_roles else []
        results.append({
            "label": mapped_label,
            "score": score,
            "raw_roles": raw_roles_trim
        })

    return results


# Small helper so other modules can check model readiness
def is_model_ready() -> bool:
    try:
        _load_label_encoder()
        _load_model()
        _load_sbert()
        _load_pca()
        _load_tfidf()
        _load_svd()
        return True
    except Exception:
        return False
