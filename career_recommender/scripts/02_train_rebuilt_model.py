# scripts/02_train_rebuilt_model.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.utils.class_weight import compute_class_weight

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Paths: prefer uploaded mapped_roles_1.csv if present
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
UPLOADED = Path("/mnt/data/mapped_roles_1.csv")
FALLBACK = ROOT / "data" / "mapped_roles.csv"
if UPLOADED.exists():
    DATA = UPLOADED
else:
    DATA = FALLBACK

MODELDIR = ROOT / "models"
MODELDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def safe_read(path: Path):
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        print("[WARN] UTF-8 failed -> trying latin-1")
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def build_profile_text(df: pd.DataFrame):
    # choose columns that look like skills/interests/certificates
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ("skill", "interest", "cert", "raw_role", "summary", "description"))]
    if not text_cols:
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()[:1]
    print("Using text columns for profile_text:", text_cols)
    df["profile_text"] = df[text_cols].astype(str).apply(lambda r: " ".join([str(x) for x in r.values if str(x).strip() != ""]), axis=1)
    return df


def reduce_and_build_features(df: pd.DataFrame, sbert_model_name="all-MiniLM-L6-v2"):
    texts = df["profile_text"].astype(str).tolist()
    n = len(texts)
    print(f"Computing SBERT embeddings for {n} profiles with model={sbert_model_name} ...")
    sbert = SentenceTransformer(sbert_model_name)
    emb = sbert.encode(texts, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb)
    print("SBERT embeddings shape:", emb.shape)

    # PCA reduce SBERT -> 128 dims
    print("Reducing SBERT embeddings via PCA -> 128 dims")
    pca = PCA(n_components=128, random_state=42)
    emb_reduced = pca.fit_transform(emb)
    print("Embeddings reduced shape:", emb_reduced.shape)

    # TF-IDF -> TruncatedSVD -> 50 dims
    print("Building TF-IDF (max_features=3000, ngram 1-2) and reducing to 50 dims via TruncatedSVD")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts)
    print("TF-IDF sparse shape:", X_tfidf.shape)

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    print("TF-IDF reduced shape:", X_svd.shape)

    # combine
    X = np.hstack([emb_reduced, X_svd])
    print("Combined feature shape:", X.shape)

    # save artifacts
    joblib.dump(sbert, MODELDIR / "sbert_model.joblib")
    joblib.dump(pca, MODELDIR / "pca.joblib")
    joblib.dump(tfidf, MODELDIR / "tfidf.joblib")
    joblib.dump(svd, MODELDIR / "svd.joblib")
    print("Saved feature artifacts to models/")

    return X


def align_proba_to_classes(proba: np.ndarray, model_classes: np.ndarray, num_classes: int):
    """
    Ensure proba has shape (n_samples, num_classes) and columns correspond to [0..num_classes-1].
    model_classes is array of class labels model used (e.g. model.classes_).
    If proba already matches, return it. Otherwise create zero matrix and map columns.
    """
    n_samples = proba.shape[0]
    if proba.shape[1] == num_classes and np.array_equal(model_classes, np.arange(num_classes)):
        return proba  # already aligned

    aligned = np.zeros((n_samples, num_classes), dtype=float)
    # model_classes are integers (0..num_classes-1) after our LabelEncoder; map accordingly
    for i, cls in enumerate(model_classes):
        if 0 <= int(cls) < num_classes:
            aligned[:, int(cls)] = proba[:, i]
    return aligned


# -----------------------------
# Trainers
# -----------------------------
def train_xgb(X_train, y_train, X_val, y_val, sample_weight_train, num_classes):
    print("Training XGBoost (no SMOTE) with sample weights...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=400,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        use_label_encoder=False
    )

    # Fit (do not pass eval params that older xgboost may reject)
    model.fit(X_train, y_train, sample_weight=sample_weight_train, verbose=False)

    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    # align proba columns using model.classes_
    proba_aligned = align_proba_to_classes(proba, getattr(model, "classes_", np.arange(proba.shape[1])), num_classes)
    return model, preds, proba_aligned


def train_lgb(X_train, y_train, X_val, y_val, num_classes):
    print("Training LightGBM (class_weight='balanced')...")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=150,
        max_depth=-1,
        min_data_in_leaf=20,
        min_child_samples=20,
        min_split_gain=0.0,
        class_weight="balanced",
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    # lightgbm usually returns aligned probabilities for encoded classes 0..num_classes-1
    proba_aligned = align_proba_to_classes(proba, getattr(model, "classes_", np.arange(proba.shape[1])), num_classes)
    return model, preds, proba_aligned


def plot_and_save_confusion(y_true, y_pred, le, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved confusion matrix ->", out_path)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading dataset:", DATA)
    df = safe_read(DATA)

    # Accept either 'mapped_role' or fallback name
    if "mapped_role" not in df.columns and "mapped_category" in df.columns:
        df = df.rename(columns={"mapped_category": "mapped_role"})

    if "mapped_role" not in df.columns:
        raise ValueError("Please ensure your CSV contains 'mapped_role' column.")

    df = df.dropna(subset=["mapped_role"])
    print("Rows available:", len(df))

    # Build profile text
    df = build_profile_text(df)

    # initial label encode just to inspect classes
    le = LabelEncoder()
    y_initial = le.fit_transform(df["mapped_role"].astype(str))
    print("Initial number of classes:", len(le.classes_))

    # Build features for the full dataset
    X_full = reduce_and_build_features(df)

    # Remove classes with <2 samples to avoid stratify/split issues
    vc = pd.Series(y_initial).value_counts()
    valid_classes = vc[vc >= 2].index
    mask = np.isin(y_initial, valid_classes)
    df_clean = df[mask].reset_index(drop=True)
    X = X_full[mask]
    print("After removing rarity classes -> rows:", len(df_clean))

    # Re-encode labels after filtering (important!)
    le = LabelEncoder()
    y = le.fit_transform(df_clean["mapped_role"].astype(str))
    joblib.dump(le, MODELDIR / "label_encoder.joblib")
    print("Remaining classes (re-encoded):", len(le.classes_))

    num_classes = len(le.classes_)

    # Train / val split (stratify on new y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # compute sample weight for XGBoost
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {c: w for c, w in zip(classes, class_weights)}
    sample_weight_train = np.array([weight_map[c] for c in y_train])
    print("Computed class weights for training.")

    # Train XGBoost
    xgb_model, xgb_pred, xgb_proba = train_xgb(X_train, y_train, X_val, y_val, sample_weight_train, num_classes)

    # Train LightGBM
    lgb_model, lgb_pred, lgb_proba = train_lgb(X_train, y_train, X_val, y_val, num_classes)

    # Compute metrics (use labels param to avoid mismatch)
    labels_range = np.arange(num_classes)

    xgb_acc = accuracy_score(y_val, xgb_pred)
    try:
        xgb_top3 = top_k_accuracy_score(y_val, xgb_proba, k=3, labels=labels_range)
    except Exception as e:
        print("top_k for XGB failed:", e)
        xgb_top3 = None

    lgb_acc = accuracy_score(y_val, lgb_pred)
    try:
        lgb_top3 = top_k_accuracy_score(y_val, lgb_proba, k=3, labels=labels_range)
    except Exception as e:
        print("top_k for LGB failed:", e)
        lgb_top3 = None

    print(f"\nXGBoost → Acc={xgb_acc:.4f} | Top-3={(f'{xgb_top3:.4f}' if xgb_top3 is not None else 'NA')}")
    print("XGBoost classification report:\n", classification_report(y_val, xgb_pred, zero_division=0))

    print(f"\nLightGBM → Acc={lgb_acc:.4f} | Top-3={(f'{lgb_top3:.4f}' if lgb_top3 is not None else 'NA')}")
    print("LightGBM classification report:\n", classification_report(y_val, lgb_pred, zero_division=0))

    # Select best by accuracy
    if lgb_acc > xgb_acc:
        best_model, best_name, best_pred = lgb_model, "lightgbm", lgb_pred
    else:
        best_model, best_name, best_pred = xgb_model, "xgboost", xgb_pred

    # Save artifacts
    joblib.dump(xgb_model, MODELDIR / "xgb_model.joblib")
    joblib.dump(lgb_model, MODELDIR / "lgb_model.joblib")
    joblib.dump(best_model, MODELDIR / f"career_best_model_{best_name}.joblib")
    print(f"Saved models, best = {best_name}")

    # Save confusion matrix
    plot_and_save_confusion(y_val, best_pred, le, MODELDIR / "dashboard_confusion_matrix.png")

    # Save a small metrics JSON
    metrics = {
        "xgb_acc": float(xgb_acc),
        "xgb_top3": float(xgb_top3) if xgb_top3 is not None else None,
        "lgb_acc": float(lgb_acc),
        "lgb_top3": float(lgb_top3) if lgb_top3 is not None else None,
        "n_samples": int(len(df_clean)),
        "n_classes": int(num_classes)
    }
    (MODELDIR / "training_metrics.json").write_text(str(metrics))
    print("Saved metrics ->", MODELDIR / "training_metrics.json")

    print("\nTraining complete. Artifacts saved to:", MODELDIR)


if __name__ == "__main__":
    main()
