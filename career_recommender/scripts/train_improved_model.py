import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mapped_prediction_data.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
def load_data():
    print(f"Loading dataset: {DATA}")
    try:
        df = pd.read_csv(DATA, encoding="utf-8")
    except:
        df = pd.read_csv(DATA, encoding="latin-1")

    print(df.head())
    print("Shape:", df.shape)
    return df


# -------------------------------------------------------------------
# MAKE TEXT FEATURES
# -------------------------------------------------------------------
def make_text_features(df):
    print("\n[STEP] Building SBERT embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["clean_text"].astype(str).fillna("").tolist()
    emb = model.encode(texts, batch_size=128, show_progress_bar=True)
    print("Embeddings:", emb.shape)

    # Dimensionality reduction using PCA (SBERT → 128 dims)
    print("\nReducing embeddings with PCA...")
    pca = PCA(n_components=128, random_state=42)
    emb_128 = pca.fit_transform(emb)
    print("Reduced embeddings:", emb_128.shape)

    # Build TF-IDF
    print("\nBuilding TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(texts)
    print("TF-IDF:", X_tfidf.shape)

    # Reduce TF-IDF using SVD (50 dims)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    print("TF-IDF Reduced:", X_svd.shape)

    # Final combined feature
    X = np.hstack([emb_128, X_svd])
    print("\nFinal feature shape:", X.shape)

    # Save vectorizers
    joblib.dump(model, OUT / "sbert_model.joblib")
    joblib.dump(tfidf, OUT / "tfidf.joblib")
    joblib.dump(svd, OUT / "svd.joblib")
    joblib.dump(pca, OUT / "pca.joblib")

    return X


# -------------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------------
def train_model(X, y, label_encoder):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTrain:", X_train.shape, " | Val:", X_val.shape)

    # XGBoost – WORKS BEST FOR TEXT EMBEDDINGS
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        n_estimators=600,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    top3 = top_k_accuracy_score(y_val, y_proba, k=3)

    print(f"\nVal Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}\n")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Save model
    joblib.dump(model, OUT / "xgb_final_model.joblib")
    print(f"Model saved → {OUT / 'xgb_final_model.joblib'}")

    # Plot confusion matrix
    cm = pd.crosstab(y_val, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(OUT / "confusion_matrix_final.png")
    plt.close()

    print("Saved dashboard → confusion_matrix_final.png")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df = load_data()

    # Target label
    label_col = "mapped_category"
    df = df[df[label_col].notna()]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    joblib.dump(le, OUT / "label_encoder.joblib")
    print("Label encoder saved:", list(le.classes_))

    # Build features
    X = make_text_features(df)

    # Train model
    train_model(X, y, le)


if __name__ == "__main__":
    main()
