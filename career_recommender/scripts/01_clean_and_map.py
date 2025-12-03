# scripts/01_clean_and_map.py
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN = DATA / "new_prediction-data.csv"
OUT = DATA / "mapped_roles.csv"


def safe_read(path):
    """Try utf-8 first, then latin-1."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except:
        print("[WARN] UTF-8 failed → using latin-1")
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("\xa0", " ")
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# -----------------------------
# FIXED: 10 Career Categories
# -----------------------------
JOB_CATEGORIES = {
    "software_engineering": [
        "developer", "engineer", "programmer",
        "full stack", "backend", "frontend",
        "app developer", "mobile developer", "software"
    ],
    "data_ai_ml": [
        "data", "machine learning", "ml", "ai",
        "analyst", "data scientist", "data engineer"
    ],
    "cybersecurity": [
        "security", "cyber", "ethical hacking", "penetration"
    ],
    "finance": [
        "finance", "accountant", "banking", "investment"
    ],
    "marketing": [
        "marketing", "seo", "digital marketing", "content"
    ],
    "hr_admin": [
        "hr", "human resource", "recruiter", "talent"
    ],
    "education": [
        "teacher", "trainer", "faculty", "tutor"
    ],
    "design_creative": [
        "designer", "graphics", "ui", "ux", "creative"
    ],
    "operations_management": [
        "operations", "manager", "project manager",
        "admin", "coordinator"
    ],
    "other": []
}


def map_role(raw_role):
    r = clean_text(raw_role)

    for cat, keywords in JOB_CATEGORIES.items():
        for kw in keywords:
            if kw in r:
                return cat

    return "other"


def main():
    print("Loading:", IN)
    df = safe_read(IN)
    print("Rows:", len(df))

    # auto-detect role column
    role_cols = [c for c in df.columns if "role" in c.lower() or "job" in c.lower()]
    role_col = role_cols[0] if role_cols else df.columns[-1]

    print("Using role column:", role_col)

    df["raw_role"] = df[role_col].astype(str)
    df["mapped_role"] = df["raw_role"].apply(map_role)

    df.to_csv(OUT, index=False)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
