# scripts/01a_clean_skills.py
import pandas as pd
import re
from pathlib import Path
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN = DATA / "new_prediction-data.csv"
CANON = DATA / "canonical_skills.csv"
OUT = DATA / "skills_normalized.csv"


def safe_read(path):
    """Try utf-8 first, then latin-1 fallback."""
    try:
        return pd.read_csv(path, low_memory=False, encoding='utf-8')
    except Exception:
        print("[WARN] UTF-8 failed. Retrying in latin-1...")
        return pd.read_csv(path, low_memory=False, encoding='latin-1')


def normalize_token(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().replace("\xa0", " ")
    s = re.sub(r'[^a-z0-9+#\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def split_skills(s):
    return [
        normalize_token(x)
        for x in re.split(r'[;,|/]', str(s))
        if x.strip() != ''
    ]


def build_canonical():
    if not CANON.exists():
        print("[WARN] canonical_skills.csv not found.")
        return []

    df = safe_read(CANON)
    cols = [c for c in df.columns if df[c].dtype == object]
    col = cols[0] if cols else df.columns[0]

    canon = [
        normalize_token(x)
        for x in df[col].dropna().unique()
    ]

    return canon


def map_skills(tokens, canon):
    mapped = []
    for t in tokens:
        if not canon:
            mapped.append(t)
            continue

        match, score, _ = process.extractOne(t, canon, scorer=fuzz.token_sort_ratio)
        if score >= 85:
            mapped.append(match)
        else:
            mapped.append(t)

    return ';'.join(sorted(set(mapped)))


def main():
    print("Loading", IN)
    df = safe_read(IN)
    print("Rows:", len(df))

    canon = build_canonical()
    print("Canonical skills loaded:", len(canon))

    # auto-detect skills column
    skill_cols = [c for c in df.columns if "skill" in c.lower()]
    if skill_cols:
        skill_col = skill_cols[0]
    else:
        skill_col = df.columns[0]  # fallback
        print("[WARN] No skills column found. Using:", skill_col)

    df['skills_norm'] = df[skill_col].fillna('').apply(
        lambda s: map_skills(split_skills(s), canon)
    )

    df.to_csv(OUT, index=False)
    print("Saved normalized file ->", OUT)


if __name__ == "__main__":
    main()

