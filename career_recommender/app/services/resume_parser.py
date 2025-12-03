import re
import pdfplumber
import docx2txt

def extract_text(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    if lower.endswith(".docx"):
        return docx2txt.process(path) or ""
    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

# simple skill list - expand as needed
SKILL_DB = [
    "python", "java", "c", "c++", "sql", "excel", "power bi", "aws",
    "azure", "ml", "ai", "machine learning", "deep learning", "nlp",
    "tableau", "javascript", "html", "css", "flask", "django", "pandas"
]

def extract_skills(text):
    t = text.lower()
    found = []
    for s in SKILL_DB:
        if s in t and s not in found:
            found.append(s)
    return found

def extract_contact(text):
    emails = re.findall(r"[a-zA-Z0-9\._%+-]+@[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}", text)
    phones = re.findall(r"\b[6-9]\d{9}\b", text)
    linkedin = re.findall(r"(https?://[^\s]*linkedin[^\s]*)", text)
    github = re.findall(r"(https?://[^\s]*github[^\s]*)", text)
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedin[0] if linkedin else None,
        "github": github[0] if github else None
    }

def extract_education(text):
    keys = ["btech", "b.e", "bsc", "b.sc", "bca", "mtech", "msc", "m.sc", "mba", "mca", "degree", "phd"]
    lines = text.splitlines()
    edu = [l.strip() for l in lines if any(k in l.lower() for k in keys)]
    return edu

def extract_experience(text):
    m = re.search(r"(\d+)\+?\s*years?", text.lower())
    return (m.group() if m else None)

def parse_resume(path):
    raw_text = extract_text(path)
    skills = extract_skills(raw_text)
    contact = extract_contact(raw_text)
    education = extract_education(raw_text)
    experience = extract_experience(raw_text)

    # Heuristic name: first non-empty short line
    name = None
    for line in raw_text.splitlines():
        l = line.strip()
        if l and len(l) < 60:
            # avoid lines that are like "email:" etc
            if "@" not in l and any(c.isalpha() for c in l):
                name = l
                break

    ai_summary = f"Candidate {name or 'Unknown'} - Top skills: {', '.join(skills[:6]) or 'N/A'}."

    return {
        "name": name,
        "title": None,
        "skills": skills,
        "experience": experience,
        "education": education,
        "contact": contact,
        "raw_text": raw_text,
        "ai_summary": ai_summary
    }
