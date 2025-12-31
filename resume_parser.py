import re
import spacy
from pathlib import Path
from pdfminer.high_level import extract_text

nlp = spacy.load("en_core_web_sm")

SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "data science", "tensorflow", "pytorch",
    "sql", "mysql", "mongodb", "html", "css", "javascript",
    "react", "django", "flask", "streamlit", "git", "docker"
]

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group(0) if match else None

def extract_skills(text):
    text = text.lower()
    return list(set(skill for skill in SKILLS_DB if skill in text))

def parse_resume(pdf_path):
    if not Path(pdf_path).exists():
        return None

    text = extract_text(pdf_path)
    doc = nlp(text)

    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return {
        "name": name,
        "email": extract_email(text),
        "mobile_number": extract_phone(text),
        "skills": extract_skills(text),
        "degree": None,
        "no_of_pages": text.count("\f") + 1
    }
