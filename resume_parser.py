# resume_parser.py
import re
from pdfminer.high_level import extract_text

def parse_resume(file_path):
    text = extract_text(file_path)

    # Extract email
    email_match = re.search(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        text
    )
    email = email_match.group(0) if email_match else "Not Found"

    # Extract name (first non-empty line)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    name = lines[0] if lines else "Not Found"

    # Basic skill extraction (extendable)
    skill_keywords = [
        "python", "java", "machine learning", "data science",
        "deep learning", "sql", "javascript", "html", "css",
        "react", "django", "flask", "tensorflow", "pandas",
        "numpy", "opencv", "aws", "git"
    ]

    skills = sorted({
        skill.title()
        for skill in skill_keywords
        if skill in text.lower()
    })

    return {
        "name": name,
        "email": email,
        "skills": skills
    }
