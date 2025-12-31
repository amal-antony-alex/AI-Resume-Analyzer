import random
import pandas as pd

job_roles = {
    "Data Science": [
        "python", "pandas", "numpy", "scikit learn", "machine learning",
        "deep learning", "tensorflow", "statistics", "data analysis"
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "django",
        "node js", "flask", "rest api", "mysql"
    ],
    "Android Development": [
        "android", "kotlin", "java", "xml", "firebase",
        "android studio", "sqlite", "gradle"
    ],
    "IOS Development": [
        "swift", "xcode", "cocoa touch", "ios development",
        "objective c", "ui kit", "core data"
    ],
    "UI-UX": [
        "figma", "adobe xd", "wireframing", "prototyping",
        "user research", "usability testing", "ui design"
    ]
}

experience_templates = {
    "Fresher": [
        "recent graduate with strong foundation in {}",
        "completed academic projects using {}",
        "seeking entry level position in {}"
    ],
    "Intermediate": [
        "completed internship and worked on projects using {}",
        "hands on experience with {}",
        "1 to 2 years of experience working with {}"
    ],
    "Experienced": [
        "over 3 years of professional experience using {}",
        "led multiple projects involving {}",
        "expertise in designing scalable systems with {}"
    ]
}

def generate_resume(role, level):
    skills = random.sample(job_roles[role], k=5)
    skill_text = ", ".join(skills)
    template = random.choice(experience_templates[level])
    
    resume_text = f"""
    {template.format(role.lower())}.
    Skilled in {skill_text}.
    Worked on real world projects and collaborated with teams.
    Strong problem solving and communication skills.
    """
    return resume_text.strip()

data = []

for role in job_roles.keys():
    for level in experience_templates.keys():
        for _ in range(17):   # 5 roles × 3 levels × 17 ≈ 255 resumes
            data.append({
                "resume_text": generate_resume(role, level),
                "job_role": role,
                "experience_level": level
            })

df = pd.DataFrame(data)
df.to_csv("data/resume_dataset.csv", index=False)

print("✅ Dataset generated successfully with", len(df), "records")
