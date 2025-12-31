# ================== IMPORTS ==================
import streamlit as st
import os, io, random
import nltk
import joblib
import pandas as pd
import plotly.express as px
from pdfminer.high_level import extract_text
from resume_parser import parse_resume
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter

from Courses import (
    ds_course, web_course, android_course,
    ios_course, uiux_course,
    resume_videos, interview_videos
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== NLTK SETUP ==================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# ================== LOAD ML MODELS ==================
job_role_model = joblib.load("models/job_role_model.pkl")
job_role_vectorizer = joblib.load("models/job_role_vectorizer.pkl")

exp_model = joblib.load("models/experience_model.pkl")
exp_vectorizer = joblib.load("models/experience_vectorizer.pkl")

# ================== PDF TEXT READER ==================
def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh):
            interpreter.process_page(page)

    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

# ================== ML PREDICTIONS ==================
def predict_with_ml(resume_text):
    role_vec = job_role_vectorizer.transform([resume_text])
    role = job_role_model.predict(role_vec)[0]
    role_conf = max(job_role_model.predict_proba(role_vec)[0]) * 100

    exp_vec = exp_vectorizer.transform([resume_text])
    exp = exp_model.predict(exp_vec)[0]
    exp_conf = max(exp_model.predict_proba(exp_vec)[0]) * 100

    return role, role_conf, exp, exp_conf

# ================== JD MATCHING ==================
def jd_resume_match(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def skill_gap_analysis(resume_skills, jd_text):
    jd_text = jd_text.lower()
    matched, missing = [], []

    for skill in resume_skills:
        if skill.lower() in jd_text:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ================== MAIN APP ==================
def run():
    st.title("🤖 AI Resume Analyzer")
    choice = st.sidebar.selectbox("Menu", ["User", "About"])

    if choice == "User":
        st.header("Upload Resume & Job Description")

        job_desc = st.text_area(
            "📄 Paste Job Description",
            height=200,
            placeholder="Paste JD from LinkedIn / Naukri / Indeed"
        )

        pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        if pdf_file:
            # Save resume
            upload_dir = "Uploaded_Resumes"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, pdf_file.name)

            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Parse resume
            resume_data = parse_resume(save_path)
            if not resume_data:
                st.error("Resume parsing failed")
                return

            resume_text = pdf_reader(save_path)

            # ML Predictions
            role, role_conf, exp, exp_conf = predict_with_ml(resume_text)

            st.subheader("🤖 AI Predictions")
            col1, col2 = st.columns(2)
            col1.success(f"Predicted Job Role: {role} ({role_conf:.2f}%)")
            col2.success(f"Experience Level: {exp} ({exp_conf:.2f}%)")

            st.subheader("📌 Extracted Resume Details")
            st.write("**Name:**", resume_data.get("name"))
            st.write("**Email:**", resume_data.get("email"))
            st.write("**Skills:**", resume_data.get("skills"))

            # ================= STEP 4: JD MATCH =================
            if job_desc.strip():
                st.subheader("📊 Resume vs Job Description Analysis")

                match_score = jd_resume_match(resume_text, job_desc)
                st.metric("Resume–JD Match Percentage", f"{match_score}%")

                st.progress(int(match_score))

                # Match interpretation
                if match_score < 40:
                    st.error("🔴 Low Match – Resume needs significant improvement")
                elif match_score < 70:
                    st.warning("🟡 Moderate Match – Some skill gaps identified")
                else:
                    st.success("🟢 Strong Match – Resume is well aligned")

                matched, missing = skill_gap_analysis(
                    resume_data.get("skills", []),
                    job_desc
                )

                # Skill coverage
                total_skills = len(matched) + len(missing)
                coverage = round((len(matched) / total_skills) * 100, 2) if total_skills else 0
                st.metric("Skill Coverage", f"{coverage}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Matched Skills")
                    st.write(matched if matched else "None")

                with col2:
                    st.subheader("❌ Missing Skills")
                    st.write(missing if missing else "No major gaps 🎉")

                # Bar chart
                skill_df = pd.DataFrame({
                    "Category": ["Matched Skills", "Missing Skills"],
                    "Count": [len(matched), len(missing)]
                })

                fig_bar = px.bar(
                    skill_df,
                    x="Category",
                    y="Count",
                    color="Category",
                    text="Count",
                    title="Skill Gap Analysis"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Pie chart
                fig_pie = px.pie(
                    values=[len(matched), len(missing)],
                    names=["Matched Skills", "Missing Skills"],
                    hole=0.4,
                    title="Skill Coverage Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Save results (CSV logging)
                result_df = pd.DataFrame({
                    "Job Role": [role],
                    "Experience Level": [exp],
                    "Match Score (%)": [match_score],
                    "Skill Coverage (%)": [coverage],
                    "Matched Skills": [matched],
                    "Missing Skills": [missing]
                })

                result_df.to_csv("analysis_results.csv", mode="a", index=False)

            # Course recommendations
            skill_map = {
                "Data Science": ds_course,
                "Web Development": web_course,
                "Android Development": android_course,
                "IOS Development": ios_course,
                "UI-UX": uiux_course
            }

            st.subheader("📚 Recommended Courses")
            for c in skill_map.get(role, []):
                st.markdown(f"- [{c[0]}]({c[1]})")

            st.video(random.choice(resume_videos))
            st.video(random.choice(interview_videos))
            st.balloons()

    else:
        st.markdown("""
        ### AI Resume Analyzer – Final Year Project

        **Core Features**
        - NLP-based Resume Parsing
        - ML Job Role & Experience Prediction
        - Resume–JD Matching (TF-IDF + Cosine Similarity)
        - Skill Gap Analysis
        - Visual Analytics & Recommendations

        **Technologies Used**
        Python, NLP, Machine Learning, Streamlit, Plotly
        """)
run()