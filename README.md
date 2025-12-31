# 🤖 AI Resume Analyzer

AI Resume Analyzer is a web-based application that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to analyze resumes, predict suitable job roles and experience levels, and match resumes against job descriptions. The system also performs **skill gap analysis** and provides **visual analytics** to help candidates improve their profiles.

---

## 🚀 Features

- 📄 Resume text extraction from PDF files  
- 🧠 Job role prediction using supervised ML models  
- 🧪 Experience level classification  
- 📊 Resume–Job Description matching using TF-IDF and Cosine Similarity  
- 🧩 Skill extraction and skill gap analysis  
- 📈 Visual analytics (bar charts, pie charts, progress indicators)  
- 🎓 Course recommendations based on predicted job role  
- 🌐 Interactive web interface built with Streamlit  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **NLP & ML:** Scikit-learn, NLTK  
- **Text Processing:** TF-IDF Vectorizer, Cosine Similarity  
- **PDF Parsing:** pdfminer.six  
- **Web Framework:** Streamlit  
- **Visualization:** Plotly  
- **Model Serialization:** Joblib  
- **Deployment:** Streamlit Cloud  

---

## 📌 System Architecture

1. Resume PDF is uploaded by the user  
2. Text is extracted using PDF parsing  
3. NLP preprocessing and feature extraction (TF-IDF)  
4. ML models predict job role and experience level  
5. Resume is matched with Job Description  
6. Skill gap analysis is performed  
7. Results are displayed using visual analytics

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/amal-antony-alex/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application locally
```bash
streamlit run App.py
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push the project to GitHub  
2. Go to https://streamlit.io/cloud  
3. Connect your GitHub repository  
4. Select `App.py` as the main file  
5. Deploy the application  

---

## 👤 Author
**Amal Antony Alex**  
GitHub: https://github.com/amal-antony-alex  

---

## 📜 License
This project is intended for educational and academic use.
