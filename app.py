
import streamlit as st
import pdfplumber
import docx2txt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io

st.set_page_config(page_title="WhiteSwan Bulk Resume Screener", layout="wide")
st.title("🦢 WhiteSwan Bulk Resume Matcher")
st.write("Upload multiple resumes (PDF or DOCX) and match them to a job description.")

# --- File Upload ---
resume_files = st.file_uploader("Upload Resumes (PDF or DOCX, multiple allowed)", type=["pdf", "docx"], accept_multiple_files=True)
jd_text = st.text_area("Paste Job Description")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text.lower()

def get_keywords(text):
    return set(clean_text(text).split())

def get_match_score(resume, jd):
    vect = CountVectorizer().fit_transform([resume, jd])
    score = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    return round(score * 100, 2)

# --- Run Matching for Multiple Resumes ---
if resume_files and jd_text:
    jd_clean = clean_text(jd_text)
    jd_keywords = get_keywords(jd_text)
    
    results = []
    for file in resume_files:
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file)
        else:
            resume_text = extract_text_from_docx(file)

        resume_clean = clean_text(resume_text)
        resume_keywords = get_keywords(resume_text)

        score = get_match_score(resume_clean, jd_clean)
        matched = sorted(jd_keywords & resume_keywords)
        missing = sorted(jd_keywords - resume_keywords)

        feedback = ""
        if score > 70:
            feedback = "Great match! This candidate is highly relevant."
        elif score > 40:
            feedback = "Moderate match. May require closer review."
        else:
            feedback = "Low match. Resume may not be suitable."

        results.append({
            "Resume": file.name,
            "Score (%)": score,
            "Matched Keywords": ", ".join(matched),
            "Missing Keywords": ", ".join(missing),
            "Feedback": feedback
        })

    df = pd.DataFrame(results)
    st.subheader("📊 Match Results")
    st.dataframe(df, use_container_width=True)
