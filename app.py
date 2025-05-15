
import streamlit as st
import pdfplumber
import docx2txt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="WhiteSwan AI Resume Screener", layout="centered")
st.title("🦢 WhiteSwan Resume Matcher")
st.write("Upload a resume and job description to see the skill match.")

# --- File Upload ---
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_text = st.text_area("Paste Job Description")

# --- Text Extraction Functions ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text.lower()

# --- Skill Matching ---
def get_match_score(resume, jd):
    vect = CountVectorizer().fit_transform([resume, jd])
    score = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    return round(score * 100, 2)

# --- Run Matching ---
if resume_file and jd_text:
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = extract_text_from_docx(resume_file)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    score = get_match_score(resume_clean, jd_clean)

    st.subheader("🔍 Match Score")
    st.write(f"✅ The resume matches the job description by **{score}%**")

    if score > 70:
        st.success("Great match! This candidate is highly relevant.")
    elif score > 40:
        st.warning("Moderate match. May require closer review.")
    else:
        st.error("Low match. Resume may not be suitable.")
