
import streamlit as st
import pdfplumber
import docx2txt
import re
import pandas as pd
import string

st.set_page_config(page_title="WhiteSwan Smart Resume Screener", layout="wide")
st.title("🦢 WhiteSwan Smart Resume Matcher")
st.write("Upload multiple resumes (PDF or DOCX) and match them to a job description using smart keyword extraction.")

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
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+]", " ", text)
    return text

def tokenize(text):
    stop_words = set([
        "the", "and", "for", "are", "with", "that", "this", "you", "your", "have",
        "has", "had", "was", "were", "not", "but", "from", "they", "their", "been",
        "will", "would", "could", "should", "about", "into", "than", "then", "out",
        "get", "got", "also", "each", "any", "all", "per", "she", "him", "her", "our",
        "its", "it's", "is", "am", "an", "a", "of", "to", "in", "on", "by", "as", "be", "at", "or", "if", "it", "so", "we", "do"
    ])
    words = clean_text(text).split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return set(keywords)

def get_match_score(resume, jd_keywords):
    resume_tokens = tokenize(resume)
    matched = jd_keywords & resume_tokens
    missing = jd_keywords - resume_tokens
    score = round((len(matched) / len(jd_keywords)) * 100, 2) if jd_keywords else 0
    return score, matched, missing

# --- Run Matching for Multiple Resumes ---
if resume_files and jd_text:
    jd_keywords = tokenize(jd_text)

    results = []
    for file in resume_files:
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file)
        else:
            resume_text = extract_text_from_docx(file)

        score, matched, missing = get_match_score(resume_text, jd_keywords)

        feedback = ""
        if score > 70:
            feedback = "✅ Great match! Candidate has most of the key skills."
        elif score > 40:
            feedback = "🟡 Moderate match. Some important skills missing."
        else:
            feedback = "❌ Low match. Resume may not meet core requirements."

        results.append({
            "Resume": file.name,
            "Score (%)": score,
            "Matched Keywords": ", ".join(sorted(matched)),
            "Missing Keywords": ", ".join(sorted(missing)),
            "Feedback": feedback
        })

    df = pd.DataFrame(results)
    st.subheader("📊 Smart Match Results")
    st.dataframe(df, use_container_width=True)
