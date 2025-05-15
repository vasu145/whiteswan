import streamlit as st
import pdfplumber
import docx2txt
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="WhiteSwan Semantic Resume Screener", layout="wide")
st.title("🧠 WhiteSwan AI-Powered Resume Matcher")
st.write("Upload resumes and compare them with a job description using semantic similarity.")

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- File Upload ---
resume_files = st.file_uploader("Upload Resumes (PDF or DOCX, multiple allowed)", type=["pdf", "docx"], accept_multiple_files=True)
jd_text = st.text_area("Paste Job Description")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(file):
    return docx2txt.process(file)

def get_resume_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    else:
        return extract_text_from_docx(file)

# --- Matching Logic ---
if resume_files and jd_text:
    with st.spinner("Analyzing resumes..."):
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        results = []

        for file in resume_files:
            resume_text = get_resume_text(file)
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
            score_percent = round(score * 100, 2)

            if score_percent > 80:
                feedback = "✅ Strong match: Resume aligns well with the job requirements."
            elif score_percent > 60:
                feedback = "🟡 Moderate match: Resume covers some important aspects."
            else:
                feedback = "❌ Weak match: Resume doesn't align closely with the job description."

            results.append({
                "Resume": file.name,
                "Similarity Score (%)": score_percent,
                "Feedback": feedback
            })

        df = pd.DataFrame(results)
        st.subheader("📊 Semantic Match Results")
        st.dataframe(df, use_container_width=True)
