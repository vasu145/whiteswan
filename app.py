import streamlit as st
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Compare JD and Resume
def compare_jd_resume(jd_text, resume_text):
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, resume_embedding).item()
    feedback = generate_feedback(similarity, jd_text, resume_text)
    return similarity, feedback

# Generate basic recruiter-style feedback
def generate_feedback(score, jd, resume):
    if score > 0.75:
        return f"✅ Good match. Resume aligns well with the job description."
    elif score > 0.5:
        return "⚠️ Partial match. Some relevant skills found, but lacks in areas like tools/experience."
    else:
        return "❌ Not a match. Resume seems unrelated to the job description."

# Streamlit App UI
st.title("WhiteSwan - Resume Screener (Free AI Version)")
st.write("Upload a job description and multiple resumes (PDFs). See which candidates are a good fit.")

jd_text = st.text_area("Paste the Job Description", height=250)

uploaded_files = st.file_uploader("Upload Resumes (PDF, multiple)", type=["pdf"], accept_multiple_files=True)

if jd_text and uploaded_files:
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        score, feedback = compare_jd_resume(jd_text, resume_text)
        st.subheader(f"📄 {file.name}")
        st.write(f"**Match Score:** {round(score * 100, 2)}%")
        st.write(f"**Feedback:** {feedback}")
        st.markdown("---")
