import streamlit as st
import pdfplumber
import docx2txt
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="WhiteSwan Resume Screener", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .title {text-align: center; font-size: 2.5em; color: #2c3e50; font-weight: bold;}
        .subheader {color: #34495e;}
        .stTextArea textarea {font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦢 WhiteSwan AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown("### Upload resumes and enter the job description to get contextual match feedback.")

# ----------- Job Description Input -------------
job_description = st.text_area("Paste the Job Description", height=250)

# ----------- File Upload -------------
uploaded_files = st.file_uploader("Upload Resume files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# ----------- Extract Resume Text -------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    temp_path.write(file.read())
    temp_path.close()
    text = docx2txt.process(temp_path.name)
    os.unlink(temp_path.name)
    return text

def get_resume_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        return ""

# ----------- Matching Logic -------------
def match_resume_to_jd(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(similarity[0][0])

# ----------- Process Resumes -------------
if st.button("Analyze Resumes"):
    if not job_description:
        st.warning("Please enter the job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Analyzing resumes..."):
            for file in uploaded_files:
                resume_text = get_resume_text(file)
                score = match_resume_to_jd(resume_text, job_description)
                st.subheader(f"📄 {file.name}")
                st.write(f"**Match Score:** {round(score * 100, 2)}%")

                if score >= 0.75:
                    st.success("✅ Strong match! Likely a good fit based on JD.")
                elif score >= 0.5:
                    st.info("⚠️ Moderate match. May need further review.")
                else:
                    st.warning("❌ Weak match. Unlikely to fit well.")

