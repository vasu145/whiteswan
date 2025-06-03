
import streamlit as st
import pdfplumber
import docx2txt
import io
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import re
import string
import base64

# ✅ Set page config FIRST
st.set_page_config(page_title="WhiteSwan Advanced Screener", layout="wide")

# 🌄 Set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.jpg")

st.title("🦢 WhiteSwan Advanced Resume Screener")
st.write("Upload resumes (PDF, DOCX, TXT) and compare them with a job description. Get a match decision plus a 5-point explanation.")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_txt(file):
    return file.read().decode('utf-8')

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
    return set([w for w in words if w not in stop_words and len(w) > 2])

def compare_jd_resume(jd_text, resume_text):
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(jd_embedding, resume_embedding).item()
    score_percent = round(score * 100, 2)
    jd_keys = tokenize(jd_text)
    res_keys = tokenize(resume_text)
    matched = jd_keys & res_keys
    missing = jd_keys - res_keys
    return score_percent, matched, missing

def generate_feedback(score, matched, missing):
    feedback = []
    feedback.append(f"1. **Overall Match Score:** {score}%")
    if matched:
        feedback.append(f"2. **Matched Skills/Keywords:** {', '.join(sorted(list(matched))[:7])}")
    else:
        feedback.append("2. **Matched Skills/Keywords:** None of the core keywords were found.")
    if missing:
        feedback.append(f"3. **Missing Key Skills/Keywords:** {', '.join(sorted(list(missing))[:7])}")
    else:
        feedback.append("3. **Missing Key Skills/Keywords:** None; candidate covers all identified keywords.")
    feedback.append("4. **Domain Relevance:** Resume content is evaluated for industry-specific terms.")
    if score > 75:
        feedback.append("5. **Recommendation:** Strong fit; proceed to next stage.")
    elif score > 50:
        feedback.append("5. **Recommendation:** Moderate fit; consider further screening.")
    else:
        feedback.append("5. **Recommendation:** Weak fit; likely not suitable for this role.")
    return feedback

jd_text = st.text_area("Paste the Job Description", height=200)
uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)

if jd_text and uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file)
        elif file.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document","application/msword"):
            resume_text = extract_text_from_docx(file)
        else:
            resume_text = extract_text_from_txt(file)
        score, matched, missing = compare_jd_resume(jd_text, resume_text)
        feedback_points = generate_feedback(score, matched, missing)
        st.write(f"**Match Score:** {score}%")
        for point in feedback_points:
            st.markdown(point)
        st.markdown("---")
