import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import base64
import re
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="WhiteSwan Smart Reviewer", layout="wide")

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
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

st.title("🦢 WhiteSwan Recruiter-Style Resume Reviewer (Smart Matching)")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8")

def clean_sentences(text):
    sentences = re.split(r'[\n\r\.]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def match_requirement(requirement, resume_sentences):
    req_embedding = model.encode(requirement, convert_to_tensor=True)
    best_score = 0
    best_sentence = ""
    for sentence in resume_sentences:
        sent_embedding = model.encode(sentence, convert_to_tensor=True)
        score = util.cos_sim(req_embedding, sent_embedding).item()
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_score, best_sentence

def get_rating(score):
    if score > 0.75:
        return "Excellent"
    elif score > 0.55:
        return "Strong"
    elif score > 0.35:
        return "Moderate"
    elif score > 0.20:
        return "Weak"
    else:
        return "Missing"

st.markdown("### Step 1: Paste Job Requirements (one per line)")
job_requirements = st.text_area("Example:\nSupport production\nExperience in FTM\nAgile development", height=200)

st.markdown("### Step 2: Upload Resume")
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if job_requirements and resume_file:
    resume_text = extract_text(resume_file)
    resume_sentences = clean_sentences(resume_text)
    jd_lines = [j.strip() for j in job_requirements.split("\n") if j.strip()]

    st.markdown("---")
    st.header(f"📄 Evaluation for: {resume_file.name}")
    total_score = 0

    for jd in jd_lines:
        score, matched_sentence = match_requirement(jd, resume_sentences)
        label = get_rating(score)
        total_score += score
        st.markdown(f"**{jd}**")
        st.markdown(f"- Match: **{label}**")
        st.markdown(f"- Evidence: _\"{matched_sentence}\"_")
        st.markdown("---")

    avg_score = total_score / len(jd_lines) if jd_lines else 0
    st.subheader("📌 Final Recommendation")
    if avg_score > 0.75:
        st.success("✅ Strong fit — Proceed to interview.")
    elif avg_score > 0.5:
        st.warning("⚠️ Moderate fit — Consider further evaluation.")
    else:
        st.error("❌ Weak fit — Resume does not align closely with role.")
