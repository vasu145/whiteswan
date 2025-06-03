
import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import base64
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Recruiter-Style Resume Screener", layout="wide")

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

st.title("🦢 WhiteSwan Recruiter-Style Resume Reviewer")

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

def score_fit(requirement, resume_text):
    req_embedding = model.encode(requirement, convert_to_tensor=True)
    res_embedding = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(req_embedding, res_embedding).item()
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

def evaluate_section(title, items, resume_text):
    st.markdown(f"### 📌 {title}")
    results = []
    gaps = []
    for i, item in enumerate(items.split("\n")):
        if item.strip():
            level = score_fit(item.strip(), resume_text)
            results.append((item.strip(), level))
            if level in ["Missing", "Weak"]:
                gaps.append(item.strip())
    for req, level in results:
        st.markdown(f"- **{req}**: {level}")
    return results, gaps

st.markdown("### Step 1: Paste Job Duties")
job_duties = st.text_area("Job Duties (one per line)", height=150)

st.markdown("### Step 2: Paste Core Skills")
core_skills = st.text_area("Core Skills (Must Have)", height=120)

st.markdown("### Step 3: Paste Secondary Skills")
secondary_skills = st.text_area("Secondary Skills (Good to Have)", height=100)

st.markdown("### Step 4: Upload Resume")
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if resume_file and (job_duties or core_skills or secondary_skills):
    resume_text = extract_text(resume_file)
    st.markdown("---")
    st.header(f"📄 Evaluation for: {resume_file.name}")

    all_results = []
    all_gaps = []

    if job_duties:
        duties, duty_gaps = evaluate_section("Job Duties Fit", job_duties, resume_text)
        all_results.extend(duties)
        all_gaps.extend(duty_gaps)

    if core_skills:
        skills, skill_gaps = evaluate_section("Core Skills Fit", core_skills, resume_text)
        all_results.extend(skills)
        all_gaps.extend(skill_gaps)

    if secondary_skills:
        secondary, sec_gaps = evaluate_section("Secondary Skills Fit", secondary_skills, resume_text)
        all_results.extend(secondary)
        all_gaps.extend(sec_gaps)

    st.markdown("### 🧠 Summary & Recommendation")
    excellent = sum(1 for _, lvl in all_results if lvl == "Excellent")
    strong = sum(1 for _, lvl in all_results if lvl == "Strong")
    total = len(all_results)
    avg_score = (excellent * 1.0 + strong * 0.8) / total if total else 0

    if avg_score > 0.75:
        st.success("✅ **Overall Fit:** Strong fit — proceed to interview.")
    elif avg_score > 0.5:
        st.warning("⚠️ **Overall Fit:** Moderate fit — consider further evaluation.")
    else:
        st.error("❌ **Overall Fit:** Weak fit — likely not suitable.")

    if all_gaps:
        st.markdown("### ⚠️ Gaps Identified")
        for gap in all_gaps:
            st.markdown(f"- {gap}")

    st.markdown("---")
    st.success("✅ Evaluation complete.")
