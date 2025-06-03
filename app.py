
import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import base64
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="WhiteSwan Structured Screener", layout="wide")

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

st.title("🦢 WhiteSwan Structured Resume Screener")
st.write("Upload resumes and provide structured job requirements to get a recruiter-style evaluation.")

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

def evaluate(requirements, resume_text):
    feedback = []
    for i, req in enumerate(requirements.split("\n")):
        if req.strip():
            level = score_fit(req, resume_text)
            feedback.append(f"{i+1}. **{req.strip()}**: {level}")
    return feedback

st.markdown("### Step 1: Paste Job Responsibilities (one per line)")
job_requirements = st.text_area("Example:\nDesign/Develop IBM FTM Solutions\nCollaborate with Teams\nTranslate Requirements...", height=200)

st.markdown("### Step 2: Upload Resume (PDF or DOCX or TXT)")
uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if job_requirements and uploaded_resume:
    resume_text = extract_text(uploaded_resume)
    st.markdown("---")
    st.subheader(f"📄 Evaluation for: {uploaded_resume.name}")
    feedback = evaluate(job_requirements, resume_text)
    for point in feedback:
        st.markdown(point)
    st.markdown("---")
    st.success("✅ Structured evaluation complete.")
