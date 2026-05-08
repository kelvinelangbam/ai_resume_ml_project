import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils.resume_parser import extract_text
from utils.text_processing import preprocess
from utils.skills import load_skills, extract_skills
from utils.suggestions import generate_suggestions

# Load models
model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))

st.title("📄 AI-Powered Resume Screening & Job Matching System")

file = st.file_uploader("Upload Resume", type="pdf")
job_desc = st.text_area("Paste Job Description")

if file:
    text = extract_text(file)
    clean = preprocess(text)

    # ---------------- ML Prediction ----------------
    vec = tfidf.transform([clean])
    pred = model.predict(vec)
    role = encoder.inverse_transform(pred)[0]

    st.subheader(f"🎯 Predicted Role: {role}")

    if job_desc:
        jd_clean = preprocess(job_desc)

        # ---------------- Match Score ----------------
        resume_vec = tfidf.transform([clean])
        jd_vec = tfidf.transform([jd_clean])

        cosine_score = cosine_similarity(resume_vec, jd_vec)[0][0]

        skills = load_skills()

        resume_skills = set(extract_skills(clean, skills))
        jd_skills = set(extract_skills(jd_clean, skills))

        common = resume_skills.intersection(jd_skills)

        if len(jd_skills) > 0:
            skill_score = len(common) / len(jd_skills)
        else:
            skill_score = 0

        final_score = (0.7 * cosine_score) + (0.3 * skill_score)
        score = round(final_score * 100, 2)

        st.subheader(f"📊 Match Score: {score}%")

        # ---------------- Skills ----------------
        found = list(resume_skills)
        missing = list(jd_skills - resume_skills)

        st.write("🛠️ Skills:", found)
        st.write("⚠️ Missing:", missing)

        # ---------------- Suggestions ----------------
        tips = generate_suggestions(score, missing)

        st.subheader("💡 Suggestions")
        for t in tips:
            st.write("- " + t)