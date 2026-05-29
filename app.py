from fpdf import FPDF

import streamlit as st
import pickle
import pandas as pd
import io
from sklearn.metrics.pairwise import cosine_similarity

from utils.resume_parser import extract_text
from utils.text_processing import preprocess
from utils.skills import load_skills, extract_skills
from utils.suggestions import generate_suggestions
from utils.education import extract_education

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
div[data-testid="InputInstructions"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("AI Resume Screening System")

# ---------------- UPLOAD ----------------
st.subheader("📂 Upload Resumes")

files = st.file_uploader(
    "",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- JOB DESCRIPTION ----------------
job_desc = st.text_area("Paste Job Description", height=200)

# ---------------- BUTTON ----------------
analyze = st.button("Analyze & Rank Candidates")

# ---------------- PDF CLEAN FUNCTION (FIX ERROR) ----------------
def clean_text(text):
    return (
        text.replace("🔥", "")
            .replace("👍", "")
            .replace("⚠️", "")
            .replace("🧠", "")
            .replace("📊", "")
            .replace("🎯", "")
            .replace("🟢", "")
            .replace("🟡", "")
            .replace("🔴", "")
    )

# ---------------- AI FEEDBACK ENGINE ----------------
def generate_ai_feedback(score, common_skills, missing_skills, role):

    feedback = []

    if score >= 80:
        feedback.append("Strong candidate with high job relevance.")
    elif score >= 60:
        feedback.append("Moderate candidate, suitable for shortlist.")
    else:
        feedback.append("Weak candidate, not recommended for selection.")

    if len(common_skills) >= 5:
        feedback.append("Strong skill alignment with job description.")
    elif len(common_skills) >= 2:
        feedback.append("Partial skill match found.")
    else:
        feedback.append("Very low skill overlap.")

    if missing_skills:
        feedback.append("Improve skills: " + ", ".join(missing_skills[:5]))

    feedback.append(f"Suggested Role Fit: {role}")

    if score >= 75 and len(common_skills) >= 3:
        feedback.append("SELECTED for interview")
    elif score >= 50:
        feedback.append("HOLD for review")
    else:
        feedback.append("REJECT")

    return feedback


# ---------------- MAIN LOGIC ----------------
if files and job_desc and analyze:

    jd_clean = preprocess(job_desc)
    required_education = extract_education(job_desc)

    jd_vec = tfidf.transform([jd_clean])

    skills = load_skills()
    jd_skills = set(extract_skills(jd_clean, skills))

    results = []

    for file in files:

        text = extract_text(file)
        if not text:
            continue

        clean = preprocess(text)
        candidate_education = extract_education(text)

        # Education filter
        if required_education:
            matched = any(
                edu.lower() in [c.lower() for c in candidate_education]
                for edu in required_education
            )
            if not matched:
                continue

        resume_vec = tfidf.transform([clean])

        pred = model.predict(resume_vec)
        role = pred[0]

        cosine_score = cosine_similarity(resume_vec, jd_vec)[0][0]

        resume_skills = set(extract_skills(clean, skills))
        common_skills = resume_skills.intersection(jd_skills)

        skill_score = (
            len(common_skills) / len(jd_skills)
            if jd_skills else 0
        )

        final_score = (0.7 * cosine_score) + (0.3 * skill_score)
        score = round(final_score * 100, 2)

        missing_skills = list(jd_skills - resume_skills)

        # AI FEEDBACK
        feedback = generate_ai_feedback(score, common_skills, missing_skills, role)

        # STORE RESULTS
        results.append({
            "Candidate": file.name,
            "Education": ", ".join(candidate_education),
            "Predicted Role": role,
            "ATS Score": score,
            "Matched Skills": ", ".join(sorted(common_skills)),
            "Missing Skills": ", ".join(sorted(missing_skills[:5])),
            "Suggestions": " | ".join(feedback)
        })

    # ---------------- RESULTS ----------------
    if results:

        df = pd.DataFrame(results)

        df["ATS Score"] = pd.to_numeric(df["ATS Score"], errors="coerce")

        df = df.sort_values(by="ATS Score", ascending=False)
        df.reset_index(drop=True, inplace=True)
        df.index = df.index + 1

        # TABLE
        st.subheader("🏆 Candidate Ranking")
        st.dataframe(df, use_container_width=True)

        top_candidate = df.iloc[0]

        st.success(
            f"Top Candidate: {top_candidate['Candidate']} "
            f"with ATS Score {top_candidate['ATS Score']}%"
        )

        # TOP 10 GRAPH
        st.subheader("📊 Top 10 Candidate Scores")

        top10 = df.head(10)
        st.bar_chart(top10.set_index("Candidate")["ATS Score"])

        # ---------------- PDF REPORT ----------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Candidate Ranking Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)

        for index, row in df.iterrows():

            line = (
                f"Rank: {index}\n"
                f"Candidate: {row['Candidate']}\n"
                f"Education: {row['Education']}\n"
                f"Role: {row['Predicted Role']}\n"
                f"ATS Score: {row['ATS Score']}%\n"
                f"Matched Skills: {row['Matched Skills']}\n"
                f"Missing Skills: {row['Missing Skills']}\n"
                f"Suggestions: {row['Suggestions']}\n"
            )

            pdf.multi_cell(0, 10, clean_text(line))
            pdf.ln(5)

        pdf_output = pdf.output(dest="S").encode("latin1", errors="ignore")

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name="candidate_ranking_report.pdf",
            mime="application/pdf"
        )

        # ---------------- EXCEL REPORT ----------------
        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="All Candidates")
            top10.to_excel(writer, index=False, sheet_name="Top 10 Candidates")

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="candidate_ranking_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.error("No matching resumes found")
