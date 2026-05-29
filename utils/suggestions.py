def generate_suggestions(score, missing_skills):
    tips = []

    if score < 50:
        tips.append(
            "Improve resume based on job description"
        )

    if len(missing_skills) > 0:
        tips.append(
            "Add missing skills: " +
            ", ".join(missing_skills[:5])
        )

    tips.append(
        "Use action verbs and measurable achievements"
    )

    return tips
