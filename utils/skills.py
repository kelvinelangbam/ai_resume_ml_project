import re

def load_skills():
    with open("data/skills.txt") as f:
        return [s.strip().lower() for s in f.readlines()]

def extract_skills(text, skills):
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")

    found = []

    for skill in skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))