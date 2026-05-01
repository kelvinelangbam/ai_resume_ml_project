def load_skills():
    with open("data/skills.txt") as f:
        return f.read().splitlines()

def extract_skills(text, skills):
    return [s for s in skills if s in text]