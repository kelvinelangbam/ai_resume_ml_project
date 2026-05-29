import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- SAMPLE DATA ----------------
resumes = [
    # Data Science
    "python machine learning data science pandas numpy ai model training",
    "tensorflow pytorch deep learning neural networks ai model",
    "data analysis visualization statistics pandas numpy ml",

    # Web Dev
    "html css javascript react frontend backend web development api",
    "react angular vue frontend ui ux design web apps",
    "nodejs express mongodb backend api development",

    # HR
    "hr recruitment hiring onboarding employee management communication",
    "talent acquisition interview scheduling payroll hr policies",

    # DevOps
    "aws docker kubernetes ci cd pipeline cloud devops linux",
    "jenkins automation deployment server monitoring cloud engineer",

    # Data Analyst
    "sql excel power bi tableau data visualization dashboard reporting",
    "business intelligence analytics charts reports insights",

    # CA
    "accounting gst taxation audit finance balance sheet tds income tax",
    "chartered accountant ledger financial statements compliance",

    # Nurse
    "nursing patient care hospital emergency medical health monitoring",
    "clinical care patient treatment hospital ward assistance"
    
    
    # Artificial Intelligence
    "tensorflow pytorch deep learning neural networks ai machine learning",
    "generative ai llm transformers nlp computer vision prompt engineering",
    "artificial intelligence ai models deep learning python tensorflow",
    "chatbot development llm ai applications machine learning"
    
]

labels = [
    "Data Scientist",
    "Data Scientist",
    "Data Scientist",

    "Web Developer",
    "Web Developer",
    "Web Developer",

    "HR",
    "HR",

    "DevOps Engineer",
    "DevOps Engineer",

    "Data Analyst",
    "Data Analyst",

    "Chartered Accountant",
    "Chartered Accountant",

    "Registered Nurse",
    "Registered Nurse"
    
    "AI Engineer",
    "AI Engineer",
    "AI Engineer",
    "AI Engineer"
]

# ---------------- TRAIN MODEL ----------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(resumes)

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)

# ---------------- SAVE MODEL ----------------
os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))

print("✅ Model recreated successfully!")
