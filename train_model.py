import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.text_processing import preprocess

# Load dataset
df = pd.read_csv("data/UpdatedResumeDataSet.csv")
df = df[['Resume', 'Category']]
df.dropna(inplace=True)

# Preprocess
df['Resume'] = df['Resume'].apply(preprocess)

# Encode labels
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
pickle.dump(le, open("models/encoder.pkl", "wb"))

print("Model saved successfully!")