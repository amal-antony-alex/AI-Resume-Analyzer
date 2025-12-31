import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# Clean missing values
df.dropna(inplace=True)

# -------------------------
# MODEL 1: JOB ROLE CLASSIFIER
# -------------------------
X = df['resume_text']
y_role = df['job_role']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_role, test_size=0.2, random_state=42
)

vectorizer_role = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer_role.fit_transform(X_train)
X_test_vec = vectorizer_role.transform(X_test)

role_model = LogisticRegression(max_iter=1000)
role_model.fit(X_train_vec, y_train)

y_pred = role_model.predict(X_test_vec)

print("\n🔹 Job Role Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(role_model, "models/job_role_model.pkl")
joblib.dump(vectorizer_role, "models/job_role_vectorizer.pkl")

# -------------------------
# MODEL 2: EXPERIENCE LEVEL CLASSIFIER
# -------------------------
y_exp = df['experience_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_exp, test_size=0.2, random_state=42
)

vectorizer_exp = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer_exp.fit_transform(X_train)
X_test_vec = vectorizer_exp.transform(X_test)

exp_model = LogisticRegression(max_iter=1000)
exp_model.fit(X_train_vec, y_train)

y_pred = exp_model.predict(X_test_vec)

print("\n🔹 Experience Level Classification Report")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(exp_model, "models/experience_model.pkl")
joblib.dump(vectorizer_exp, "models/experience_vectorizer.pkl")

print("\n✅ Models trained and saved successfully")
