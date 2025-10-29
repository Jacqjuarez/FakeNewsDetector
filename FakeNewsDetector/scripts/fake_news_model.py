import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

df = pd.read_csv("../data/news.csv")

# --- Visualize data distribution ---
plt.figure(figsize=(6, 4))
df["label"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Fake vs Real News Articles")
plt.xlabel("Label")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../results/news_distribution.png")
plt.show()


X = df["text"]
y = df["label"]  # FAKE or REAL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))



cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("Confusion Matrix: Fake News Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png")
plt.show()
