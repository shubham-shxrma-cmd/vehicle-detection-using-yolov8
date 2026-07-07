
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Step 2: Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Convert Labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Prediction & Evaluation
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Try Custom Input
def predict_spam(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"

# Example
while True:
    sample = input("\n📩 Enter a message to check (or type 'exit' to quit): ")
    if sample.lower() == 'exit':
        break
    print("🔍 Prediction:", predict_spam(sample))



