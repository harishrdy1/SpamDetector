import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Download NLTK stopwords
nltk.download('stopwords')

# Preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Step 1: Load dataset
df = pd.read_csv('spam_ham_dataset.csv')  # Your file name here
df['cleaned'] = df['text'].apply(preprocess)
X_text = df['cleaned']
y = df['label_num']  # 0 = ham, 1 = spam

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 6: Predict new email
def predict_email(raw_text):
    text = preprocess(raw_text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Spam" if result == 1 else "Not Spam"

# Example
# --- Interactive mode ---
while True:
    print("\nEnter your email content (type 'exit' to quit):")
    user_input = input(">> ")
    
    if user_input.strip().lower() == 'exit':
        print("Goodbye!")
        break

    prediction = predict_email(user_input)
    print("Prediction:", prediction)


# Step 7: Save model
joblib.dump(model, 'email_spam_model.pkl')
joblib.dump(vectorizer, 'email_vectorizer.pkl')
