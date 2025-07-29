import tkinter as tk
from tkinter import messagebox
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('email_spam_model.pkl')
vectorizer = joblib.load('email_vectorizer.pkl')

# Preprocessing setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_email():
    raw_text = entry.get("1.0", tk.END).strip()
    if not raw_text:
        messagebox.showwarning("Warning", "Please enter email text.")
        return

    text = preprocess(raw_text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    prediction = "Spam" if result == 1 else "Not Spam"
    result_label.config(text=f"Prediction: {prediction}", fg='green' if result == 0 else 'red')

# Build GUI
root = tk.Tk()
root.title("Email Spam Classifier")
root.geometry("500x400")
root.resizable(False, False)

title = tk.Label(root, text="Spam Email Classifier", font=("Helvetica", 16, "bold"))
title.pack(pady=10)

entry = tk.Text(root, height=10, width=60)
entry.pack(pady=10)

predict_button = tk.Button(root, text="Check Email", command=predict_email, font=("Helvetica", 12))
predict_button.pack(pady=5)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
