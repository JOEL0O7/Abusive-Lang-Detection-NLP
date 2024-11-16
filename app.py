# app.py

from flask import Flask, request, render_template
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

app = Flask(__name__)

# Load and preprocess data
start_time = time.time()
df = pd.read_csv('train.csv')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df['cleaned_comment'] = df['comment_text'].apply(clean_text)

# Define features and labels
X = df['cleaned_comment']
y = df[['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train model
model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Calculate performance metrics
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, target_names=['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate'])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        user_input = request.form['text']
        cleaned_text = clean_text(user_input)
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)
        labels = ['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate']
        is_abusive = any(prediction[0])
        
        result = {
            "text": user_input,
            "prediction": "Abusive Language Detected" if is_abusive else "No Abusive Language Detected",
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1,
            "classification_report": report
        }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
