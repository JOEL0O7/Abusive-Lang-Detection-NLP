# Importing required libraries import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd

nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv('train.csv')

# Preprocessing text data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    # Remove stopwords and lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df['cleaned_comment'] = df['comment_text'].apply(clean_text)

# Define feature (X) and relevant labels for abusive language detection (y)
X = df['cleaned_comment']
y = df[['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate']]  # Focus on abusive language-related labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model building: OneVsRest classifier with Logistic Regression
model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Function to predict whether user input text contains abusive language
def predict_abusive_language(text):
    # Preprocess the user input
    cleaned_text = clean_text(text)
    # Transform it using the same TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    # Predict the labels
    prediction = model.predict(text_tfidf)
    labels = ['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate']
    
    # If any of these labels are predicted as 1, we classify it as abusive
    is_abusive = any(prediction[0])
    
    if is_abusive:
        return "Abusive Language Detected"
    else:
        return "No Abusive Language Detected"

# Test on user input
user_input = input("Enter a comment to check for abusive language: ")
result = predict_abusive_language(user_input)

# Displaying prediction result
print(result)
