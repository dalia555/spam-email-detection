#naive bayes
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
data = pd.read_csv("spam.csv", encoding='latin-1')

# Fix columns (important!)
data = data[['Category', 'Message']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ✅ APPLY preprocessing
data['message'] = data['message'].apply(preprocess)

# ✅ Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

def predict_text(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    result = model_nb.predict(vector)[0]
    
    return "Spam" if result == 1 else "Not Spam"
while True:
    text = input("Enter a message (or type 'exit'): ")
    
    if text.lower() == "exit":
        break
    
    print(predict_text(text))