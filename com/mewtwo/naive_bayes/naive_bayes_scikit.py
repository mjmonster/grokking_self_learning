"""sklearn implementation of naive Bayes classifier for email spam detection."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

### DataFrame Loading ###
df = pd.read_csv("emails.csv")

### Split data ###
X_train, X_test, y_train, y_test = (
    train_test_split(df['text'], df['label'], test_size=0.2, random_state=42))

### max_df tells the vectorizer to ignore terms that appear in too many documents
### that is, words so common they’re no longer informative.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

### Train classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

### Predictions
y_predict = model.predict(X_test_tfidf)

### Evaluate
print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))
