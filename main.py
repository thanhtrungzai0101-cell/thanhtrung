# Import necessary libraries and packages
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Download the Reuters dataset from NLTK and load it
nltk.download('reuters')
from nltk.corpus import reuters

# Print the number of documents and categories in the Reuters dataset
print(f"Number of documents: {len(reuters.fileids())}")
print(f"Number of categories: {len(reuters.categories())}")

# Data preparation
categories = ['grain', 'crude', 'trade']
docs = [(reuters.raw(fileid), category)
        for category in categories
        for fileid in reuters.fileids(category)]

texts, labels = zip(*docs)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tf = tfidf_vectorizer.fit_transform(X_train)
X_test_tf = tfidf_vectorizer.transform(X_test)



# Model Training and Prediction
classifier = MultinomialNB()
classifier.fit(X_train_tf, y_train)
y_pred = classifier.predict(X_test_tf)

# Evaluation
print("Classification Report:")
print("===================================")
print(classification_report(y_test, y_pred, labels=categories))
print("\n")

# Visualization of Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=categories)
print("Confusion Matrix:")
print("===================================")
print(conf_matrix)
print("\n")

# Heatmap Visualization
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix Heatmap")
plt.show()