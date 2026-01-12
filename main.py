import nltk
import re
import string  
# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = ("hi my name is Trunk and i love sing so much!!! i want in the future to be a singer perfromer and travel the world. this is my big dream." \
" i also love listen to music in free time and i am a big fan of us-uk music bands. i usually listen music when myself miss someone like the person i love " \
    "in my free time i enjoy karaoke alone to practice my singing skillalthough i dont know myself can become a good singer in the future yet ")
print("Original Text:\n" + text)
print("-" * 60)
#Tokenization
tokens = word_tokenize(text)
print("Tokens:\n", tokens)
print("-" * 60)

#Lowercasing
lower_tokens = [token.lower() for token in tokens]
print("Lowercased Tokens:\n", lower_tokens)
print("-" * 60)

#Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in lower_tokens if token not in stop_words]
print("Filtered Tokens (Stopwords Removed):\n", filtered_tokens)
print("-" * 60)

# Punctuation Removal
punctuation_removed_tokens = [re.sub(f"[{re.escape(string.punctuation)}]", "", token) for token in filtered_tokens]
punctuation_removed_tokens = [token for token in punctuation_removed_tokens if token]  #
print("Tokens after Punctuation Removal:\n", punctuation_removed_tokens)
print("-" * 60)

#Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in punctuation_removed_tokens]
print("Stemmed Tokens:\n", stemmed_tokens)
print("-" * 60)

#Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in punctuation_removed_tokens]
print("Lemmatized Tokens:\n", lemmatized_tokens)
print("-" * 60)

#Text normalization funcion
def normalize_text(input_text):
    tokens = word_tokenize(input_text)
    lower_tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lower_tokens if token not in stop_words]
    punctuation_removed_tokens = [re.sub(f"[{re.escape(string.punctuation)}]", "", token) for token in filtered_tokens]
    punctuation_removed_tokens = [token for token in punctuation_removed_tokens if token]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in punctuation_removed_tokens]
    return lemmatized_tokens