import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('Dataset---Hate-Speech-Detection-using-Deep-Learning.csv')
# Display the first few rows of the dataset
print(data.head())
# Print the shape of the data frame
print(data.shape)
# Check the info about their columns
print(data.info())

# Plot class distribution
plt.pie(
 data['class'].value_counts().values,  # counts of each class
 labels=data['class'].value_counts().index, # class names
 autopct='%1.1f%%',  # show percentages
 startangle=90 # rotate for better look
)
plt.title("Class Distribution")
plt.show()

# Balancing the Dataset, using a combination of upsampling and downsampling
class_0 = data[data['class'] == 0] # Hate Speech
class_1 = data[data['class'] == 1].sample(n=3500, random_state=42) # Offensive Language
class_2 = data[data['class'] == 2] # Neutral

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

# Visualize the balanced distribution
plt.pie(balanced_df['class'].value_counts().values,
 labels=balanced_df['class'].value_counts().index,
 autopct='%1.1f%%')
plt.title("Balanced Class Distribution")
plt.show()
# Text Preprocessing
data['tweet'] = data['tweet'].str.lower()

punctuations_list = string.punctuation
def remove_punctuations(text):
 temp = str.maketrans('', '', punctuations_list)
 return text.translate(temp)

data['tweet']= data['tweet'].apply(lambda x: remove_punctuations(x))
data.head()

def preprocess_text(text):
 stop_words = set(stopwords.words('english'))
 lemmatizer = WordNetLemmatizer()
 words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
 return " ".join(words)

balanced_df['tweet'] = balanced_df['tweet'].apply(preprocess_text)
balanced_df.head()
# Word Cloud Visualization
def plot_word_cloud(data, typ):
 corpus = " ".join(data['tweet'])
 wc = WordCloud(max_words=100, width=800, height=400, collocations=False).generate(corpus)
 plt.figure(figsize=(10, 5))
 plt.imshow(wc, interpolation='bilinear')
 plt.axis('off')
 plt.title(f"Word Cloud for {typ} Class", fontsize=15)
 plt.show()

plot_word_cloud(balanced_df[balanced_df['class'] == 2], typ="Neutral")
plot_word_cloud(balanced_df[balanced_df['class'] == 1], typ="Offensive")
plot_word_cloud(balanced_df[balanced_df['class'] == 0], typ="Hate Speech")
# Tokenization and Padding
features = balanced_df['tweet']
target = balanced_df['class']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# One-hot encode the labels
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

# Tokenization
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
# Build the LSTM Model
max_words = 10000
max_len = 100 

model = keras.models.Sequential([
 layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
 layers.Bidirectional(layers.LSTM(16)),
 layers.Dense(512, activation='relu', kernel_regularizer='l1'),
 layers.BatchNormalization(),
 layers.Dropout(0.3),
 layers.Dense(3, activation='softmax')
])

model.build(input_shape=(None, max_len))

model.compile(loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])

model.summary()
# Train the Model
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)
history = model.fit(X_train_padded, Y_train,
 validation_data=(X_val_padded, Y_val),
 epochs=50,
 batch_size=32,
 callbacks=[es, lr])
# Evaluate the Model
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(title="Loss")

history_df[['accuracy', 'val_accuracy']].plot(title="Accuracy")
plt.show()
