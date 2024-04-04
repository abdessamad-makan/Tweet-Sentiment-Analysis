
import pandas as pd 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
dfts = pd.read_csv('sentiment_analysis.csv')


import re
import nltk
from nltk.corpus import stopwords
from contractions import contractions_dict
from nltk.stem import WordNetLemmatizer
import wordsegment



wordsegment.load()
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

 #Function to clean text
def clean_text(text):
    text = text.lower()  #lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  #URLs
    text = re.sub(r'@\w+', '', text)  #mentions
    text = re.sub(r'#', '', text)  #hashtags
    text = re.sub(r'[^\w\s]', '', text)  #punctuation
    text = re.sub(r"["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", "", text) #removes emojis
    text = re.sub(r'[^A-Za-z0-9_]', ' ', text)
    text = re.sub(r'\d+', '', text)#digits in the first of word
    text = re.sub(r'\s+', ' ', text).strip()# Remove extra whitespace
    return text

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        contraction = match.group(0)
        return contractions_dict.get(contraction, contraction)
    return contractions_re.sub(replace, text)

def reduce_repeated_characters(text):
    text = re.sub(r'(.)\1+', r'\1\1', text)
    return text

def clean_digits(text):
  return re.sub(r'\b[0-9]+\b\s*', '',text)

def clean_morespace(text):
  return re.sub(' +', ' ', text)

def segment_compound_word(text):
    segmented_text = []
    words = text.split()  # Split the text into individual words
    for word in words:
        segmented_word = wordsegment.segment(word)
        segmented_text.append(" ".join(segmented_word))
    return " ".join(segmented_text)

# Function to apply all cleaning steps
def clean_data(text):
    text = clean_text(text)
    text = clean_digits(text)
    text = reduce_repeated_characters(text)
    text = clean_morespace(text)
    text = segment_compound_word(text)
    text = expand_contractions(text)
    text = lemmatize_words(text)
    text = remove_stopwords(text)
    return text

# Clean the text
dfts['cleaned_text'] = dfts['tweet'].apply(clean_data)

y = dfts["label"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(dfts['cleaned_text']).toarray()

# Train the model
model = MultinomialNB(alpha=2)
model.fit(X_train, y)




# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))