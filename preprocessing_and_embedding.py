#!/home/peiran/anaconda3/bin/python

# embedding matrix 
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# define constant
INPUT_DIR = "./input/"
MAX_SEQUENCE_LENGTH = 30
EMB_DIM = 300
seed = 1

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = str(text).lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\<", " < ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, word stemming
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

X_train = pd.read_csv(INPUT_DIR+"X_train.csv")
X_test = pd.read_csv(INPUT_DIR+"X_test.csv")

X_train_q1 = X_train["question1"]
X_train_q2 = X_train["question2"]
X_test_q1 = X_test["question1"]
X_test_q2 = X_test["question2"]

X_train_q1 = X_train_q1.apply(text_to_wordlist)
X_train_q2 = X_train_q2.apply(text_to_wordlist)
X_test_q1 = X_test_q1.apply(text_to_wordlist)
X_test_q2 = X_test_q2.apply(text_to_wordlist)

# prepare string sequence
text_sequence = pd.Series(X_train_q1.tolist()+X_train_q2.tolist()
                      +X_test_q1.tolist()+X_test_q2.tolist()).tolist()
w2v_input = pd.Series(X_train_q1.tolist()+X_train_q2.tolist()
                      +X_test_q1.tolist()+X_test_q2.tolist()).apply(lambda x: x.split()).tolist()
word_set = set(" ".join(text_sequence).split())

# word embedding dict
w2v_model = models.Word2Vec(w2v_input, size=EMB_DIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)

# tokenize strings to 
tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(text_sequence)

# transform to string index
train_q1 = tokenizer.texts_to_sequences(X_train_q1.tolist())
train_q2 = tokenizer.texts_to_sequences(X_train_q2.tolist())
test_q1 = tokenizer.texts_to_sequences(X_test_q1.tolist())
test_q2 = tokenizer.texts_to_sequences(X_test_q2.tolist())

# padding sequence
train_pad_q1 = pad_sequences(train_q1, maxlen=30)
train_pad_q2 = pad_sequences(train_q2, maxlen=30)
test_pad_q1 = pad_sequences(test_q1, maxlen=30)
test_pad_q2 = pad_sequences(test_q2, maxlen=30)

# embedding matrix
embedding_mat = np.zeros([len(tokenizer.word_index)+1, EMB_DIM])

for word, idx in tokenizer.word_index.items():
    embedding_mat[idx,:] = w2v_model.wv[word]

# save the padded sequence and embedding matrix
np.save(INPUT_DIR+"embedding.npy",embedding_mat)
np.save(INPUT_DIR+"train_pad_q1.npy",train_pad_q1)
np.save(INPUT_DIR+"train_pad_q2.npy",train_pad_q2)
np.save(INPUT_DIR+"test_pad_q1.npy",test_pad_q1)
np.save(INPUT_DIR+"test_pad_q2.npy",test_pad_q2)
