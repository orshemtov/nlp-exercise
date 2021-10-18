import json
import os
import re

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Classifier:
    def __init__(self):
        self.model = self.load_model()
        self.data = self.load_data()
        self.texts, self.labels = self.parse_data(self.data)
        self.texts_cleaned = self.clean_data(self.texts)
        self.train_set, self.test_set = self.train_test_split(
            self.texts_cleaned, self.labels)
        self.train_texts, self.train_labels = self.train_set
        self.test_texts, self.test_labels = self.test_set
        self.tokenizer = self.build_corpus(self.train_texts)

    def load_model(self, model_path=os.path.join('models', 'model.h5')):
        if not os.path.exists(model_path):
            raise FileNotFoundError('Please train the model first.')
        model = tf.keras.models.load_model(model_path)
        return model

    def load_data(self, data_path=os.path.join('data', 'data.json')):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def parse_data(self, data):
        texts = [post['post'] for post in data]
        labels = [post['label'] for post in data]

        return texts, labels

    def clean_data(self, texts):
        def clean_text(text):
            '''Clean sentences, remove punctuation, convert to lowercase, split and remove stopwords'''
            text = re.sub("[^a-zA-Z]", ' ', text)
            text = text.lower().split()
            swords = set(stopwords.words("english"))
            swords.add('https')
            text = [word for word in text if word not in swords]
            text = " ".join(text)

            return text

        texts_cleaned = list(map(clean_text, texts))

        return texts_cleaned

    def train_test_split(self, texts, labels, train_size=0.8):
        train_size = int(len(texts) * train_size)

        # Train
        train_texts = texts[:train_size]
        train_labels = labels[:train_size]

        # Test
        test_texts = texts[train_size:]
        test_labels = labels[train_size:]

        return (train_texts, train_labels), (test_texts, test_labels)

    def build_corpus(self, train_texts, vocab_size=10000, oov_token='<OOV>'):
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(train_texts)
        return tokenizer

    def tokenize(self,
                 sequence,
                 tokenizer,
                 max_length=250,
                 padding_type='post',
                 trunc_type='post'):
        sequence = tokenizer.texts_to_sequences(sequence)
        padded = pad_sequences(sequence,
                               padding=padding_type,
                               truncating=trunc_type,
                               maxlen=max_length)
        return padded

    def predict(self, texts):
        TOPICS = [
            'entertainment',
            'python',
            'science',
            'gaming',
            'books',
            'technology',
            'music',
            'politics',
            'sports',
            'travel',
        ]

        # Preprocess
        texts = self.clean_data(texts)
        padded = self.tokenize(texts, self.tokenizer)
        padded = np.array(padded, dtype='int32')

        # Make predictions
        predictions = self.model.predict(padded)
        predictions = [TOPICS[np.argmax(p)] for p in predictions]

        return predictions
