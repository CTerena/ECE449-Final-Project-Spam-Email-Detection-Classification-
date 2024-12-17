# src/main/data_preprocessing.py
import os
import email
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class EmailPreprocessor:
    def __init__(self):
        # download necessary nltk data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def read_email(self, file_path):
        """read content of a single email"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except:
            return ""

    def preprocess_text(self, text, preserve_features=True):
        """
        Preprocess text with option to preserve features for feature engineering
        """
        if preserve_features:
            # Keep original text for feature engineering
            self.original_text = text
            # Only lowercase and basic cleaning for feature extraction
            processed_text = text.lower()
            return processed_text
        else:
            # Traditional preprocessing for TF-IDF
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            return ' '.join(tokens)

    def load_dataset(self, base_path):
        """load dataset (enron1)"""
        ham_path = os.path.join(base_path, 'ham')
        spam_path = os.path.join(base_path, 'spam')
        
        raw_emails = []  # Store original emails
        processed_emails = []  # Store processed emails
        labels = []
        
        # load normal emails
        for filename in os.listdir(ham_path):
            if filename.endswith('.txt'):
                content = self.read_email(os.path.join(ham_path, filename))
                if content:
                    raw_emails.append(content)
                    processed_emails.append(self.preprocess_text(content, preserve_features=True))
                    labels.append(0)
        
        # load spam emails
        for filename in os.listdir(spam_path):
            if filename.endswith('.txt'):
                content = self.read_email(os.path.join(spam_path, filename))
                if content:
                    raw_emails.append(content)
                    processed_emails.append(self.preprocess_text(content, preserve_features=True))
                    labels.append(1)
        
        return raw_emails, processed_emails, labels