# src/main/feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import re
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def extract_statistical_features(self, email):
        """Extract statistical features from email text"""
        features = []
        
        # 1. Length features
        features.append(len(email))  # Total length
        features.append(len(email.split()))  # Word count
        
        # 2. Special character features
        features.append(len(re.findall(r'[!]', email)))  # Count of !
        features.append(len(re.findall(r'[$]', email)))  # Count of $
        features.append(len(re.findall(r'[%]', email)))  # Count of %
        
        # 3. Capital letter features
        words = email.split()
        capitals = sum(1 for word in words if word.isupper())
        features.append(capitals)  # Count of words in ALL CAPS
        features.append(capitals / (len(words) + 1))  # Ratio of capital words
        
        # 4. URL and email features
        urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email))
        emails = len(re.findall(r'[\w\.-]+@[\w\.-]+', email))
        features.append(urls)  # Count of URLs
        features.append(emails)  # Count of email addresses
        
        return features

    def extract_punctuation_features(self, email):
        """Extract punctuation-related features"""
        features = []
        
        # Common punctuation marks
        punctuation_marks = ['?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'"]
        for mark in punctuation_marks:
            features.append(email.count(mark))
            
        # Sequences of punctuation
        features.append(len(re.findall(r'[!]{2,}', email)))  # Multiple !
        features.append(len(re.findall(r'[?]{2,}', email)))  # Multiple ?
        
        return features

    def extract_word_based_features(self, email):
        """Extract word-based features"""
        features = []
        words = word_tokenize(email.lower())
        
        # Common spam words (you can expand this list)
        spam_words = ['free', 'winner', 'won', 'prize', 'urgent', 'offer', 'limited', 'money', 'click', 'guarantee']
        for word in spam_words:
            features.append(sum(1 for w in words if w == word))
            
        # Average word length
        avg_word_length = sum(len(word) for word in words) / (len(words) + 1)
        features.append(avg_word_length)
        
        return features

    def fit_transform(self, emails, labels, test_size=0.2):
        """Transform text data to features"""
        # 1. TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(emails)
        
        # 2. Statistical features
        statistical_features = np.array([self.extract_statistical_features(email) for email in emails])
        
        # 3. Punctuation features
        punctuation_features = np.array([self.extract_punctuation_features(email) for email in emails])
        
        # 4. Word-based features
        word_features = np.array([self.extract_word_based_features(email) for email in emails])
        
        # Combine all features
        additional_features = np.hstack([statistical_features, punctuation_features, word_features])
        X_additional = additional_features
        
        # Combine TF-IDF with additional features
        X = hstack([X_tfidf, X_additional])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

    def transform(self, emails):
        """Transform new text data to features"""
        # 1. TF-IDF features
        X_tfidf = self.vectorizer.transform(emails)
        
        # 2. Statistical features
        statistical_features = np.array([self.extract_statistical_features(email) for email in emails])
        
        # 3. Punctuation features
        punctuation_features = np.array([self.extract_punctuation_features(email) for email in emails])
        
        # 4. Word-based features
        word_features = np.array([self.extract_word_based_features(email) for email in emails])
        
        # Combine all features
        additional_features = np.hstack([statistical_features, punctuation_features, word_features])
        X_additional = additional_features
        
        # Combine TF-IDF with additional features
        X = hstack([X_tfidf, X_additional])
        
        return X