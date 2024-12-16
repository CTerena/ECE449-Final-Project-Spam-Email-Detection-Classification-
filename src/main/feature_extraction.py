import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    def __init__(self, method="tfidf", max_features=5000, glove_file=None, embedding_dim=50):
        """
        Initialize the feature extractor.

        Parameters:
        - method (str): "tfidf" or "glove" to select the feature extraction method.
        - max_features (int): Number of features for TF-IDF (only used if method="tfidf").
        - glove_file (str): Path to GloVe embeddings file (only used if method="glove").
        - embedding_dim (int): Dimensionality of GloVe embeddings (only used if method="glove").
        """
        self.method = method
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        elif method == "glove":
            if not glove_file:
                raise ValueError("GloVe file path must be provided for GloVe method.")
            self.embedding_dim = embedding_dim
            self.glove_embeddings = self.load_glove_embeddings(glove_file)
        else:
            raise ValueError("Invalid method. Choose 'tfidf' or 'glove'.")

    def load_glove_embeddings(self, glove_file):
        """Load GloVe embeddings from a file."""
        embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def text_to_vector(self, text):
        """Convert a text to a vector using GloVe embeddings."""
        words = text.split()
        word_vectors = [self.glove_embeddings.get(word, np.zeros(self.embedding_dim)) for word in words]
        return np.mean(word_vectors, axis=0)  # Average the word vectors

    def fit_transform(self, emails, labels, test_size=0.2):
        """
        Transform text data to features and split into training and testing sets.

        Parameters:
        - emails (list of str): List of email texts.
        - labels (list): Corresponding labels for the emails.
        - test_size (float): Proportion of data to include in the test split.

        Returns:
        - X_train, X_test, y_train, y_test: Split feature and label data.
        """
        if self.method == "tfidf":
            X = self.vectorizer.fit_transform(emails)
        elif self.method == "glove":
            X = np.array([self.text_to_vector(email) for email in emails])
        else:
            raise ValueError("Invalid method. Choose 'tfidf' or 'glove'.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def transform(self, emails):
        """Transform new text data to features."""
        if self.method == "tfidf":
            return self.vectorizer.transform(emails)
        elif self.method == "glove":
            return np.array([self.text_to_vector(email) for email in emails])
        else:
            raise ValueError("Invalid method. Choose 'tfidf' or 'glove'.")
