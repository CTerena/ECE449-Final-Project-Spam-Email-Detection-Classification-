# src/main/model.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class SpamClassifier:
    def __init__(self, model_type='naive_bayes'):
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        """train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """predict X"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """evaluate the model"""
        predictions = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))