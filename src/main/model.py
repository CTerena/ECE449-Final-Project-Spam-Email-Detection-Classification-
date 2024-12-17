# src/main/model.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

class SpamClassifier:
    def __init__(self, model_type='naive_bayes'):
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_type = model_type

    def train(self, X_train, y_train):
        """train the model"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """predict X"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions"""
        if self.model_type == 'svm':
            # For SVM, convert decision function to probabilities
            decision_values = self.model.decision_function(X)
            return 1/(1 + np.exp(-decision_values))
        else:
            return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """evaluate the model with enhanced metrics"""
        predictions = self.predict(X_test)
        probas = self.predict_proba(X_test)

        # Basic metrics
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('../../models/roc_curve.png')
        plt.close()