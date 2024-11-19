# src/main/train.py

import os
from data_preprocessing import EmailPreprocessor
from feature_extraction import FeatureExtractor
from model import SpamClassifier
import joblib

def train_model():
    # data path
    base_path = "../../data/enron_data/enron1"
    
    # data preprocessing
    preprocessor = EmailPreprocessor()
    emails, labels = preprocessor.load_dataset(base_path)
    
    # feature extraction
    feature_extractor = FeatureExtractor()
    X_train, X_test, y_train, y_test = feature_extractor.fit_transform(emails, labels)

    # train model
    classifier = SpamClassifier(model_type='naive_bayes')
    classifier.train(X_train, y_train)
    
    # evaluate model
    classifier.evaluate(X_test, y_test)
    
    # save model
    os.makedirs('../../models', exist_ok=True)
    joblib.dump(classifier, '../../models/spam_classifier.pkl')
    joblib.dump(feature_extractor, '../../models/feature_extractor.pkl')

if __name__ == "__main__":
    train_model()