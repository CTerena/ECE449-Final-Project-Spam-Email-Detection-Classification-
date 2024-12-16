# src/main/train.py

import os
import argparse
from data_preprocessing import EmailPreprocessor
from feature_extraction import FeatureExtractor
from model import SpamClassifier
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Train spam classifier with specified model type')
    parser.add_argument('--model_type', 
                      type=str, 
                      choices=['naive_bayes', 'svm', 'random_forest'],
                      default='naive_bayes',
                      help='Type of model to train')
    parser.add_argument('--feature', 
                      type=str, 
                      choices=['glove', 'tfidf'],
                      default='tfidf',
                      help='Type of feature extraction')
    return parser.parse_args()

def train_model(model_type, feature):

    # Base path for the dataset
    base_path = "../../data/enron_data/enron1"

    model_dir = f"../../models/{model_type}/{feature}"
    os.makedirs(model_dir, exist_ok=True)
    
     # Initialize preprocessor and feature extractor
    preprocessor = EmailPreprocessor()
    emails, labels = preprocessor.load_dataset(base_path)
    
    if(feature == "glove"):
        embedding_path = "../../glove.6B/glove.6B.50d.txt"
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(feature, glove_file=embedding_path)

        # Transform emails and split the dataset
        X_train, X_test, y_train, y_test = feature_extractor.fit_transform(emails, labels)
    else:
        # feature extraction
        feature_extractor = FeatureExtractor(feature)
        X_train, X_test, y_train, y_test = feature_extractor.fit_transform(emails, labels)
    
    # train model
    print(f"\nTraining {model_type} classifier...")
    classifier = SpamClassifier(model_type=model_type)
    classifier.train(X_train, y_train)
    
    # evaluate model
    print(f"\nEvaluating {model_type} classifier...")
    classifier.evaluate(X_test, y_test)
    
    # save model and feature extractor
    model_path = os.path.join(model_dir, 'spam_classifier.pkl')
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
    
    print(f"\nSaving model to {model_path}")
    joblib.dump(classifier, model_path)
    joblib.dump(feature_extractor, feature_extractor_path)
    
    return X_test, y_test  # Return test data for potential additional evaluation

if __name__ == "__main__":
    args = parse_args()
    train_model(args.model_type, args.feature)