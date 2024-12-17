# src/main/train.py
import os
from data_preprocessing import EmailPreprocessor
from feature_extraction import FeatureExtractor
from model import SpamClassifier
import joblib
import json

def train_model():
    # data path
    base_path = "../../data/enron_data/enron1"
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
    # data preprocessing
    preprocessor = EmailPreprocessor()
    raw_emails, processed_emails, labels = preprocessor.load_dataset(base_path)

    # feature extraction
    feature_extractor = FeatureExtractor()
    X_train, X_test, y_train, y_test = feature_extractor.fit_transform(raw_emails, labels)

<<<<<<< Updated upstream
    # train model
    classifier = SpamClassifier(model_type='naive_bayes')
    classifier.train(X_train, y_train)
    
    # evaluate model
    classifier.evaluate(X_test, y_test)
    
    # save model
    os.makedirs('../../models', exist_ok=True)
    joblib.dump(classifier, '../../models/spam_classifier.pkl')
    joblib.dump(feature_extractor, '../../models/feature_extractor.pkl')
=======
    # train and evaluate models with different algorithms
    models = ['naive_bayes', 'svm', 'random_forest']
    results = {}

    for model_type in models:
        print(f"\nTraining {model_type} classifier...")
        classifier = SpamClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        
        # evaluate model
        classifier.evaluate(X_test, y_test)
        
        # save results
        results[model_type] = {
            'model_type': model_type,
            'feature_count': X_train.shape[1]
        }

    # save best model (you can modify this to save based on performance)
    best_model = 'random_forest'  # or choose based on evaluation metrics
    final_classifier = SpamClassifier(model_type=best_model)
    final_classifier.train(X_train, y_train)

    # create models directory if it doesn't exist
    os.makedirs('../../models', exist_ok=True)
    
    # save model and feature extractor
    joblib.dump(final_classifier, '../../models/spam_classifier.pkl')
    joblib.dump(feature_extractor, '../../models/feature_extractor.pkl')
    
    # save model info
    with open('../../models/model_info.json', 'w') as f:
        json.dump(results, f, indent=4)
>>>>>>> Stashed changes

if __name__ == "__main__":
    train_model()