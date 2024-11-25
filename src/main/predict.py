# src/main/predict.py

import joblib
import argparse
from data_preprocessing import EmailPreprocessor

def predict_email(email_content, model_type):
    # load classifier and feature extractor
    model_dir = f"../../models/{model_type}"
    classifier = joblib.load(model_dir+'/spam_classifier.pkl')
    feature_extractor = joblib.load(model_dir+'/feature_extractor.pkl')
    
    # preprocess email
    preprocessor = EmailPreprocessor()
    processed_email = preprocessor.preprocess_text(email_content)
    
    # feature extraction
    email_features = feature_extractor.transform([processed_email])
    
    # predict
    prediction = classifier.predict(email_features)
    
    return "Spam" if prediction[0] == 1 else "Ham"

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained spam classifier')
    parser.add_argument('--model_type', 
                      type=str, 
                      choices=['naive_bayes', 'svm', 'random_forest'],
                      default='naive_bayes',
                      help='Type of model to evaluate')
    return parser.parse_args()

if __name__ == "__main__":
    # test case
    test_email = "Get rich quick! Buy now!"
    args = parse_args()
    result = predict_email(test_email, args.model_type)
    print(f"Prediction: this email is {result}")