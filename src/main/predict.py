# src/main/predict.py

import joblib
from data_preprocessing import EmailPreprocessor

def predict_email(email_content):
    # load classifier and feature extractor
    classifier = joblib.load('../../models/spam_classifier.pkl')
    feature_extractor = joblib.load('../../models/feature_extractor.pkl')
    
    # preprocess email
    preprocessor = EmailPreprocessor()
    processed_email = preprocessor.preprocess_text(email_content)
    
    # feature extraction
    email_features = feature_extractor.transform([processed_email])
    
    # predict
    prediction = classifier.predict(email_features)
    
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    # test case
    test_email = "Get rich quick! Buy now!"
    result = predict_email(test_email)
    print(f"Prediction: this email is {result}")