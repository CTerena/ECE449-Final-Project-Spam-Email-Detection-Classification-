# src/main/predict.py
import joblib
from data_preprocessing import EmailPreprocessor
import numpy as np

def predict_email(email_content):
    # load classifier and feature extractor
    classifier = joblib.load('../../models/spam_classifier.pkl')
    feature_extractor = joblib.load('../../models/feature_extractor.pkl')

    # preprocess email
    preprocessor = EmailPreprocessor()
    processed_email = preprocessor.preprocess_text(email_content, preserve_features=True)

    # feature extraction
    email_features = feature_extractor.transform([email_content])

    # get prediction and probability
    prediction = classifier.predict(email_features)
    probability = classifier.predict_proba(email_features)

    # prepare result
    result = {
        'classification': "Spam" if prediction[0] == 1 else "Ham",
        'confidence': float(probability[0]),
        'features_used': email_features.shape[1]
    }

    return result

if __name__ == "__main__":
    # test case
    test_email = """
    Get rich quick! Buy now! 
    SPECIAL OFFER! Don't miss this AMAZING opportunity!
    Click here: http://example.com
    Contact us: test@example.com
    """
    result = predict_email(test_email)
    print("\nPrediction Results:")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Number of features used: {result['features_used']}")