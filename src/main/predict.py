# src/main/predict.py

import joblib
from data_preprocessing import EmailPreprocessor

def predict_email(email_content):
    # 加载模型和特征提取器
    classifier = joblib.load('../../models/spam_classifier.pkl')
    feature_extractor = joblib.load('../../models/feature_extractor.pkl')
    
    # 预处理邮件
    preprocessor = EmailPreprocessor()
    processed_email = preprocessor.preprocess_text(email_content)
    
    # 特征转换
    email_features = feature_extractor.transform([processed_email])
    
    # 预测
    prediction = classifier.predict(email_features)
    
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    # 测试示例
    test_email = "Get rich quick! Buy now!"
    result = predict_email(test_email)
    print(f"预测结果: {result}")