# src/main/train.py

import os
from data_preprocessing import EmailPreprocessor
from feature_extraction import FeatureExtractor
from model import SpamClassifier
import joblib

def train_model():
    # 设置数据路径
    base_path = "../../data/enron_data/enron1"
    
    # 数据预处理
    preprocessor = EmailPreprocessor()
    emails, labels = preprocessor.load_dataset(base_path)
    
    # 特征提取
    feature_extractor = FeatureExtractor()
    X_train, X_test, y_train, y_test = feature_extractor.fit_transform(emails, labels)
    
    # 训练模型
    classifier = SpamClassifier(model_type='naive_bayes')
    classifier.train(X_train, y_train)
    
    # 评估模型
    classifier.evaluate(X_test, y_test)
    
    # 保存模型和特征提取器
    os.makedirs('../../models', exist_ok=True)
    joblib.dump(classifier, '../../models/spam_classifier.pkl')
    joblib.dump(feature_extractor, '../../models/feature_extractor.pkl')

if __name__ == "__main__":
    train_model()