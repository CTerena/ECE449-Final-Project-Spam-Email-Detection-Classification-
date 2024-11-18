# src/main/feature_extraction.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit_transform(self, emails, labels, test_size=0.2):
        """转换文本特征并划分数据集"""
        # 转换文本为TF-IDF特征
        X = self.vectorizer.fit_transform(emails)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, emails):
        """转换新的文本数据"""
        return self.vectorizer.transform(emails)