# src/main/data_preprocessing.py

import os
import email
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class EmailPreprocessor:
    def __init__(self):
        # 下载必要的nltk数据
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def read_email(self, file_path):
        """读取单个邮件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except:
            return ""
    
    def preprocess_text(self, text):
        # turn characters to lower ones
        text = text.lower()
        # remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # tokenize
        tokens = word_tokenize(text)
        # remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def load_dataset(self, base_path):
        """load dataset (enron1)"""
        ham_path = os.path.join(base_path, 'ham')
        spam_path = os.path.join(base_path, 'spam')
        
        emails = []
        labels = []
        
        # load normal email
        for filename in os.listdir(ham_path):
            if filename.endswith('.txt'):
                content = self.read_email(os.path.join(ham_path, filename))
                if content:
                    emails.append(self.preprocess_text(content))
                    labels.append(0)
        
        # load spam email
        for filename in os.listdir(spam_path):
            if filename.endswith('.txt'):
                content = self.read_email(os.path.join(spam_path, filename))
                if content:
                    emails.append(self.preprocess_text(content))
                    labels.append(1)
        
        return emails, labels