# 🚀 Spam Email Detection System

## 📋 Overview
This project implements a machine learning-based spam email detection system using Python. It processes raw email data, extracts features, and trains different classifiers to identify spam emails.

## ✨ Features
- 📧 Email preprocessing and cleaning
- 🔍 Text feature extraction using TF-IDF
- 🤖 Multiple classification models:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- 📊 Model evaluation and performance metrics
- 🔄 Easy-to-use prediction interface

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
