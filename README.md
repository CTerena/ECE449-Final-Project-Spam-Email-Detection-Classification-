# 🚀 Spam Email Detection System

## 📋 Overview
This project is the final project for ECE449: Machine Learning course at ZJUI (Zhejiang University - University of Illinois at Urbana-Champaign Institute) in Fall 2024. It implements a machine learning-based spam email detection system using Python, which processes raw email data, extracts features, and trains different classifiers to identify spam emails.

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
git clone https://github.com/CTerena/ECE449_FinalProject_Spam-Email-Detection-Classification.git
cd ECE449_FinalProject_Spam-Email-Detection-Classification

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## 📁 Project Structure
```
src/main/
├── data_preprocessing.py    # Data preprocessing utilities
├── evaluation.py            # Evaluation script
├── feature_extraction.py    # Feature extraction methods
├── model.py                 # Model definitions
├── train.py                 # Training script
└── predict.py               # Prediction script
```
## 🚀 Quick Start
Train the model:
```bash
cd src/main
python train.py --model_type 
```
Evaluate the model:
```bash
python evaluate.py --model_type 
```
Make predictions:
```bash
python predict.py
```

## 💻 Usage Example
```python
from predict import predict_email

# Test with a sample email
email_content = "Get rich quick! Buy now!"
result = predict_email(email_content, model_name)
print(f"Prediction: {result}")
```
## 📊 Model Performance
The system provides detailed performance metrics including:

Classification Report
Confusion Matrix
Specificity and Sensitivity scores
## 🔧 Requirements
- Python 3.7+
- scikit-learn
- NLTK
- NumPy
- joblib

## 📝 Dependencies
```txt
scikit-learn>=1.0.2
numpy>=1.22.0
scipy>=1.7.3
nltk>=3.6.7
joblib>=1.1.0
```
