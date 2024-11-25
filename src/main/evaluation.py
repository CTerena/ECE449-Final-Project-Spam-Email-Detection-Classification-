# src/main/evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging
import json
from datetime import datetime
import os
import joblib
import argparse
from data_preprocessing import EmailPreprocessor
from feature_extraction import FeatureExtractor

class ModelEvaluator:
    def __init__(self, model_type, output_dir=None):
        """
        Initialize the evaluator with model type and output directory.
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'svm', or 'random_forest')
            output_dir (str, optional): Directory to save evaluation results
        """
        self.model_type = model_type
        self.model_dir = f"../../models/{model_type}"
        self.output_dir = output_dir or f"../../evaluation_results/{model_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self, base_path):
        """Load and preprocess the dataset"""
        self.logger.info(f"Loading dataset from {base_path}")
        
        preprocessor = EmailPreprocessor()
        emails, labels = preprocessor.load_dataset(base_path)
        
        feature_extractor = joblib.load(os.path.join(self.model_dir, 'feature_extractor.pkl'))
        X = feature_extractor.transform(emails)
        
        return X, labels

    def evaluate_model(self, y_true, y_pred, y_prob=None):
        """Evaluate the model using multiple metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        self.logger.info(f"\nEvaluation Results for {self.model_type.upper()} classifier:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.output_dir, f'confusion_matrix_{self.model_type}.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Confusion matrix saved to {save_path}")

    def plot_roc_curve(self, y_true, y_prob):
        """Plot and save ROC curve if applicable"""
        if y_prob is None:
            self.logger.warning(f"ROC curve not available for {self.model_type}")
            return
            
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type.upper()}')
        plt.legend(loc="lower right")
        
        save_path = os.path.join(self.output_dir, f'roc_curve_{self.model_type}.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"ROC curve saved to {save_path}")

    def save_results(self, metrics):
        """Save evaluation results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'evaluation_results_{self.model_type}_{timestamp}.json'
        save_path = os.path.join(self.output_dir, filename)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"Evaluation results saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained spam classifier')
    parser.add_argument('--model_type', 
                      type=str, 
                      choices=['naive_bayes', 'svm', 'random_forest'],
                      default='naive_bayes',
                      help='Type of model to evaluate')
    return parser.parse_args()

def main():
    """Main function to run the evaluation"""
    args = parse_args()
    try:
        # Initialize evaluator with model type
        evaluator = ModelEvaluator(args.model_type)
        
        # Load and preprocess evaluation data
        base_path = "../../data/enron_data/enron2"
        X, y_true = evaluator.load_and_preprocess_data(base_path)
        
        # Load the trained model
        model_path = os.path.join(evaluator.model_dir, 'spam_classifier.pkl')
        classifier = joblib.load(model_path)
        
        # Make predictions
        y_pred = classifier.predict(X)
        
        # Get probabilities if available (not available for SVM)
        y_prob = None
        if args.model_type in ['naive_bayes', 'random_forest']:
            y_prob = classifier.model.predict_proba(X)[:, 1]
        
        # Run evaluation
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob)
        evaluator.plot_confusion_matrix(y_true, y_pred)
        if y_prob is not None:
            evaluator.plot_roc_curve(y_true, y_prob)
        evaluator.save_results(metrics)
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()