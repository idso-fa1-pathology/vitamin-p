# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    # Ensure y_true and y_pred have the same shape
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }