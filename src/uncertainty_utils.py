import os, sys
import yaml
import argparse
import numpy as np
import geopandas as gpd
import pandas as pd

# Brier score, binary and multiclass
# Negative log likelihood, binary and multiclass
# Expected calibration error, binary
# Prediction entropy and average prediction entropy, binary or multiclass
# Uncertainty-aware accuracy, binary or multiclass


def brier_score_binary(y_true, y_pred):
    """
    Calculate the Brier Score for binary classification. Between 0 (perfect accuracy) and 1.
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true binary labels (0 or 1).
    - y_pred: np.ndarray of shape (n_samples,), predicted probabilities for the positive class.
    
    Returns:
    - brier_score: float, the Brier Score for binary classification.
    """
    # Ensure predictions are between 0 and 1
    # Calculate the mean squared error
    brier_score = np.mean((y_pred - y_true) ** 2)
    return brier_score

def brier_score_multiclass(y_true, y_pred):
    """
    Calculate the Brier Score for multiclass classification. Between 0 and 2. 
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true class labels (integer encoded).
    - y_pred: np.ndarray of shape (n_samples, n_classes), predicted probabilities for each class.
    
    Returns:
    - brier_score: float, the Brier Score for multiclass classification.
    """
    # One-hot encode the true labels
    n_samples, n_classes = y_pred.shape
    dict_class = {A: B for A, B in zip(np.unique(y_true), np.arange(len(y_true)))}
    # print(f"The classes have been remapped {dict_class}.")
    # y_true.replace(dict_class,inplace=True)
    for key, val in dict_class.items():
        y_true[y_true == key] = val

    y_true_one_hot = np.zeros_like(y_pred).astype('int64')
    y_true_one_hot[np.arange(n_samples), np.array(y_true)] = 1
    
    # Calculate the mean squared error across all classes and samples
    brier_score = np.mean((y_pred - y_true_one_hot) ** 2)
    return brier_score

def expected_calibration_error(y_true, y_pred_lbl, y_pred, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    The lower the better. 
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true class labels (integer encoded).
    - y_pred: np.ndarray of shape (n_samples, n_classes), predicted probabilities for each class.
    - num_bins: int, number of bins to use for calibration.

    Returns:
    - ece: float, the Expected Calibration Error.
    """
    # Get the predicted probabilities and predicted classes
    y_true = np.array(y_true)
    confidences = np.max(y_pred, axis=1)
    predictions = np.array(y_pred_lbl)

    # Initialize ECE
    ece = 0.0
    bin_size = 1.0 / num_bins

    for i in range(num_bins):
        # Define the bin range
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size
        
        # Get indices for samples with confidence within the bin range
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        # If there are samples in this bin
        if np.sum(in_bin) > 0:
            # Calculate the accuracy and average confidence for samples in this bin
            bin_accuracy = np.mean(predictions[in_bin] == y_true[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            # Update ECE with the weighted difference between accuracy and confidence
            ece += np.abs(bin_accuracy - bin_confidence) * np.mean(in_bin)
    
    return ece


def average_prediction_entropy(pred_probs):
    """
    Calculate the average prediction entropy for multiple samples.
    In a binary classification problem, the prediction entropy can range from 0 to approximately 0.693
    For 3 classes, it ranges from 0 to 1.10
    
    Parameters:
    - pred_probs: np.ndarray of shape (n_samples,), predicted probabilities for the positive class.
    
    Returns:
    - avg_entropy: float, average entropy across all predictions.
    """
    def prediction_entropy(p):
        """
        In binary classification:
        (0.5, 0.5): 0.693
        (0.75, 0.25): 0.56
        (1, 0): 0
        
        With three classes: 
        (0.33, 0.33, 0.33): 1.10
        (0.15,0.7,0.15): 0.81
        (1,0,0): 0
        """
        p = np.array(p)
        p = np.clip(p, 1e-10, 1)
        entropy_values = -np.sum(p * np.log(p))
        return entropy_values
    
    # entropies = [prediction_entropy(row) for row in pred_probs.itertuples(index=False)]
    entropies = [prediction_entropy(pred_probs[row,:]) for row in range(pred_probs.shape[0])]
    avg_entropy = np.mean(entropies)
    return avg_entropy

def binary_nll(y_true, y_pred):
    """
    Compute the Negative Log Likelihood for binary classification.
    A common baseline target for binary classification is to aim for a NLL bellow 0.3, but this varies based on the problem. -log(0.5) = 0.69
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true binary labels (0 or 1).
    - y_pred: np.ndarray of shape (n_samples,), predicted probabilities for the positive class.
    
    Returns:
    - nll: float, average negative log likelihood.
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1)
    nll = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return nll

def multiclass_nll(y_true, y_pred):
    """
    Compute the Negative Log Likelihood for multiclass classification.
    For a model with 3 classes, a "good" NLL value is ideally below ≈1.10, since −log⁡(1/3)≈1.10. The worst case, all probabilities are 0 leads to infinity. -log(0.5) = 0.69
    
    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true class labels (integer encoded).
    - y_pred: np.ndarray of shape (n_samples, n_classes), predicted probabilities for each class.
    
    Returns:
    - nll: float, average negative log likelihood.
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    
    # Select the probabilities corresponding to the correct classes
    dict_class = {A: B for A, B in zip(np.unique(y_true), np.arange(len(y_true)))}
    # y_true.replace(dict_class)
    for key, val in dict_class.items():
        y_true[y_true == key] = val
    # print(f"The classes have been remapped {dict_class}.")
    correct_class_probs = np.array(y_pred)[np.arange(len(y_true)), y_true]

    nll = -np.mean(np.log(correct_class_probs))
    return nll

def uncertainty_aware_accuracy(predicted_labels, pred_probs, true_labels, confidence_threshold=0.7):
    """
    Calculate uncertainty-aware accuracy by filtering out low-confidence predictions.
    
    Parameters:
    - pred_probs: np.ndarray of shape (n_samples, n_classes), model's predicted probabilities.
    - true_labels: np.ndarray of shape (n_samples,), true class labels.
    - confidence_threshold: float, minimum confidence to accept a prediction.

    Returns:
    - accuracy: float, uncertainty-aware accuracy.
    - n_confident_predictions: int, number of predictions above the confidence threshold.
    """
    # Get predicted labels and confidence (maximum probability) for each prediction
    confidence_scores = np.max(pred_probs, axis=1)
    
    # Filter for predictions with confidence above the threshold
    confident_indices = confidence_scores >= confidence_threshold
    confident_predictions = predicted_labels[confident_indices]
    confident_true_labels = true_labels[confident_indices]
    
    # Calculate accuracy only on confident predictions
    if len(confident_predictions) == 0:
        print("No predictions met the confidence threshold.")
        return 0.0, 0  # Return 0 accuracy if no predictions are confident enough
    
    accuracy = np.mean(confident_predictions == confident_true_labels)
    n_confident_predictions = len(confident_predictions)
    
    return accuracy, round(n_confident_predictions/len(true_labels),2)
