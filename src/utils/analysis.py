import numpy as np
from sklearn.linear_model import LogisticRegression

def platt_scaled_predictions(origin_probs: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Apply Platt scaling to calibrate predicted probabilities.

    Args:
        origin_probs (np.ndarray): model predicted probabilities (shape: [n_samples])
        true_labels (np.ndarray): true binary labels (0 or 1)
    
    Returns:
        np.ndarray: calibrated probabilities
    """
    # Avoid numerical issues with exact 0 or 1
    eps = 1e-12
    origin_probs = np.clip(origin_probs, eps, 1 - eps)
    
    # Convert probabilities to logits
    logits = np.log(origin_probs / (1 - origin_probs)).reshape(-1, 1)
    
    # Fit logistic regression (Platt scaling)
    platt = LogisticRegression(fit_intercept=True, solver='lbfgs')
    platt.fit(logits, true_labels)
    
    # Predict calibrated probabilities
    calibrated_probs = platt.predict_proba(logits)[:, 1]
    return calibrated_probs