import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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

def evaluate(pred: np.ndarray, proba: np.ndarray, ground_truth: np.ndarray) -> dict:
    report = classification_report(ground_truth, pred, output_dict=True)
    # Convert any NumPy types in the report to native Python types
    report = {k: {kk: (float(vv) if isinstance(vv, (np.float32, np.float64)) else vv)
                for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in report.items()}

    auc = float(roc_auc_score(ground_truth, proba))
    cm = confusion_matrix(ground_truth, pred).tolist()

    return {
        "classification_report": report,
        "roc_auc": auc,
        "confusion_matrix": cm
    }