from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, KFold
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib


class LightGBMConfig(BaseModel):
    feature_cols: list[str] = [
        "text1_mean_entropy", "text1_sum_entropy", "text1_n_tokens",
        "text2_mean_entropy", "text2_sum_entropy", "text2_n_tokens",
        "delta_mean_entropy", "delta_sum_entropy", "delta_n_tokens"
    ]
    target_col: str = "label"
    params: dict = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 1000
    }
    test_ratio: float = 0.2
    random_state: int = 42
    early_stopping_rounds: int = 50
    log_eval_period: int = 50
    n_splits_oof: int = 5


class LightGBMClassifierModel:
    def __init__(self, ds_feats: pd.DataFrame, ds_embs: np.ndarray, config: LightGBMConfig | None = None):
        self.config = config or LightGBMConfig()
        self.X = ds_feats[self.config.feature_cols]
        self.y = ds_feats[self.config.target_col]
        self.embeddings = ds_embs
        self.model = LGBMClassifier(**self.config.params)

        # Placeholders for train/test splits
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.emb_train = self.emb_test = None

    def split_data(self):
        """Split features, labels, and embeddings into train/test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test, self.emb_train, self.emb_test = train_test_split(
            self.X, self.y, self.embeddings,
            test_size=self.config.test_ratio,
            stratify=self.y,
            random_state=self.config.random_state
        )

    def fit(self) -> None:
        """Fit LightGBM classifier with early stopping and evaluation."""
        if self.X_train is None:
            self.split_data()

        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            eval_metric="auc",
            callbacks=[
                early_stopping(self.config.early_stopping_rounds),
                log_evaluation(period=self.config.log_eval_period)
            ]
        )

    def predict(self, test_set: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict on the a set."""
        y_pred = self.model.predict(test_set)
        y_proba = self.model.predict_proba(test_set)[:, 1]
        return y_pred, y_proba

    def save_model(self, path: str) -> None:
        """Save trained LightGBM model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def save_embeddings(self, path: str) -> None:
        """Save train/test embeddings separately."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, train=self.emb_train, test=self.emb_test)

    def load_model(self, path: str) -> None:
        """Loads a pre-trained model from a joblib file."""
        self.model = joblib.load(path)

    def oof_predict(self, n_splits: int | None = None, random_state: int | None = None) -> np.ndarray:
        """
        Compute out-of-fold (OOF) predictions on the training data using K-Fold CV.

        Args:
            n_splits: Number of folds (defaults to config.n_splits_oof)
            random_state: Random state for reproducibility (defaults to config.random_state)

        Returns:
            oof_probs: numpy array of OOF probabilities (shape = len(X_train))
        """
        if self.X_train is None:
            self.split_data()

        n_splits = n_splits or self.config.n_splits_oof
        random_state = random_state or self.config.random_state

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_probs = np.zeros(len(self.X_train))

        for train_idx, val_idx in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = LGBMClassifier(**self.config.params)
            model.fit(X_tr, y_tr)
            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        return oof_probs
