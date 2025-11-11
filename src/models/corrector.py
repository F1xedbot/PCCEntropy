from pydantic import BaseModel
import numpy as np


class CorrectorConfig(BaseModel):
    alpha: float = 0.6
    f_min: float = 0.1
    p_min: float = 0.2


class CounterfactualCorrector:
    """
    A lightweight post-hoc corrector that evaluates model prediction stability
    using neighborhood similarities and counterfactual reasoning.

    It computes several metrics:
        - Stability Score (SS): How consistent a prediction is among similar examples.
        - Counterfactual Proximity (CP): How close the most dissimilar prediction is.
        - Fragility Index (FI): Weighted blend of instability and contradiction potential.
        - Neighbor Mean (NM): Similarity-weighted mean probability of neighbors.
        - Adjusted Probability: Softly adjusted probability considering FI and NM.

    The class also decides whether to **trust** the current prediction or **escalate**
    to a language model based on confidence and fragility thresholds.
    """

    def __init__(self, config: CorrectorConfig | None = None):
        self.config = config or CorrectorConfig()

    def compute_stability_score(self, p: float, neigh_p: np.ndarray, sims: np.ndarray) -> float:
        """
        Compute the stability score (SS): 1 - weighted mean absolute difference.

        Args:
            p: Model's original probability.
            neigh_p: Neighbor probabilities.
            sims: Similarity weights between neighbors and query (non-negative).

        Returns:
            Stability score between 0 (unstable) and 1 (stable).
        """
        sims_sum = np.sum(sims)
        if sims_sum == 0:
            return 0.0
        diff = np.abs(neigh_p - p)
        return 1.0 - float(np.sum(sims * diff) / sims_sum)

    def compute_counterfactual_proximity(self, p: float, neigh_p: np.ndarray, sims: np.ndarray) -> float:
        """
        Compute counterfactual proximity (CP): potential influence of opposite predictions.

        Args:
            p: Model probability.
            neigh_p: Neighbor probabilities.
            sims: Similarity weights.

        Returns:
            Weighted maximum of opposite neighbor influence (0-1 range).
        """
        p_opp = 1.0 - p
        neigh_opp = 1.0 - neigh_p
        diffs = neigh_opp - p_opp

        mask = diffs > 0
        if not np.any(mask):
            return 0.0

        pos_diffs = diffs[mask]
        pos_sims = sims[mask]
        return float(np.max(pos_sims * pos_diffs))

    def compute_neighbor_mean(self, neigh_p: np.ndarray, sims: np.ndarray | None = None) -> float:
        """
        Compute a (possibly similarity-weighted) neighbor mean probability.

        Args:
            neigh_p: Neighbor probabilities.
            sims: Optional similarity weights.

        Returns:
            Weighted or unweighted neighbor mean.
        """
        if sims is not None and np.sum(sims) > 0:
            return float(np.sum(neigh_p * sims) / np.sum(sims))
        return float(np.mean(neigh_p))

    def compute_fragility_index(self, ss: float, cp: float) -> float:
        """
        Compute the fragility index (FI), balancing instability and contradiction.

        Args:
            ss: Stability score.
            cp: Counterfactual proximity.

        Returns:
            Fragility index between 0 (stable) and 1 (fragile).
        """
        alpha = self.config.alpha
        fi = alpha * (1 - ss) + (1 - alpha) * cp
        return float(np.clip(fi, 0.0, 1.0))

    def is_confident(self, p: float, fi: float) -> bool:
        """
        Determine whether a prediction is considered confident.

        Args:
            p: Adjusted or raw probability.
            fi: Fragility index.

        Returns:
            True if confidence is strong, else False.
        """
        confident_prob = np.abs(p - 0.5) >= self.config.p_min
        stable_fi = fi <= self.config.f_min
        return bool(confident_prob or stable_fi)

    def compute_cf_metrics(self, p: float, neigh_p: np.ndarray, sims: np.ndarray) -> dict:
        """
        Compute all counterfactual fragility metrics and adjusted prediction.

        Args:
            p: Model's raw predicted probability (0-1).
            neigh_p: Neighbor probabilities.
            sims: Similarity weights for neighbors.

        Returns:
            A dictionary containing:
                - ss: Stability score
                - cp: Counterfactual proximity
                - fi: Fragility index
                - nm: Neighbor mean
                - adjusted_prob: Smoothed probability
                - use_llm: Whether external correction is recommended
        """
        ss = self.compute_stability_score(p, neigh_p, sims)
        cp = self.compute_counterfactual_proximity(p, neigh_p, sims)
        fi = self.compute_fragility_index(ss, cp)
        nm = self.compute_neighbor_mean(neigh_p, sims)

        # Blend model prediction and neighbor consensus by fragility
        adjusted_proba = p * (1 - fi) + nm * fi

        # Decision: is it confident enough to trust?
        use_llm = not self.is_confident(adjusted_proba, fi)

        return {
            "ss": ss,
            "cp": cp,
            "fi": fi,
            "nm": nm,
            "adjusted_proba": adjusted_proba,
            "use_llm": use_llm,
        }

    def compute_batch(
        self,
        ps: np.ndarray,
        neigh_ps: np.ndarray,
        sims_arr: np.ndarray
    ) -> list[dict]:
        """
        Vectorized computation for multiple predictions.

        Args:
            ps: Array of model probabilities (shape: [N]).
            neigh_ps: Neighbor probabilities for each sample (shape: [N, M]).
            sims_arr: Neighbor similarity weights (shape: [N, M]).

        Returns:
            List of dicts with the same keys as `compute_cf_metrics`.
        """
        results = []

        for i in range(len(ps)):
            results.append(self.compute_cf_metrics(ps[i], neigh_ps[i], sims_arr[i]))

        return results
    