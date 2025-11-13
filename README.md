# Entropy-Guided Framework for Code Patch Validation
Modern automated program repair systems often generate plausible code patches that compile successfully but remain functionally incorrect.
This leads to a bottleneck where developers must manually review numerous low-quality suggestions.

This repository explores a case study on a lightweight, multi-layered approach for code patch validation.
The framework combines simple entropy analysis from a language model with statistical methods to predict patch correctness efficiently.
On a held-out test set of 1,988 samples, the framework consistently achieves 98.8% ROC-AUC, with potential peaks up to 99.8% when the final LLM layer performs optimally.
The LLM component introduces some variability, but overall results remain stable and competitive compared to more complex empirical methods.

---

## Framework Overview

This project implements a three-layer framework that combines lightweight statistical models with LLM-based validation.
Each layer serves a specific purpose in balancing accuracy, stability, and computational cost.

### **Layer 1: Fast Triage (GBDT Classifier)**

A Gradient Boosting model trained on code naturalness features derived from token-level cross-entropy between buggy and patched code using **StarCoder 3B**.
This layer provides a fast first-pass filter to identify likely correct patches.

### **Layer 2: Counterfactual Reliability Analysis**

Evaluates the stability of Layer 1 predictions using two metrics:

* **Stability Score (SS):** Measures agreement with nearby patches.
* **Counterfactual Proximity (CP):** Measures distance to the prediction boundary.
  Uses out-of-fold calibration and Platt scaling for consistent probability estimates and adjusts uncertain predictions based on neighborhood consensus.

### **Layer 3: LLM-Guided Validation Gate**

Triggered only for low-confidence or unstable samples.
Uses a large language model to perform a final semantic check on the patch, improving reliability while keeping LLM calls limited.

---

## Results Summary

Our evaluation shows that integrating structured reliability metrics with a targeted LLM review layer significantly boosts precision and calibration — without depending fully on LLM-based evaluation.

With **1,988 test samples**, the system achieved a **fixed ROC-AUC of 0.988 (98.8%)**, verified on held-out data.
While the LLM layer can occasionally push the ROC-AUC up to **99.8%**, this improvement is not guaranteed due to the model’s non-deterministic behavior.
Nonetheless, the pipeline remains stably accurate at 98.8%.

| Layer | Description                | ROC-AUC   | Notes                                                               |
| ----- | -------------------------- | --------- | ------------------------------------------------------------------- |
| 1     | Baseline GBDT              | **0.973** | 92% accuracy on held-out data                                       |
| 2     | Counterfactual Reliability | **0.988** | 29% relative error reduction, better calibration                    |
| 3     | LLM-Gate Review            | —         | Recovered ~65% of remaining FP/FN, applied only to 12.4% of total samples |

<p align="center">
  <img src="/assets/cm_base.png" alt="Baseline Confusion Matrix" width="45%"/>
  <img src="/assets/cm_corrective.png" alt="Corrective Confusion Matrix" width="45%"/>
</p>

**Figure:** Comparison of baseline (left) and corrective (right) confusion matrices.
---

## Repository Structure

```
PCCEntropy/
│
├── assets/                # Images and diagrams used in the README
├── data/                  # Sample data
├── notebooks/             # Jupyter notebooks for analysis and experiments
├── src/                   # Source code for feature extraction and validation pipeline
│
└── TECHNICAL_APPENDIX.md  # Detailed definitions and mathematical derivations
```


