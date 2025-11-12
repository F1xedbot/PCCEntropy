# Entropy-Guided Framework for Code Patch Validation

This repository presents a case study exploring a multi-layered approach to evaluating the correctness and reliability ofx` code patches.
The goal is to analyze how different validation layers—ranging from statistical models to large language models (LLMs), can be combined to balance accuracy, certainty, and cost efficiency.

---

## Problem Overview
Modern automated program repair systems often generate plausible code patches that compile successfully but remain functionally incorrect.
This leads to a bottleneck where developers must manually review numerous low-quality suggestions.

---

## Framework Overview
The study investigates a three-layer reliability framework that combines lightweight statistical modeling with LLM-based semantic validation.
Each layer contributes a different balance between speed, interpretability, and computational cost.

### **Layer 1: Fast Triage (GBDT Classifier)**

* Gradient Boosting model trained on *code naturalness* features.
* Features come from token-level cross-entropy between buggy and patched code using **StarCoder 3B**.
* Acts as a quick filter for likely-correct patches.

### **Layer 2: Counterfactual Reliability Analysis**

* Measures how *stable* Layer 1’s predictions are.
* Uses two metrics:

  * **Stability Score (SS):** Agreement with nearby patches.
  * **Counterfactual Proximity (CP):** Distance to where the prediction would flip.
* Applies **out-of-fold calibration** and **platt scaling** for reliable probabilities.
* Adjusts uncertain predictions using neighborhood consensus.

### **Layer 3: LLM-Guided Validation Gate**

* Invoked only for low-confidence or unstable patches.
* Performs semantic checks with a **large language model**.
* Boosts reliability while keeping LLM usage minimal.

---

## Results Summary

Preliminary experiments suggest that combining structured reliability metrics with targeted LLM review improves both precision and confidence calibration.

With 1988 test samples:

| Layer | Description                | ROC-AUC   | Notes                                                                      |
| ----- | -------------------------- | --------- | -------------------------------------------------------------------------- |
| 1     | Baseline GBDT              | **0.973** | 92% accuracy on held-out data                                              |
| 2     | Counterfactual Reliability | **0.988** | 29% relative error reduction, better calibration                           |
| 3     | LLM-Gate Review            | —         | Recovered ~65% of remaining false positives/negatives, used on only 12.4% of samples |

Overall, the layered setup provides a more cost-efficient alternative to full LLM-based evaluation, while still maintaining strong accuracy and reliability.

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
