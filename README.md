# Entropy-Guided Framework for Code Patch Validation
Modern automated program repair systems often generate plausible code patches that compile successfully but remain functionally incorrect.
This leads to a bottleneck where developers must manually review numerous low-quality suggestions.

This repository explores a case study on a lightweight, multi-layered approach for code patch validation.
The framework combines simple entropy analysis from a language model with statistical methods to predict patch correctness efficiently.
On a held-out test set of 1,988 samples, the framework consistently achieves **98.8%** ROC-AUC, with potential peaks up to **99.8%** when the final LLM layer performs optimally.
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

Evaluation shows that integrating structured reliability metrics with a targeted LLM review layer significantly boosts precision and calibration — without depending fully on LLM-based evaluation.

With **1,988 test samples**, the system achieved a **fixed ROC-AUC of 0.988 (98.8%)**, verified on held-out data.
While the LLM layer can occasionally push the ROC-AUC up to **99.8%**, this improvement is not guaranteed due to the model’s non-deterministic behavior.
Nonetheless, the pipeline remains stably accurate at 98.8%.

| Layer | Description                | ROC-AUC   | Notes                                                               |
| ----- | -------------------------- | --------- | ------------------------------------------------------------------- |
| 1     | Baseline GBDT              | **0.973** | 92% accuracy on held-out data                                       |
| 2     | Counterfactual Reliability | **0.988** | 29% relative error reduction, better calibration                    |
| 3     | LLM-Gate Review            | **~0.99** | Recovered ~65% of remaining FP/FN, applied only to 12.4% of samples |

<div align="center" style="margin-top: 30px; margin-bottom: 10px;">
  <img src="/assets/cm_base.png" alt="Baseline Confusion Matrix" width="44%" style="margin-right: 3%;" />
  <img src="/assets/cm_corrective.png" alt="Corrective Confusion Matrix" width="44%" />
</div>

</div> <p align="center" style="font-size: 0.85em; color: #555; margin-top: 5px;"> <b>Figure:</b> Baseline (left) vs. corrective (right) confusion matrices. </p>

---

## Repository Structure

```
PCCEntropy/
│
├── assets/                # Plots
├── data/                  # Sample data
├── notebooks/             # Jupyter notebooks for analysis and experiments
├── src/                   # Source code for the framework
├── USAGE.md               # How to setup and use this repo
└── TECHNICAL_APPENDIX.md  # Detailed definitions and mathematical derivations
```

---

## Further Resources

For a detailed breakdown of the framework — from conceptual design to full mathematical definitions and proofs — see **[TECHNICAL_APPENDIX.md](./TECHNICAL_APPENDIX.md)**.

To review the original analysis, and chart generation steps, explore the Jupyter notebooks in the **[`/notebooks/`](./notebooks/)** folder.

If you’d like to run this project yourself, check out **[USAGE.md](./USAGE.md)** for setup instructions and local execution details.

Credit is appreciated but not required — contributions and improvements are always welcome.
Have fun experimenting!

---

## References
The ideas and methods in this project build on existing work in automated program repair, language model analysis, and reliability-aware machine learning. Key references include:

1. Smith, E.K., Barr, E.T., Le Goues, C., Brun, Y. (2015). *Is the cure worse than the disease? Overfitting in automated program repair.* ESEC/FSE 2015. [DOI](https://doi.org/10.1145/2786805.2786825)
2. Wang, S., et al. (2021). *Automated patch correctness assessment: How far are we?* ASE ’20. [DOI](https://doi.org/10.1145/3324884.3416590)
3. Zhang, Q., et al. (2024). *APPT: Boosting Automated Patch Correctness Prediction via Fine-tuning Pre-trained Models.* [arXiv](https://arxiv.org/abs/2301.12453)
4. Tian, H., et al. (2022). *Predicting Patch Correctness Based on the Similarity of Failing Test Cases.* ACM Trans. Softw. Eng. Methodol., 31(4). [DOI](https://doi.org/10.1145/3511096)
5. Wilks, S.S. (1938). *The large-sample distribution of the likelihood ratio for testing composite hypotheses.* Ann. Math. Stat., 9(1), 60–62.
6. Cover, T., Hart, P. (1967). *Nearest neighbor pattern classification.* IEEE Trans. Inf. Theory, 13(1), 21–27.
7. Stone, C.J. (1977). *Consistent nonparametric regression.* Ann. Stat., 595–620.

---
