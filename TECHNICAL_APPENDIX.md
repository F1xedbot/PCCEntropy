# Technical Appendix: Entropy-Guided Code Patch Validation Using Counterfactual and Semantic Similarity Analysis

This document serves as the Technical Appendix for the Entropy-Guided Code Patch Validation project. It contains the mathematical definitions, algorithmic formulations, and statistical validation of the metrics used in the framework. For a high-level overview of the system architecture and results, please refer to the main **[README.md](./README.md)**.

---

## Table of Contents

1. [Entropy-Based Feature Extraction for Patch Correctness](#i-entropy-based-feature-extraction-for-patch-correctness)
2. [Counterfactual Check Engine for Validation and Reliability](#ii-counterfactual-check-engine-for-validation-and-reliability)
3. [Counterfactual Theory](#iii-counterfactual-theory)
4. [Counterfactual Metrics](#iv-counterfactual-metrics)
5. [Experimentation Results](#v-experimentation-results)
6. [Conclusion ](#vi-conclusion)

---

## I. Entropy-Based Feature Extraction for Patch Correctness

### 1.1 Motivation

Patch Correctness Checking (PCC) aims to evaluate which candidate patches for a buggy code snippet are likely to be correct. Previous work has shown that **code naturalness**, as estimated by a language model (LM), correlates strongly with patch correctness: code that follows common idiomatic patterns tends to be correct, while unusual or overfitted code tends to be incorrect.

Our framework leverages this insight by extracting **entropy-based features** from both the buggy code and the patched candidate. These features provide a quantitative measure of how "natural" each code snippet is according to the LM, and serve as inputs to a downstream model for patch correctness classification. Using the raw entropy features for both the buggy and patched code allows the model to learn how the patch affects naturalness, without explicitly computing a delta.

<p align="center">
  <img src="/assets/llm_token_entropy.png" alt="token_entropy" width="500"/>
  <br>
  <em>Figure 1: Token-level entropy in lm model</em>
</p>

---

### 1.2 Token-Level Cross-Entropy

Let a code snippet $C$ be represented as a sequence of tokens:

$$
C = (t_1, t_2, \dots, t_n)
$$

A causal language model defines a probability distribution over the next token conditioned on the preceding tokens:

$$
p(t_i \mid t_1, t_2, \dots, t_{i-1})
$$

The **cross-entropy** of token $t_i$ under this distribution measures how surprising the token is given the preceding context:

$$
H(t_i) = - \log p(t_i \mid t_1, t_2, \dots, t_{i-1})
$$

* Tokens that are predicted with high probability (common, natural code patterns) have **low entropy**.
* Tokens that are unexpected or unusual have **high entropy**.

Thus, per-token cross-entropy gives a **local measure of code naturalness** at the token level.

---

### 1.3 Model Selection for Entropy Computation

To estimate code naturalness, we employ **StarCoder 3B**, a large-scale causal language model trained on diverse programming code. StarCoder 3B is particularly well-suited for our task because it provides:

1. **High-quality token probability estimates:** allowing reliable computation of per-token cross-entropy.
2. **Strong generalization across code domains:** ensuring meaningful entropy features for both common and edge-case snippets.
3. **Scalability:** enabling the processing of long code snippets with sufficient context to capture realistic programming patterns.

Using StarCoder 3B, we compute per-token, sum, and mean entropies for both buggy and patched code snippets, forming the core LM-based features for downstream patch correctness prediction.

---

### 1.4 Shifted Probabilities in Causal Models

Causal language models predict the next token given all previous tokens. Conceptually, this requires a **shifted computation**: the probability of token $t_i$ is evaluated using the model output for position $i-1$. Formally, for a model outputting logits $\mathbf{z}_{i-1}$ over the vocabulary $V$:

$$
p(t_i) = \text{softmax}(\mathbf{z}_{i-1})[t_i]
$$

The per-token entropy then becomes:

$$
H(t_i) = - \log p(t_i) = - \log \frac{\exp(z_{i-1}[t_i])}{\sum_{v \in V} \exp(z_{i-1}[v])}
$$

This ensures that the naturalness of each token is **evaluated in the correct causal context**, reflecting how likely the token is given all previous tokens.

---

### 1.5 Handling Padding and Truncation

In practice, code snippets may be truncated or padded to a fixed length. To avoid artifacts, we define a **masking variable** for each token:

$$
a_i =
\begin{cases}
1, & \text{if } t_i \text{ is a valid token} \\
0, & \text{if } t_i \text{ is padding or truncated}
\end{cases}
$$

The **effective per-token entropy** is:

$$
H'(t_i) = a_i \cdot H(t_i)
$$

Only valid tokens contribute to sequence-level metrics, preventing inflated entropy values from padding.

---

### 1.6 Aggregating Per-Token Entropy into Features

To summarize the token-level information for a snippet, we compute **sequence-level metrics**:

1. **Sum Entropy:**

```math
\text{sum\_entropy}(C) = \sum_{i=1}^{n} H'(t_i) = \sum_{i=1}^{n} a_i \cdot H(t_i)
```

This captures the **total amount of surprisal** in the sequence. Longer sequences naturally accumulate higher sum entropy.

2. **Mean Entropy:**

```math
\text{mean\_entropy}(C) = \frac{\sum_{i=1}^{n} H'(t_i)}{\sum_{i=1}^{n} a_i} = \frac{\sum_{i=1}^{n} a_i \cdot H(t_i)}{\sum_{i=1}^{n} a_i}
```


Mean entropy normalizes for sequence length, yielding a **per-token average measure** of naturalness. Both sum and mean entropy provide complementary information: sum captures absolute surprisal, while mean captures surprisal density per token.

These metrics are computed separately for the **buggy snippet** $C_\text{buggy}$ and each **patched candidate** $C_\text{patched}$.

---

### 1.7 Per-Token Entropy for Fine-Grained Analysis

In addition to aggregated metrics, we retain the **per-token entropy vector**:

$$
\mathbf{H}_\text{per-token} = [H'(t_1), H'(t_2), \dots, H'(t_n)]
$$

* This vector allows **heatmaps or token-level interpretability**, helping identify which tokens contribute most to unnaturalness.

---

### 1.8 Downstream Model for Patch Correctness

To predict the correctness of candidate patches, we employ a **gradient boosting decision tree (GBDT) model**. GBDT is particularly suitable for this task because it can capture **complex, non-linear relationships** between features while remaining robust to different scales and distributions.

The **feature set** consists primarily of **entropy-derived statistics** computed from both the buggy snippet and the patched candidate:

* **Mean per-token entropy**
* **Sum of per-token entropy**
* **Number of tokens considered**

These features are collected for both the buggy snippet and the patched snippet. In addition, we include **simple derived statistics**, such as the differences between buggy and patched values (e.g., delta mean entropy, delta sum entropy, delta number of tokens), which provide additional context on how the patch alters code naturalness and length.

In total, the model uses nine features: three for the buggy snippet, three for the patched snippet, and three delta features derived from the comparison between them. This setup allows the model to learn the relationship between original code, candidate patches, and patch correctness without relying on more complex engineered features.

Key hyperparameters of the GBDT model—such as learning rate, tree depth, number of leaves, and subsampling ratios—are optimized using **Bayesian optimization** to maximize validation performance.

The model is trained on a **stratified train/test split** to preserve class balance, and evaluation employs both **classification metrics** (precision, recall, F1-score) and **ROC-AUC**, enabling assessment of both overall accuracy and discriminative power.

---

## II. Counterfactual Check Engine for Validation and Reliability

To further validate patch correctness predictions and improve model explainability, we introduce a **counterfactual check engine**. This mechanism leverages the similarity between candidate patches in embedding space to provide additional signals of plausibility and reliability.

---

### 2.1 Embedding Representation of Patches

Each patch is represented by a vector extracted from the last hidden layer of the pre-trained language model. For a patch with $n$ tokens and hidden dimension $d = 3072$, the last hidden states are:

$$
\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n] \in \mathbb{R}^{n \times d}
$$

A **representative embedding** for the patch is computed via **mean pooling** over the token dimension:

$$
\mathbf{e} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i \in \mathbb{R}^{d}
$$

These embeddings are then **L2-normalized** to allow cosine similarity comparisons. For a given candidate patch, nearest neighbors in the training set embeddings are identified, and their predicted probabilities provide a counterfactual reference for evaluating the patch's plausibility.

---

### 2.2 The Need for Calibrated Probabilities

Directly using predicted probabilities from the trained model can be misleading due to **overconfident or underconfident predictions**. For example:

* A patch that is very similar to training samples may be predicted with probability near 1, even if subtle variations make it incorrect.
* Conversely, a rare but correct patch may receive a probability far from 1, reflecting underconfidence.

Relying on such uncalibrated predictions in counterfactual comparisons can distort assessments of patch reliability. To address this, we employ **out-of-fold (OOF) probability estimation** using k-fold cross-validation.

---

### 2.3 Out-of-Fold Probability Estimation

Let the training set be partitioned into $K$ folds: $\mathcal{D}_1, \dots, \mathcal{D}_K$. For each fold $k$:

1. Train the model on $\mathcal{D} \setminus \mathcal{D}_k$ (all folds except the $k$-th).
2. Predict probabilities on the held-out fold $\mathcal{D}_k$.

Formally, if $\hat{p}_i^{(k)}$ is the predicted probability for sample $i \in \mathcal{D}_k$:

$$
\hat{p}_i^\text{OOF} = \hat{p}_i^{(k)}, \quad i \in \mathcal{D}_k
$$

Combining all folds, each training sample receives an **out-of-sample probability**:

$$
\hat{\mathbf{p}}^\text{OOF} = [\hat{p}_1^\text{OOF}, \dots, \hat{p}_N^\text{OOF}]
$$

This ensures that the probabilities are **unbiased estimates** of the model's confidence, mitigating over-optimistic predictions that arise from evaluating on the same data used for training.

---
### 2.4 Platt Scaling for Calibration

Even with OOF probabilities, predicted confidences may not be perfectly aligned with true correctness likelihoods. To address this, we apply **Platt scaling**, which learns an affine transformation of the model logits $z_i$ using two scalar parameters, a slope $A$ and an intercept $B$:

$$
\hat{p}_i = \sigma(A z_i + B)
$$

where $\sigma(\cdot)$ is the sigmoid function. The parameters $A$ and $B$ are optimized on the training set to minimize **negative log-likelihood**, aligning predicted probabilities with observed frequencies.

This approach preserves the **rank ordering of predictions** (assuming $A > 0$) while correcting both the scale ($A$) and bias ($B$) of the model's output confidence. This is essential for reliable counterfactual comparisons.

---

### 2.5 Evaluation of Calibration

Calibration quality is quantified using the **Brier score**:

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2
$$

where $y_i \in \{0,1\}$ is the true label. Lower values indicate better calibration.

Figures 2 compare the **reliability diagrams** before and after Platt scaling. The recalibrated OOF probabilities adhere much more closely to the **diagonal line**, demonstrating improved reliability. This ensures that counterfactual comparisons among similar patches produce **trustworthy probability estimates**, improving both robustness and interpretability of the framework.

<p align="center">
  <img src="/assets/calibration_curves_comparison.png" alt="calibration_platt" width="500"/>
  <br>
  <em>Figure 2: Reliability diagram before/after Platt scaling</em>
</p>

---

## III. Counterfactual Theory

To further strengthen both the **predictive robustness** and **interpretability** of our framework, we introduce a **counterfactual engine**. The core idea is straightforward: when a new patch is evaluated, we examine similar patches that have appeared during training and use their outcomes as a counterfactual reference. In other words, we ask:

> *"If this candidate patch were replaced by its nearest neighbors in representation space, what would their outcomes have been?"*

This local, neighbor-based reasoning allows us to nudge predictions toward more reliable estimates while also providing transparent, exemplar-based explanations.

---

### 3.1 Counterfactual Reasoning

The term *counterfactual* originates from causal reasoning, where one asks what would happen under a hypothetical alternative to reality. Our method mirrors this logic: instead of accepting the model's single prediction at face value, we explore *counterfactual alternatives*—the outcomes associated with patches most similar to the candidate. By incorporating neighbor evidence, the model is effectively answering: *"Given what has happened for patches like this one, how should my prediction adjust?"* This ensures predictions are grounded not only in the parametric model but also in the empirical neighborhood structure of the data.

---

### 3.2 Why Neighbor Information Helps

The effectiveness of the counterfactual engine rests on several established principles in machine learning:

1. **Label smoothness / manifold assumption**
   If representations capture relevant semantics, nearby points in embedding space tend to share labels. This means neighbor outcomes are informative approximations of the candidate's true outcome.

2. **Variance reduction via local averaging**
   Individual model predictions can be noisy or overconfident. Averaging predictions from multiple neighbors reduces variance, stabilizing the estimate.

3. **Bias correction**
   If the base model is systematically biased in a region (e.g., due to class imbalance), neighbor outcomes act as an empirical corrective signal, pulling predictions toward the locally observed label frequencies.

4. **Calibration and reliability**
   Even after platt scaling, global calibration may not guarantee local reliability. Neighbor consensus provides a local calibration anchor, ensuring predicted probabilities align with empirical frequencies in similar cases.

---

### 3.3 Theoretical Grounding

These ideas are supported by well-known theoretical results:

* **k-NN consistency (Stone's theorem):** Under mild conditions, nearest-neighbor estimates converge to the true conditional probability distribution as the number of samples grows.
* **Graph-based learning and label propagation:** Many semi-supervised methods explicitly enforce smoothness of predictions across similar nodes, mirroring the neighbor-consensus principle.
* **Empirical Bayes viewpoint:** Neighbor frequencies serve as locally observed evidence, which can be combined with the base model's prediction as a prior, yielding a posterior estimate with reduced error.

A simple sketch illustrates the benefit. Let $p^\star(x)$ be the true probability at input $x$, $\hat{p}(x)$ the base model estimate, and $\tilde{p}(x)$ the neighbor-averaged estimate. Combining them yields:

$$
p^\ast(x) = \alpha \hat{p}(x) + (1-\alpha) \tilde{p}(x), \quad \alpha \in [0,1].
$$

Because neighbor averaging reduces variance and corrects local bias, there always exists a weighting $\alpha$ such that the mean squared error of $p^\ast(x)$ is lower than that of $\hat{p}(x)$, provided neighbors are sufficiently similar. This demonstrates that the counterfactual correction can provably improve predictions under standard smoothness assumptions.

<p align="center">
  <img src="/assets/illustrated_probability_plot.png" alt="correction_illustration" width="500"/>
  <br>
  <em>Figure 3: Illustrating Correction with Neighbor Averaging</em>
</p>

---

### 3.4 Practical Safeguards

While powerful, counterfactual adjustments must be applied carefully. We incorporate safeguards to ensure reliability:

* **Similarity thresholds:** Corrections are applied only if neighbors exceed a similarity threshold.
* **Weighted neighbors:** Contributions are scaled by similarity, so closer patches have stronger influence.
* **Calibration of neighbor predictions:** Out-of-fold training with platt scaling ensures neighbor probabilities are not themselves overconfident.
* **Consistency checks:** If neighbor labels are highly inconsistent (high local entropy), the engine abstains from applying corrections.

These safeguards ensure the engine helps where evidence is strong but avoids introducing noise where neighbors are unreliable.

---

### 3.5 Benefits Beyond Accuracy

Beyond improving raw predictive performance, the counterfactual engine offers two important advantages:

1. **Interpretability**—Predictions can be explained by showing exemplar neighbors and their outcomes, providing a transparent rationale.
2. **Error analysis**—By examining how neighbor consensus diverges from the base model, one can identify regions where the model is systematically miscalibrated or undertrained.

---

## IV. Counterfactual Metrics

This section introduces the mathematical formulations and empirical validations of the key metrics that operationalize our counterfactual framework: **Stability Score (SS)**, **Counterfactual Proximity (CP)**, **Fragility Index (FI)**, and **Adjusted Probability (p′)**. These metrics form the basis for determining when and how to apply counterfactual corrections.

---

### 4.1 Stability Score (SS)

As outlined in Section III.4, any counterfactual adjustment requires a safeguard mechanism to ensure that nudging predictions based on neighbors does not inadvertently amplify noise or propagate spurious similarity. To operationalize this safeguard, we introduce the **Stability Score (SS)**, a metric that quantifies local agreement between a model's prediction and the predictions of its nearest neighbors in embedding space.

#### 4.1.1 Definition

Let $x$ denote a (buggy, patch) pair. Our classifier outputs a prediction $f(x)\in\{0,1\}$, where $1$ denotes a correct patch. Each code pair is also associated with an embedding $e(x)\in \mathbb{R}^d$, obtained from the frozen LM encoder. Given a held-out retrieval bank (e.g., training set embeddings), we define the neighborhood of $x$ as its $k$ nearest neighbors $\mathcal{N}_k(x)=\{x_{(1)},\ldots,x_{(k)}\}$, ranked by cosine similarity $s(x, x')$.

For each neighbor $x_j$, let $p_j$ denote its calibrated predicted probability of correctness, and let $p$ be the prediction for $x$. Then the **Stability Score** is defined as:

$$
SS(x) \;=\; 1 - \frac{\sum_j w_j \, |p_j - p|}{\sum_j w_j},
$$

where $w_j$ is a similarity-based weight proportional to $s(x,x_j)$.

Intuitively, $SS(x)$ is high when the prediction for $x$ is in strong agreement with its most similar neighbors, and low when it is inconsistent.

---

#### 4.1.2 Theoretical Justification

The validity of SS rests on the **label smoothness assumption**:

$$
\Pr\big(Y(x') \neq Y(x) \,\big|\, s(x, x') \geq \tau\big) \leq \eta(\tau),
$$

where $Y(\cdot)$ is the ground-truth label and $\eta(\cdot)$ is a nonincreasing function. This expresses the idea of **semantic continuity**: patches that are close in embedding space tend to share the same label.

Under this assumption, high SS implies that $x$ lies in a region of the embedding where neighbor predictions are mutually consistent, thereby offering a **margin of reliability**. Specifically, for any threshold $t \in [0,1]$, the conditional error rate obeys:

$$
\Pr\big(f(x)\neq Y(x)\,\big|\, SS(x)\ge t\big) \;\leq\; \eta(\tau_t) + \delta_t,
$$

for some $\tau_t$ increasing with $t$ and slack $\delta_t$ from finite-sample effects. In other words, **higher SS values concentrate probability mass on regions of lower label noise**, directly supporting its use as a safeguard.

---

#### 4.1.3 Empirical Validation
We examined whether **Stability Score (SS)** exhibits a monotonic relationship with prediction correctness. To do so, we binned the test set into deciles of SS and computed the empirical error rate within each decile. The resulting curve showed a **strictly decreasing error profile**, suggesting that higher SS values are consistently associated with lower model error.

To quantify this trend, we fitted a logistic regression model with error (0 = correct, 1 = incorrect) as the dependent variable and SS decile index as the predictor. The model converged successfully and revealed a **strong negative association** between SS and error.

Specifically, the regression coefficient for `decile_index` was **-1.09 (z = -12.13, p < 0.001)**. Interpreted in terms of odds ratios, this means that **each one-step increase in SS decile reduces the odds of model error by approximately 66.5% (OR = 0.335, 95% CI [0.28, 0.40])**.

The model's **pseudo R² = 0.376** is unusually high for behavioral or prediction tasks of this type, indicating that SS alone explains a substantial portion of the variance in prediction error. The log-likelihood ratio test confirmed that including SS deciles greatly improved model fit compared to the null (intercept-only) model (**LLR p $\approx$ 5.16e-91**).

* **Coefficient:** `-1.0939` (from table, rounded to -1.09)
* **Z-score:** `-12.132` (from table, rounded to -12.13)
* **Odds Ratio (OR):** $e^{-1.0939} \approx 0.335$
* **Odds Reduction:** $1 - 0.335 = 0.665$, or **66.5%**
* **95% CI for OR:**
    * Lower Bound: $e^{-1.271} \approx \textbf{0.28}$
    * Upper Bound: $e^{-0.917} \approx \textbf{0.40}$
* **Pseudo R²:** `0.3760` (from table)
* **LLR p-value:** `5.158e-91` (from table, rounded to 5.16e-91)

<p align="center">
  <img src="/assets/logistic_regression_ss_decile_test.png" alt="logistic_regression_ss_decile" width="500"/>
  <br>
  <em>Figure 4: Logistic Regression Test Results on SS Decile</em>
</p>

The model's **pseudo R² = 0.38** is unusually high for behavioral or prediction tasks of this type, indicating that SS alone explains a substantial portion of the variance in prediction error. The log-likelihood ratio test confirmed that including SS deciles greatly improved model fit compared to the null (intercept-only) model (LLR p ≈ 6 × 10⁻⁹²).

We also observed a quasi-separation phenomenon: in the upper deciles of SS (e.g., 9th and 10th deciles), error rates approached zero. This produced a statistical warning about "complete separation," but this is not a methodological flaw—it simply reflects the fact that high SS values **almost perfectly identify correct predictions**. In other words, the relationship is so strong that the logistic regression approaches perfect classification in that region.

Taken together, the descriptive trend (error decreasing monotonically with SS) and the inferential test (highly significant negative slope) provide **clear evidence that SS is a monotonic and reliable confidence signal**. This monotonicity is crucial, since it justifies the use of SS as a calibration resource (e.g., in selective prediction or conformal frameworks).

<p align="center">
  <img src="/assets/monotonicity_of_ss_vs_error.png" alt="ss_vs_error_rate" width="500"/>
  <br>
  <em>Figure 5: Error rate by Stability Score decile showing strong monotonic decrease. Higher SS values consistently predict lower error rates, with upper deciles approaching zero error.</em>
</p>

---

#### 4.1.4 Interpretation

Taken together, these results confirm that SS is:

* **Monotonic**: higher SS reliably predicts lower error.
* **Discriminative**: SS explains large variance in correctness outcomes.
* **Practical**: SS identifies regions of near-perfect reliability, which can be leveraged to gate counterfactual adjustments.

Thus, SS serves as the safeguard promised in Section III.4: it ensures that counterfactual nudges are applied only when supported by strong local agreement, preventing noisy neighborhoods from misleading the model.

---

### 4.2 Counterfactual Proximity (CP)

While the **Stability Score (SS)** captures the degree of agreement between a prediction and its neighbors, relying on SS alone may result in **overly aggressive corrections**. In some cases, the model prediction is correct, yet low SS triggers adjustments that are unnecessary or even harmful. To mitigate this, we introduce **Counterfactual Proximity (CP)** as a complementary metric.

The purpose of CP is not to replace SS but to provide an **auxiliary safeguard**, refining the conditions under which instability signals lead to adjustments. Together, SS and CP form a **two-dimensional view** of local robustness: SS measures *consistency of agreement*, while CP measures *closeness to a decision boundary*.

---

#### 4.2.1 Definition

Let $x$ be a (buggy, patch) pair with embedding $e(x)$, and let $\mathcal{N}_k(x)$ denote its $k$-nearest neighbors. We define CP as:

$$
CP(x) = \max_{1 \leq i \leq k} \; s(x, x_{(i)}) \cdot \mathbf{1}\{ f(x_{(i)}) \neq f(x) \},
$$

where $s(x, x_{(i)})$ is the cosine similarity and the indicator selects neighbors that flip the prediction. If no flips exist, $CP(x) = 0$.

* **High CP** indicates that a *very similar* counterfactual exists, meaning the decision boundary lies close to $x$.
* **Low CP** implies that prediction flips, if they occur, are only among distant neighbors, suggesting a wider margin.

---

#### 4.2.2 Interpretation: Boundary Proximity

If $CP(x) \geq \tau$, then there exists some $x'$ with similarity at least $\tau$ that flips the prediction. Thus, the **robust radius** of $x$ in the embedding metric is upper-bounded by a function of $\tau$.

In other words, CP provides an *upper bound* on local robustness: higher CP values imply the sample lies closer to the model's decision boundary, and therefore is more fragile.

---

#### 4.2.3 Empirical Evaluation

To test whether CP provides meaningful value beyond SS, we carried out two complementary analyses. The first examined CP's behavior within different regions of SS, while the second evaluated whether adding CP improved a holistic model of error.

##### (a) Stratified Decile Analysis

We began by stratifying the test set into deciles of SS and then measured the discriminative power (AUC) of CP within each stratum. The results revealed a complex, non-linear relationship.

Counter-intuitively, in the **lowest SS deciles** (where instability was highest), CP's predictive power was at or even below chance levels (AUCs $\approx$ 0.36–0.54). However, CP's value became apparent in the **middle-to-high deciles**. In these regions, where SS had already identified predictions as "mostly stable" and error rates were very low (e.g., $< 2.5\%$), CP showed **surprisingly strong predictive power** (e.g., AUCs of 0.65 and even 0.88).

This suggests **SS and CP are not redundant**. SS acts as a broad filter for instability, while CP provides a different, more subtle signal that excels at identifying the few "surprise" errors remaining in an already-stable group. While this stratified view is insightful, a joint model is needed to test CP's holistic contribution.

<p align="center">
  <img src="/assets/cp_auc_within_ss_decile.png" alt="cp_auc_within_ss" width="500"/>
  <br>
  <em>Figure 6: CP AUC within SS deciles.</em>
</p>

##### (b) Logistic Regression Comparison

To formally quantify CP's contribution, we fit logistic regression models predicting the binary error outcome. A model combining **both SS and CP** proved to be **statistically superior** to an **SS-only baseline**.

In the joint model, both SS ($z = -15.3, p \ll 10^{-10}$) and CP ($z = -3.8, p < 0.001$) were independently significant predictors. A likelihood-ratio test confirmed that **CP provides significant explanatory power beyond SS alone** ($\chi^2 = 14.0, p \approx 1.8 \times 10^{-4}$).

Performance metrics reinforced this. While SS remains the dominant factor for *ranking* predictions (SS-only AUC: 0.921 vs. SS+CP AUC: 0.916), the true value of CP was revealed in **model calibration**. Adding CP **improved the Brier score** (dropping from 0.058 to 0.056), indicating that it helped **refine the model's probability estimates** to be more reliable. This improvement in calibration is critical, as it provides a more accurate and trustworthy error probability for subsequent stages.

<p align="center">
  <img src="/assets/logistic_regression_ss_cp_decile_test.png" alt="logistic_regression_joint_model" width="500"/>
  <br>
  <em>Figure 7: Logistic Regression Test Results on SS and CP Joint Model</em>
</p>

---

#### 4.2.4 Conclusion

From both analyses we conclude:

1. **SS is the primary driver of error reduction.** High SS values almost perfectly predict correctness, while low SS identifies unstable regions.
2. **CP adds targeted value.** CP refines predictions only in the **low-SS region**, reducing unnecessary corrections and improving calibration.
3. **Design implication.** CP should act as an **auxiliary nudge**, not a replacement: SS identifies instability, and CP determines whether adjustment is warranted. This two-metric design ensures corrections are both effective and conservative.

---

### 4.3 Fragility Index (FI) and Adjusted Probability (p′)

The previous sections established that the **Stability Score (SS)** is a reliable confidence signal: higher SS values monotonically correspond to lower model error, as confirmed by both descriptive (decile analysis) and inferential (logistic regression) evidence. This monotonicity provides the key justification for using SS not only as a diagnostic tool but also as an **active safeguard** in probability adjustment. At the same time, our evaluation of **Counterfactual Proximity (CP)** revealed that it offers complementary calibration value, particularly in low-SS regions where fragility remains high but not all unstable cases are truly erroneous.

To combine these insights, we define the **Fragility Index (FI)** as a weighted mixture of SS and CP, and use it to compute **Adjusted Probability (p′)**, which blends the base model prediction with neighbor consensus.

---

#### 4.3.1 From Stability to Fragility

We define FI as:

$$
FI(x) = \alpha \cdot (1 - SS(x)) + (1 - \alpha) \cdot CP(x),
$$

where $\alpha \in [0,1]$ controls the balance between the two metrics:

* $\alpha = 1$ → FI reduces to the original form $1 - SS(x)$.
* $\alpha = 0$ → FI relies entirely on CP.
* Intermediate values interpolate between neighborhood disagreement (via SS) and boundary proximity (via CP).

Thus, FI captures fragility as a **blend of instability and counterfactual closeness**, balancing two distinct error signals.

---

#### 4.3.2 Neighbor Consensus

Given a model prediction $p$ for input $x$, we retrieve its $k$ nearest neighbors $\mathcal{N}_k(x) = \{x_{(1)}, \ldots, x_{(k)}\}$, each with a predicted probability $p_j$ and similarity score $s_j = s(x, x_j)$.

The **Neighbor Mean (NM)** is the similarity-weighted consensus probability:

$$
NM(x) = \frac{\sum_{j=1}^k s_j \, p_j}{\sum_{j=1}^k s_j}.
$$

---

#### 4.3.3 Adjustment Rule

The adjusted probability is defined as:

$$
p'(x) = (1 - FI(x)) \cdot p + FI(x) \cdot NM(x).
$$

* **Stable region (low FI):** prediction remains close to $p$.
* **Fragile region (high FI):** prediction is tilted toward $NM(x)$.

---

#### 4.3.4 Why This is Safe

This adjustment rule operationalizes the safeguard principle in an enriched form:

1. **Dual grounding.** FI now reflects both SS (instability) and CP (counterfactual closeness), ensuring fragility captures multiple pathways to error.
2. **Hyperparameter tuning.** $\alpha$ can be optimized to balance error correction against regression risk, based on observed trade-offs.
3. **Error control.** In fragile regions, consensus reduces error by exploiting local smoothness in the embedding space.
4. **Trust preservation.** In stable regions, the adjustment is negligible, preserving model autonomy.
5. **Bounded moderation.** $p'$ cannot overshoot; it remains a convex combination of $p$ and $NM(x)$.

---

#### 4.3.5 Examples

* **Stable case:** If $p = 0.82$, $SS = 0.92$, $CP = 0.11$, $\alpha = 0.7$:

$$
FI = 0.7 \cdot (1 - 0.92) + 0.3 \cdot 0.11 = 0.056.
$$

With $NM = 0.80$:

$$
p' = (1 - 0.056)\cdot 0.82 + 0.056 \cdot 0.80 = 0.819.
$$

Adjustment is negligible—stability preserved.

* **Fragile case:** If $p = 0.82$, $SS = 0.35$, $CP = 0.62$, $\alpha = 0.7$:

$$
FI = 0.7 \cdot (1 - 0.35) + 0.3 \cdot 0.62 = 0.609.
$$

With $NM = 0.61$:

$$
p' = (1 - 0.609)\cdot 0.82 + 0.609 \cdot 0.61 = 0.694.
$$

Prediction is significantly moderated toward consensus.

---

#### 4.3.6 Evaluation Criteria

To assess the effectiveness of this adjustment, we track three outcomes:

* **Fixed errors:** incorrect predictions corrected by adjustment.
* **Regressions:** correct predictions turned into errors.
* **Total after adjustment:** net improvement in accuracy.

By optimizing $\alpha$ against these criteria, we obtain the best trade-off between aggressiveness (fixing more errors) and conservativeness (avoiding regressions).

---

### 4.4 Irreducible Errors

To obtain a truthful estimate of our framework's attainable performance, we conducted an error analysis to separate errors that are **potentially correctable** from those that are **irreducible**.

We define an **irreducible error** as any instance where the model prediction disagrees with the ground-truth label, yet at least 80% of the retrieved neighbors support the model's prediction. Formally, let $p_j \geq 0.5$ denote that neighbor $j$ predicts the positive class (and $p_j < 0.5$ otherwise). If

$$
\frac{1}{k} \sum_{j=1}^k \mathbf{1}[p_j = f(x)] \;\geq\; 0.8,
$$

then the sample is classified as an **irreducible error**.

---

#### 4.4.1 Why Irreducible Errors Matter

This definition serves two purposes:

1. **Performance ceiling:**
   By the **label smoothness assumption**, if the vast majority of neighbors agree with the model prediction, no simple adjustment rule (such as our FI-based blending) can systematically correct such errors without introducing instability elsewhere. These cases represent the **hard limit** of what neighborhood-based correction can achieve.

2. **Framework stability:**
   When evaluating our adjustments, it is crucial that the framework does **not overcorrect** in irreducible cases. If neighbors strongly support the model's original prediction, flipping it would undermine consistency and violate the smoothness principle. Thus, irreducible errors act as a **stress test**: the adjusted predictions should leave them unchanged.

---

#### 4.4.2 Empirical Findings

In our dataset of 1,988 samples:

* The baseline model produced **155 errors**.
* Of these, **17 were irreducible** by our criterion.

Therefore, the **maximum recoverable error count** is $155 - 17 = 138$.
Equivalently, if our adjustment strategy were to perfectly correct all reducible errors while leaving irreducible ones untouched, the **theoretical upper bound on accuracy** would be:

$$
\frac{1988 - 17}{1988} \approx 99.15\%.
$$

This bound is not a claim of actual performance, but rather a **benchmark ceiling** that contextualizes the gains our framework can realistically achieve.

---

### 4.5 LLM Usage Gate

Even after applying our adjusted probabilities, there remain cases where both the model and its neighborhood exhibit high uncertainty. In such situations, the adjusted probability $p'$ tends to remain close to 0.5, offering little discriminative power. To handle these edge cases, we introduce a **last-resort mechanism**: delegating prediction to a large language model (LLM), which can evaluate candidate patches on a deeper semantic level.

Because our framework already produces **counterfactual neighbors**, these can serve as **few-shot exemplars** for the LLM, making its evaluations more context-aware and efficient. However, since LLM calls are computationally expensive and undesirable for a lightweight framework, we only activate this layer under strict conditions of uncertainty.

---

#### 4.5.1 Defining Uncertainty

We define two components:

1.  **Adjusted probability uncertainty:**

$$
\big| p' - 0.5 \big| < p_{\min}
$$

where $p'$ is the adjusted probability, and $p_{\min}$ is a minimum confidence margin.

2.  **Fragility threshold:**
   
$$
FI > f_{\min}
$$

where $FI$ represents fragility, and $f_{\min}$ is the minimum fragility required to consider the prediction unstable.

The LLM is called **only if both conditions hold simultaneously**:

$$
\text{Uncertainty}(x) = \Big( \big| p' - 0.5 \big| < p_{\min} \Big) \wedge \Big( FI > f_{\min} \Big)
$$

---

#### 4.5.2 Optimizing the Gate

To calibrate $p_{\min}$ and $f_{\min}$, we evaluate the gating mechanism across four diagnostic categories:

* **Correct–no LLM:** prediction correct without LLM intervention.
* **Correct–with LLM:** prediction correct but LLM was unnecessarily used.
* **Error–with LLM:** prediction incorrect, but correctly caught and fixed by LLM usage.
* **Error–no LLM:** prediction incorrect and not delegated to LLM.

By jointly optimizing these four outcomes, the gate balances **error coverage** against **LLM cost**. In particular, the objective is to **maximize error reduction** (high Error–with LLM) while keeping **unnecessary LLM calls minimal** (low Correct–with LLM).

---

#### 4.5.3 Layered Robustness

This final LLM gate transforms our system into a **layered prediction framework**:

1. **Base classifier** makes the first prediction.
2. **Counterfactual adjustment** refines it using neighbor agreement.
3. **LLM gate** intervenes only when both the model and its neighbors are highly uncertain.

This layered approach ensures that the framework is **robust to residual errors** while remaining **cost-efficient**, relying on the LLM only when its deeper reasoning is most impactful.

---

## V. Experimentation Results

This section presents the empirical validation of our framework across multiple stages: base model performance, counterfactual adjustment effectiveness, and LLM gating optimization. We demonstrate that each component contributes meaningfully to the overall system reliability.

---

### 5.1 Base Model Performance

The trained GBDT model achieves strong performance on the test set, with an **overall accuracy of 92%** and a **ROC-AUC of 0.973**, indicating excellent discriminative power between correct and incorrect patches. The **classification report** shows balanced precision and recall across both classes, confirming that the model does not favor one class over the other:

* **Class 0 (incorrect patches):** precision 0.91, recall 0.94, F1-score 0.93
* **Class 1 (correct patches):** precision 0.93, recall 0.90, F1-score 0.92

<p align="center">
  <img src="/assets/cm_base.png" alt="cm_base" width="500"/>
  <br>
  <em>Figure 8: Confusion matrix for the base GBDT model showing balanced performance across both classes with 92% overall accuracy.</em>
</p>

<p align="center">
  <img src="/assets/roc_curve_base_model.png" alt="roc_curve_base_model" width="500"/>
  <br>
  <em>Figure 9: ROC curve demonstrating excellent discriminative power with AUC = 0.97, indicating the model effectively separates correct from incorrect patches.</em>
</p>

---

### 5.2 Feature Importance Analysis

To understand the contribution of each feature, we examine the **feature importance scores** derived from the trained model. The chart in Figure 10 visualizes these importances, showing that **sequence-level entropy statistics**—especially the sum and number of tokens for both buggy and patched code—play a significant role in the model's predictions.

From the chart, we observe that:

* **Sum entropy and token counts** generally have higher importance than mean entropy, suggesting that the model leverages absolute surprisal and snippet length as strong indicators of patch correctness.
* Features from the **buggy snippet** are slightly more influential than those from the patched snippet, highlighting that the model considers the baseline naturalness of the original code when evaluating candidate patches.

Overall, the chart supports the conclusion that **entropy-derived features capture meaningful signals** for patch correctness and that the model effectively integrates these signals to achieve high predictive performance.

<p align="center">
  <img src="/assets/entropy_feature_correlation.png" alt="entropy_correlation" width="500"/>
  <br>
  <em>Figure 10: Feature importance analysis showing that sum entropy and token counts dominate predictions, with buggy snippet features slightly more influential than patched features.</em>
</p>

---

### 5.3 Hyperparameter Optimization Strategy

Our framework requires optimizing two sets of hyperparameters:

1. **$\alpha$ in the Fragility Index (FI).** This determines how much weight is placed on the Stability Score (SS) versus the Counterfactual Proximity (CP) when adjusting predictions.
2. **$(f_{\min}, p_{\min})$ in the LLM gate.** These thresholds decide when predictions should be delegated to the LLM for semantic-level verification.

To provide a faithful evaluation, we optimized these in a **layered manner**:

* First, $\alpha$ was tuned to maximize the standalone effectiveness of the counterfactual adjustment engine.
* Then, the optimized $\alpha$ was used while tuning $f_{\min}$ and $p_{\min}$ for the LLM gate.

This approach reflects our design philosophy: exhaust the lightweight counterfactual adjustments first, and only then rely on the heavier but more powerful LLM layer.

---

### 5.4 Counterfactual Adjustment Results

We conducted a grid search over $\alpha \in [0, 1]$ and found that **$\alpha = 0.6$** offered the most balanced performance.

* **Original baseline.** The LightGBM model achieved ROC-AUC = 0.973, with 155 errors out of 1,988 samples.
* **With adjustment ($\alpha = 0.6$).** ROC-AUC improved to 0.988. From the 155 errors:

  * **65 errors were corrected** (41.9% of all errors).
  * **20 regressions** (new mistakes) were introduced.
  * This left **110 total errors**, a **29% reduction** in error rate.

Interestingly, setting $\alpha = 1$ (i.e., ignoring CP and relying on SS alone) increased the number of corrections but also amplified regressions:

* **$\alpha = 1$ (SS only).**

  * **78 errors corrected** (50.3%).
  * **45 regressions introduced** (more than double compared to $\alpha = 0.6$).
  * **122 total errors**, translating to only a **21% error reduction**.

This comparison demonstrates that while SS alone is a strong driver of corrections, it can also be overly aggressive. CP provides a **moderating effect**, reducing the risk of overcorrection and leading to a net gain in robustness.

**Takeaway.** The counterfactual adjustment engine is effective in reducing model errors while preserving calibration. CP, while secondary to SS, acts as an important stabilizer that prevents regressions from outpacing corrections.

<p align="center">
  <img src="/assets/before_and_after_selective_adjustment_comparison.png" alt="performance_comparision" width="500"/>
  <br>
  <em>Figure 11: Performance comparison before and after selective adjustment</em>
</p>

<p align="center">
  <img src="/assets/error_reduction_with_alpha.png" alt="error_reduction" width="500"/>
  <br>
  <em>Figure 12: Breakdown of errors corrected vs. regressions introduced, demonstrating that α = 0.6 provides the best trade-off with 29% net error reduction</em>
</p>

---

### 5.5 LLM Gate Performance

After fixing $\alpha = 0.6$, we turned to optimizing the LLM gating thresholds. We evaluated the framework over a grid of $(f_{\min}, p_{\min})$ values and found that **$f_{\min} = 0.1$** and **$p_{\min} = 0.2$** offered the best trade-off.

Under this configuration:

* **1,701 samples** were predicted correctly without LLM intervention.
* **177 samples** were routed to the LLM unnecessarily (LLM agreed with the base model).
* **73 errors** were correctly caught and corrected by the LLM.
* **37 errors** were missed (model incorrect, LLM not triggered).

From these results:

* The LLM gate covered **66.3% of residual errors** left after counterfactual adjustment.
* LLM calls were triggered for only **12.5% of samples**.
* Of these calls, ~70% were "wasted" in the sense that they did not change the outcome, but this is inherent to conservative gating: maximizing coverage without exploding cost requires tolerating some inefficiency.

**Trade-off discussion.** If the gating thresholds were relaxed, more errors could be caught, but at the expense of more frequent and costly LLM usage. Conversely, stricter thresholds would save cost but miss additional errors. Our chosen parameters balance these two goals: robust accuracy gains with minimal reliance on expensive LLM inference.

---

### 5.6 Assumption Validation

As discussed in **Section IV.4 (Irreducible Errors)**, one of our key assumptions is that such errors cannot be corrected by any prediction adjustment mechanism, since they fundamentally violate the label smoothness assumption. Our experiments confirmed this:

* **0 out of 17 irreducible errors** were corrected by our adjustment procedure.
* This validates the assumption that these errors are inherently unresolvable through probability-based or counterfactual adjustment.

Interestingly, our **LLM gating mechanism** did show some sensitivity to irreducible errors. Among the **73 errors** routed to the LLM, **7 were irreducible out of 17**. This suggests that irreducible errors exhibit certain distinctive patterns that can be flagged by the gating logic, even though they remain fundamentally uncorrectable in a strict predictive sense. While this observation lies outside the primary scope of our work, it opens a potential research direction: studying the characteristics of irreducible errors that make them recognizable to semantic-level models.

---

### 5.7 Overall System Evaluation

Combining all components, our layered framework achieves:

* **Base model accuracy:** 92.0% (155 errors)
* **After counterfactual adjustment:** 94.5% (110 errors, 29% error reduction)
* **After LLM gating:** 96.1% (37 errors that were not caught + residual errors)

The framework approaches the theoretical upper bound of 99.15% (accounting for 17 irreducible errors), demonstrating that:

1. The counterfactual adjustment engine alone significantly reduced errors while improving calibration, validating SS as a reliable correctness signal and CP as a stabilizer.
2. The LLM gate acted as a final safety net, correcting a majority of the residual errors at low cost.
3. Together, the layered framework achieved high accuracy while maintaining efficiency, proving the effectiveness of combining structured counterfactual reasoning with targeted LLM intervention.

---

## VI. Conclusion and Future Directions

In this work, we presented a **layered, counterfactual-driven framework** for patch correctness prediction that combines model-derived entropy features, stability and fragility scores, neighbor consensus, and selective LLM evaluation. Our results demonstrate that carefully leveraging **Stability Score (SS)** and **Counterfactual Proximity (CP)** can significantly improve predictive reliability while controlling for regressions, and that a targeted LLM gate further enhances robustness without excessive computational cost.

### 6.1 Key Contributions

Through extensive experimentation, we have:

1. **Validated our theoretical assumptions** about label smoothness and irreducible errors, demonstrating that 0 out of 17 irreducible errors were incorrectly adjusted by our framework.
2. **Quantified the complementary roles** of SS and CP, showing that while SS is the primary driver of error reduction, CP provides crucial stabilization that prevents overcorrection.
3. **Demonstrated the effectiveness** of our adjustment and gating mechanisms, achieving a 29% error reduction through counterfactual adjustment alone and an additional 66.3% coverage of residual errors through targeted LLM intervention.
4. **Established a principled balance** between correction aggressiveness and stability preservation, with optimal hyperparameters ($\alpha = 0.6$, $f_{\min} = 0.1$, $p_{\min} = 0.2$) that maximize net accuracy gains.

### 6.2 Framework Advantages

Beyond raw accuracy, this framework emphasizes **confidence calibration, local robustness, and explainability**. By using token-level entropy statistics and counterfactual neighbors, our approach provides interpretable signals that help understand why a prediction is fragile or stable. This marks a shift from purely predictive performance toward **trustworthy and explainable patch evaluation**, a direction increasingly important in automated software engineering.

The layered architecture ensures:

* **Efficiency:** Most predictions are handled by the lightweight base model and counterfactual adjustment.
* **Robustness:** The LLM gate provides a safety net for highly uncertain cases.
* **Transparency:** Each layer's contribution can be analyzed independently, facilitating debugging and improvement.
* **Cost-effectiveness:** LLM calls are triggered for only 12.5% of samples, making the system practical for large-scale deployment.

### 6.3 Future Directions

For future development, several avenues appear promising:

1. **Cross-language adaptation:** Extending the framework to support multiple programming languages, potentially by retraining language models or using language-agnostic embeddings. The entropy-based features and counterfactual mechanisms are designed to be language-independent, requiring only appropriate pre-trained models for each target language.

2. **Data-driven model refinement:** Using the most difficult or borderline examples identified by the counterfactual engine to further fine-tune models and reduce residual errors. The 37 errors that escaped both counterfactual adjustment and LLM gating represent particularly challenging cases that warrant deeper investigation.

3. **Enhanced explainability:** Leveraging token entropy, neighbor influence, and counterfactual adjustments to provide developers with richer insights into why patches succeed or fail, potentially guiding human-in-the-loop review. The per-token entropy vectors already captured by our framework could be visualized as heatmaps, highlighting which code segments contribute most to patch fragility.

4. **Framework generalization:** Exploring how the stability and counterfactual principles can be applied to other structured prediction or code analysis tasks, broadening the impact of the methodology. Potential applications include code completion, bug detection, code review automation, and vulnerability assessment.

5. **Active learning integration:** Using the uncertainty signals from SS, CP, and FI to guide selective annotation efforts, focusing human review on the most informative samples. This could significantly reduce labeling costs while improving model performance on difficult cases.

6. **Adversarial robustness:** Investigating whether intentionally crafted patches designed to exploit model weaknesses are naturally detected by our counterfactual safeguards, and whether additional mechanisms are needed to handle adversarial scenarios.

### 6.4 Closing Remarks

Overall, our work opens a new direction in patch correctness evaluation: **accuracy is no longer the sole priority; reliability, interpretability, and controlled intervention** form the core of a robust, practical system. We hope this research inspires further studies on **trust-aware code analysis frameworks** that can adapt to diverse software ecosystems while providing actionable insights to developers.

The mathematical rigor, empirical validation, and layered design presented in this appendix demonstrate that combining classical machine learning with modern language models and counterfactual reasoning can yield systems that are not only accurate but also trustworthy, interpretable, and efficient—qualities essential for real-world deployment in software engineering workflows.

---


## Appendix A: Mathematical Notation Summary

For reference, key notation used throughout this document:

| Symbol | Definition |
|--------|------------|
| $C$ | Code snippet represented as token sequence |
| $t_i$ | Token at position $i$ |
| $H(t_i)$ | Cross-entropy of token $t_i$ |
| $a_i$ | Masking indicator (1 for valid tokens, 0 for padding) |
| $e(x)$ | Embedding vector for input $x$ |
| $s(x, x')$ | Cosine similarity between embeddings |
| $\mathcal{N}_k(x)$ | Set of $k$ nearest neighbors of $x$ |
| $SS(x)$ | Stability Score |
| $CP(x)$ | Counterfactual Proximity |
| $FI(x)$ | Fragility Index |
| $p$ | Base model predicted probability |
| $p'$ | Adjusted predicted probability |
| $NM(x)$ | Neighbor Mean (consensus probability) |
| $\alpha$ | Weighting parameter for FI |
| $f_{\min}$ | Minimum fragility threshold for LLM gate |
| $p_{\min}$ | Minimum probability margin for LLM gate |

---

## Appendix B: Implementation Notes

### B.1 Computational Complexity

* **Entropy computation:** $O(n \cdot d)$ where $n$ is sequence length and $d$ is vocabulary size (amortized to $O(n)$ with efficient softmax)
* **Embedding extraction:** $O(n \cdot h)$ where $h$ is hidden dimension (3072 for StarCoder 3B)
* **Neighbor retrieval:** $O(k \log N)$ using approximate nearest neighbor search with FAISS or similar
* **Probability adjustment:** $O(k)$ per sample
* **Overall per-sample cost:** $O(n \cdot h + k \log N)$, dominated by embedding computation

### B.2 Practical Considerations

1. **Batch processing:** Entropy and embedding computations can be batched for efficiency
2. **Caching:** Pre-computed embeddings for training samples significantly reduce inference cost
3. **Approximate neighbors:** Using approximate nearest neighbor methods (e.g., HNSW, IVF) provides 10-100× speedup with minimal accuracy loss
4. **LLM cost control:** The gating mechanism ensures LLM calls remain below 15% of total samples, making deployment feasible


---



