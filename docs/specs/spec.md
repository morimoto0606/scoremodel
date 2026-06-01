Please audit the current implementation against the paper

    Malliavin Calculus for Score-based Diffusion Models
    arXiv:2503.16917v3

The PDF is located at:

    docs/references/Malliavin-Mirafzali.pdf

Do not modify code in this pass.
First read the paper, especially the pages containing Algorithm 4, Algorithm 5, Algorithm 6, and Algorithm 7.
Then inspect the implementation files:

    src/scoremodel_ext/malliavin/sde_nonlinear.py
    src/scoremodel_ext/malliavin/experiment_mirafzali_nonlinear.py
    src/scoremodel_ext/malliavin/models.py

The goal is to determine whether the current implementation is a faithful reproduction of Algorithms 4–7, or only a Mirafzali-inspired approximation.

============================================================
TASK 1. Extract the paper algorithms
============================================================

From the PDF, extract and summarize the mathematical content of:

- Algorithm 4
- Algorithm 5
- Algorithm 6
- Algorithm 7

For each algorithm, write:

- inputs
- outputs
- main mathematical operations
- whether it constructs Malliavin weights, conditional score estimates, neural-network training data, or reverse samples

Do not rely on memory. Use the PDF.

============================================================
TASK 2. Audit the Malliavin weight H computation
============================================================

In the current code, inspect:

    src/scoremodel_ext/malliavin/sde_nonlinear.py
    function: simulate_malliavin_nl

The current implementation appears to compute

    D_s X_T = g(s) Y_T Y_s^{-1}

then

    gamma = ∫ D_s X_T (D_s X_T)^T ds

then

    U_s = (D_s X_T)^T gamma^{-1}

then

    delta(U) ≈ Σ_s U_s^T ΔW_s

and finally

    H = -delta(U)

Check this against the paper.

Answer the following:

1. Does this match the paper's definition of the Malliavin weight?
2. Is the stochastic term S implemented?
3. Is the deterministic Skorokhod correction term D implemented?
4. If the paper has A/B/C correction terms, are they implemented?
5. Does the current implementation track the second variation process Z?
6. Does the current implementation implement Algorithm 4 and Algorithm 5 exactly, partially, or not at all?

Use exact code references.

============================================================
TASK 3. Audit the conditional expectation step
============================================================

The score is mathematically represented as

    score(x) = E[H | X_T = x]

or equivalently, depending on notation,

    ∇ log p_T(x) = E[-delta(U) | X_T = x].

Check whether the current implementation estimates this conditional expectation explicitly before neural-network training.

Search for:

- kernel regression
- Nadaraya–Watson estimator
- local averaging
- binned conditional expectation
- kNN conditional expectation
- any function that maps pairs (X_T, H) to smoothed pairs (X_T, score_hat)

Determine whether the training target is:

Case A:

    (X_T, score_hat(X_T))

where score_hat is a nonparametric estimate of E[H | X_T]

or Case B:

    (X_T, H)

where the neural network directly learns E[H | X_T] by noisy regression.

============================================================
TASK 4. Audit Algorithm 6/7 neural network training
============================================================

Inspect:

    src/scoremodel_ext/malliavin/experiment_mirafzali_nonlinear.py
    src/scoremodel_ext/malliavin/models.py

Check whether the implementation of the neural network training matches Algorithm 6/7.

Please verify:

1. What architecture is used?
2. Does it match the paper's architecture?
3. What is the training target?
4. Is the loss plain MSE?
5. Are all samples used, or is there subsampling?
6. Are time and state both fed into the model?
7. Is Fourier feature embedding used? If yes, is that in the paper or an implementation choice?
8. Is reverse sampling consistent with the paper?

============================================================
TASK 5. Classify the current implementation
============================================================

Classify the current implementation as exactly one of:

A. Fully faithful reproduction of Algorithms 4–7
B. Faithful Algorithm 6/7 training, but approximate Algorithm 4/5 Malliavin weight
C. Direct noisy regression version of Mirafzali
D. Mirafzali-inspired implementation only
E. Not a reproduction

Justify the classification with evidence.

============================================================
TASK 6. Required changes for faithful reproduction
============================================================

If the current code is not a faithful reproduction, list the required changes.

For each change, provide:

- file
- function
- current behavior
- paper-required behavior
- mathematical reason
- implementation difficulty: low / medium / high

Pay special attention to:

1. whether second variation Z is needed,
2. whether deterministic Skorokhod correction terms are missing,
3. whether Algorithm 4 requires explicit nonparametric conditional expectation before NN training,
4. whether current residual correction is a new method or just a delayed implementation of Algorithm 4.

============================================================
OUTPUT FORMAT
============================================================

Return a markdown report with exactly the following sections.

# Mirafzali Reproduction Audit

## 1. Paper algorithm summary

### Algorithm 4
- Inputs:
- Outputs:
- Mathematical operations:
- Role in pipeline:

### Algorithm 5
- Inputs:
- Outputs:
- Mathematical operations:
- Role in pipeline:

### Algorithm 6
- Inputs:
- Outputs:
- Mathematical operations:
- Role in pipeline:

### Algorithm 7
- Inputs:
- Outputs:
- Mathematical operations:
- Role in pipeline:

## 2. Current code path

Describe the actual current pipeline:

    dataset
    -> forward SDE
    -> Malliavin weight H
    -> teacher dataset
    -> neural network training
    -> reverse sampling
    -> metrics

For each arrow, name the file and function.

## 3. Malliavin weight H comparison

Create a table:

| Component | Paper | Current code | Match? | Notes |
|---|---|---|---|---|
| X simulation | | | | |
| first variation Y | | | | |
| second variation Z | | | | |
| Malliavin covariance gamma | | | | |
| stochastic term S | | | | |
| deterministic correction D | | | | |
| A/B/C terms | | | | |
| final sign of H | | | | |

## 4. Conditional expectation comparison

Create a table:

| Component | Paper | Current code | Match? | Notes |
|---|---|---|---|---|
| H samples | | | | |
| kernel / NW smoothing | | | | |
| binned smoothing | | | | |
| target for NN | | | | |
| direct noisy regression | | | | |

## 5. Algorithm 6/7 training comparison

Create a table:

| Component | Paper | Current code | Match? | Notes |
|---|---|---|---|---|
| architecture | | | | |
| hidden units | | | | |
| number of layers | | | | |
| input variables | | | | |
| Fourier features | | | | |
| loss | | | | |
| optimizer | | | | |
| epochs | | | | |
| batch size | | | | |
| reverse sampler | | | | |

## 6. Final classification

State one of:

- Fully faithful reproduction of Algorithms 4–7
- Faithful Algorithm 6/7 training, but approximate Algorithm 4/5
- Direct noisy regression version of Mirafzali
- Mirafzali-inspired implementation only
- Not a reproduction

Then give a short explanation.

## 7. Required changes

Create a table:

| Priority | File | Function | Required change | Reason | Difficulty |
|---|---|---|---|---|---|

## 8. Recommended next implementation plan

Give a step-by-step plan for making the implementation faithful.

Important:
Do not modify files yet.
Only produce the audit report.