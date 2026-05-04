# Spec 01: 1D NW Teacher Validation

Status: TODO  
Priority: Highest

---

## Goal

Use existing 1D nonlinear SDE experiment as clean benchmark.

Validate whether:

- raw teacher
- binned teacher
- NW teacher
- kNN-NW teacher

improve score estimation.

---

## SDE

\[
dX_t = (-x-x^3)dt + 0.8 dW_t
\]

Initial:

\[
X_0=1.5
\]

Times:

```python
[0.05,0.10,0.20,0.40,0.60,0.80,1.00]