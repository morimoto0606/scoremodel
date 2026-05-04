# Spec 03: Reproduce Mirafzali Section 4

Status: TODO  
Priority: High

---

## Goal

Faithfully reproduce numerical results of:

Malliavin Calculus for Score-based Diffusion Models

Section 4.

---

## Datasets

- Checkerboard
- Swiss Roll
- 8-GMM

Train size:

8000

---

## Linear SDE

Implement:

- VE
- VP
- sub-VP

Settings:

```python
T = 1.0
dt = 0.004
n_steps = 250