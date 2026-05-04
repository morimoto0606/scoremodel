# Spec 02: 2D NW Teacher Generation

Status: TODO  
Priority: High

---

## Goal

Use existing successful 2D 8-GMM reverse sampling pipeline.

Replace binned teacher with NW teacher and compare.

---

## Dataset

8 Gaussian mixture in 2D.

- radius = 2
- 8 modes
- isotropic Gaussian noise

---

## Forward Process

Use existing time-dependent nonlinear SDE.

Use current working implementation.

---

## Compare Methods

- raw teacher
- binned teacher
- NW teacher
- kNN-NW teacher

---

## Reverse Sampling

Existing reverse sampler.

Use identical seeds and schedules.

---

## Metrics

- Mode coverage
- Nearest mode distance
- MMD
- Wasserstein
- Reverse stability
- NaN rate

---

## Outputs

```text
results/2d_teacher_compare/
raw/
binned/
nw/
knn_nw/