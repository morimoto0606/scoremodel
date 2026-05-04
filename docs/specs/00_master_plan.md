# Master Plan: Malliavin Teacher Variance Reduction Project

Status: DOING  
Priority: Highest

---

## Core Research Question

Malliavin / Skorokhod pathwise teacher labels are theoretically unbiased but noisy.

Can we improve score-based diffusion training by estimating:

\[
s_t(x)=\mathbb E[Y_t \mid X_t=x]
\]

more efficiently using local smoothing methods such as:

- Binned averaging
- Nadaraya-Watson kernel regression
- kNN-truncated kernel smoothing

where

\[
Y_t=-\delta_t(u_t)
\]

---

## Phase Roadmap

### Phase 1

1D theoretical validation.

Goal:

- Compare raw / binned / NW teachers
- Compare against numerical true score
- Reverse sampling quality

Spec:

`01_1d_nw_teacher_validation.md`

---

### Phase 2

2D generative validation.

Goal:

- 8-GMM dataset
- reverse generation quality
- mode recovery
- MMD / Wasserstein

Spec:

`02_2d_nw_teacher_generation.md`

---

### Phase 3

Reproduce Mirafzali paper Section 4.

Goal:

- benchmark parity
- fair comparison baseline

Spec:

`03_mirafzali_reproduction.md`

---

### Phase 4

Research paper direction.

Potential title:

Variance-Reduced Malliavin Teachers for Score-based Diffusion Models

---

## Success Criteria

1. NW teacher outperforms raw teacher.
2. Results stable across seeds.
3. Competitive against Mirafzali baseline.
4. Extendable to latent high-dimensional models.