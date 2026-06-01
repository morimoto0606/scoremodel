Please first fix docs/specs/audit_report.md.

Important corrections:
1. The stochastic term should be written as

    S = γ^{-1}_{X_t} Y_t ∫_0^t (Y_u^{-1} σ(u))^T dB_u

   Do not include [A−B] in the stochastic term.

2. The deterministic correction should be written as

    D = ∫_0^t (Y_u^{-1} σ(u))^T A − B − C du

   Do not put an outer Y_t or Y_T in front of D.

3. The Skorokhod integral is

    δ_t(u_t) = S − D

   and the score label is

    H = −δ_t(u_t).

Please update every inconsistent occurrence in the audit report, especially:
- Section 3 Malliavin weight comparison table
- Required changes table
- Recommended implementation plan Step 3

After fixing the audit report, implement the full Mirafzali Algorithm 4/5 in a new separate function, without overwriting the current approximate implementation.

Required structure:
- keep current approximate implementation as simulate_malliavin_nl_approx
- add new full implementation as simulate_malliavin_nl_mirafzali_full
- add dispatcher option correction="approx" and correction="mirafzali_full"

The full implementation must include:
1. second variation process Z
2. W, Ω, Θ
3. I1, I2
4. A, B, C correction terms
5. stochastic term S
6. deterministic correction D
7. δ = S − D
8. H = −δ

Do not change the residual correction code yet. First make the full Mirafzali H computation available and add tests verifying shape, finite values, no NaNs, and that both approx and mirafzali_full modes run.