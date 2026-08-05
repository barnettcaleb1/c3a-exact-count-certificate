# Finite exact-count certificate: C_3a > 1.1873823054

Certificate and verification package for an improved lower bound on constant 3a
(the Gyarmati–Hennecart–Ruzsa sum–difference constant) of
[teorth/optimizationproblems](https://github.com/teorth/optimizationproblems):

**C_3a ≥ θ(U) > 1.187382305438637309089264177964**, improving the previous
exact-count record 1.1835129324.

## The set

U = { Σ_{i=0}^{255} a_i · 379^i : a_i ∈ M, Σ_i a_i ≤ 11923 },
M = ⟨24,26,36,39⟩ ∩ [0,189] (numerical semigroup mask, |M| = 99, base Q = 379).

By the Gyarmati–Hennecart–Ruzsa lemma, any finite U ∋ 0 with |U−U| < 2·max(U)+1 gives
C_3a ≥ 1 + log(|U−U|/|U+U|)/log(2·max(U)+1). All four quantities are bounded in the
safe direction by digit-string dynamic programming (max U ≈ 10^659; no enumeration):
|U+U| from above (exact integer DP), |U−U| from below (realizable-witness count,
directed-rounding DP) and from above (exact integer DP, verifying the admissibility
condition < 2·max(U)+1 exactly), max U exactly (greedy, with optimality proof).

## Contents

- `certificate.txt` / `certificate.json` — the certificate: construction, the four
  certified integers in full, the mathematical arguments (F1–F3), and θ evaluated
  with outward rounding.
- `verify.py` — self-contained verifier; recomputes everything from the four
  generators. Run: `python3 verify.py 256 23846` (~10 min; auto-compiles `tdp.c`
  with clang). `verify_rerun.log` is a saved successful run.
- `tdp.c` — the difference-count lower-bound DP (float64, `fesetround(FE_DOWNWARD)`
  with runtime rounding self-test, exact power-of-two rescaling, prefix corridor).
- `exact1d.py` — exact big-integer DPs for the sum-side upper bound and
  difference-side upper bound (packed-integer polynomial powering, with an
  independent mod-prime cross-check).
- `validate.py`, `validate_c.py` — brute-force validation on small analogs where U
  can be enumerated exactly (all bound directions, witness realizability, greedy
  max-U optimality), and C-DP-vs-exact-reference checks.
- `scan.py` — the (m, L) scan used to select parameters.
- `independent-verification/` — a second, independent implementation of the
  difference-count lower bound in pure exact integer arithmetic (no floating point):
  `indep_tdp.py`, validated against brute force at m = 2, 3 and run at m = 64 and
  m = 128 for cross-agreement with the primary pipeline.
- `SHA256SUMS` — checksums of all files above.

## Provenance / AI disclosure

Prepared with substantial AI assistance (Anthropic Claude), directed and reviewed by
the human contributor. The certificate is machine-checkable independent of
provenance via `verify.py`.
