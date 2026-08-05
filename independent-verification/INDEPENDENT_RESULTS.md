# Independent replication results (exact integer arithmetic, no floating point)

`indep_tdp.py` is a from-scratch reimplementation of the difference-count lower bound
T(2L) = #{ (d_i) ∈ (M−M)^m : prefix sums within ±W, Σκ(d_i) + |Σd_i| ≤ 2L },
using numpy int64 with certified floor-halving rescaling: every stored state satisfies
state ≤ true_count / 2^shift exactly, so the output is a rigorous lower bound with no
dependence on floating-point rounding modes. Written without reference to `tdp.c`.

## Validation (exact agreement with brute-force enumeration)

13/13 cases match exactly (shift = 0, no halving active):
- ⟨3,5⟩∩[0,7] analog, base 15: m = 2,3 × L ∈ {6,9,12} — 6 cases
- real mask ⟨24,26,36,39⟩∩[0,189]: m = 2 × L ∈ {120,150,189} incl. **binding corridors**
  W ∈ {56, 113} — 7 cases

## Cross-comparison with the primary pipeline (tdp.c, float64 FE_DOWNWARD)

Both are certified lower bounds of the same count; the integer-halving version loses
more tail mass, so it brackets the primary from below:

| m | 2L | W | ln T (this code) | ln T (tdp.c) | note |
|---|----|---|------------------|--------------|------|
| 4 | 378 | 400 | exact match | exact match | both = brute-force-consistent exact value 215512593 |
| 8 | 755 | 452 | exact match | exact match | |
| 64 | 6040 | 2539 | 353.16045237 | 353.16047971 | mine −2.7e−5 (tail loss) |
| 128 | 12002 | 3591 | 708.73932377 | 709.69893323 | mine looser; still certifies θ ≥ 1.18455 at m=128 alone |
| 256 | 23846 | 5078 | 1384.13628508 | 1421.85363958 | mine much looser at this depth (floor-halving tail loss compounds: certifies only θ ≥ 1.1626 alone); included for completeness — the no-float record confirmation comes from the m=128 row |

The tdp.c side was separately validated against CRT-exact ground truth at m = 8 and
m = 16 (certified-lower at every threshold, both rounding and rescale regimes active)
during the adversarial review pass; those scripts are archived with the review.

## Files

- `indep_tdp.py` — implementation + validation harness (`validate`) + full runs (`full m 2L W`)
- `indep_T_m64.json`, `indep_T_m128.json`, `indep_T_m256.json`, logs — run outputs
