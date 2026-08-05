"""Render the final human-readable text certificate from certificate_m{m}.json."""
import json
import sys
import textwrap

m = int(sys.argv[1])
cert = json.load(open(f"certificate_m{m}.json"))
cc = cert["certified_counts"]
con = cert["construction"]
dp = cert["dp_parameters"]

Tnum = cc["T_lower_le_diffset_numerator"]
Tden = cc["T_lower_le_diffset_denominator_pow2"]
Tstr = Tnum if Tden == 0 else f"{Tnum} / 2^{Tden}"

txt = f"""\
================================================================================
 COMPUTATIONAL CERTIFICATE — constant 3a (exact-count record)
 theta(U) = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
================================================================================

RESULT
  theta(U) > {cert['theta_certified_lower_bound']}
           > {cert['target_to_exceed']}   (previous record target)
  margin over target: {cert['margin_over_target']}

CONSTRUCTION
  Mask:  M = numerical semigroup <{','.join(map(str, con['mask_semigroup_generators']))}>
             intersected with [0, {con['mask_interval'][1]}]     (|M| = {con['mask_size']}, 0 in M)
  Base:  Q = 2*{con['mask_interval'][1]} + 1 = {con['base_Q']}
  Set:   U = {{ sum_(i=0)^(m-1) a_i Q^i : a_i in M, sum_i a_i <= L }}
         with m = {con['m_digits']} digits and budget L = {con['budget_L']}.
  U is finite, 0 in U (all-zero string), and max(U) has ~{cert['log10_maxU_approx']:.1f}
  decimal digits, so U is represented and counted purely via digit-string DPs.

CERTIFIED QUANTITIES (all directions safe)
  N_upper  = exact #{{(s_i) in (M+M)^m : sum s_i <= 2L}}          >= |U+U|
  T_lower  = certified-lower count of a realizable subset of
             {{(d_i) in (M-M)^m : sum kappa(d_i) + |sum d_i| <= 2L}} <= |U-U|
  D_upper  = exact #{{(d_i) in (M-M)^m : sum kappa(d_i) <= 2L}}   >= |U-U|
  maxU     = exact (greedy, optimality proved in F3 below)
  q        = 2*maxU + 1                                            exact

  N_upper = {cc['N_upper_ge_sumset']}

  T_lower = {Tstr}

  D_upper = {cc['D_upper_ge_diffset']}

  maxU    = {cc['maxU_exact']}

VERIFIED INEQUALITIES
  (i)   T_lower <= |U-U|          [F2: every counted string realizable, distinct]
  (ii)  |U+U|  <= N_upper         [F1: carry-free digit strings, injective]
  (iii) |U-U|  <= D_upper < q     [F2 necessity + exact integer comparison]
        => |U-U| < 2*max(U) + 1   (admissibility condition of constant 3a)
  (iv)  theta(U) >= 1 + (ln T_lower - ln N_upper)/ln q
                 >  {cert['target_to_exceed']}
        evaluated with directed rounding (Decimal, prec 60: ln floor/ceil in the
        safe direction per term, floor subtraction/division; 2-ulp margins).

MATHEMATICAL FACTS
  F1 (sums). Digits of u in U satisfy 0 <= a_i <= B=189 <= Q-1: base-Q strings
     are unique. For u,v in U, a_i + b_i <= 2B = Q-1: addition is carry-free,
     so u+v determines the string (s_i)=(a_i+b_i) in (M+M)^m uniquely, and
     sum_i s_i = (sum a_i)+(sum b_i) <= 2L. Distinct sums give distinct
     strings, hence |U+U| <= N_upper.
  F2 (differences). With |d_i| <= B and Q = 2B+1, balanced base-Q strings are
     unique, so u-v <-> (d_i)=(a_i-b_i) in (M-M)^m bijectively (on U-U).
     kappa(d) := min{{ a+b : a,b in M, a-b=d }}.
     Necessity: any u,v realizing (d_i) satisfy
       sum kappa(d_i) <= sum (a_i+b_i) <= 2L, hence |U-U| <= D_upper.
     Sufficiency: take the minimal witness alpha_i=(kappa(d_i)+d_i)/2,
     beta_i=(kappa(d_i)-d_i)/2 (both in M by definition of kappa). Then
       sum alpha_i = (sum kappa + sum d)/2 <= (sum kappa + |sum d|)/2 <= L
       sum beta_i  = (sum kappa - sum d)/2 <= (sum kappa + |sum d|)/2 <= L,
     so u = sum alpha_i Q^i and v = sum beta_i Q^i are in U and u-v has
     string (d_i). Distinct strings are distinct differences: T_lower <= |U-U|.
  F3 (maxU greedy). To maximize sum a_i Q^i under sum a_i <= L: suppose an
     optimal string first differs from the greedy string at position i (from
     the top), where greedy's digit is larger by delta >= 1. Positions below i
     contribute at most sum_{j<i} B*Q^j = B*(Q^i-1)/(Q-1) = (Q^i-1)/2 < Q^i
     in total (using B = (Q-1)/2), no matter how the remaining budget is
     spent. So greedy's value exceeds the alternative's by more than
     delta*Q^i - Q^i >= 0 when delta >= 1... precisely: by at least
     Q^i - (Q^i-1)/2 > 0. Hence top-down greedy (largest a in M affordable
     within remaining budget) is optimal; 0 in M keeps every suffix feasible.

DP / ROUNDING GUARANTEES
  * N_upper, D_upper: exact integer arithmetic (Python big ints). The counts
    are prefix sums of coefficients of P(x)^m truncated at degree 2L, computed
    by packed-integer polynomial powering; slot width > m*log2(P(1)) + 32 bits
    excludes slot overflow. Independently re-verified modulo two 31-bit primes
    by a direct per-position convolution DP (different algorithm & code path).
  * T_lower: C program tdp.c (float64). ALL float additions execute under
    fesetround(FE_DOWNWARD) (runtime self-test asserts scalar + NEON adds
    round down), so every stored value is <= the exact count scaled by 2^-shift.
    Rescaling multiplies by exact powers of two. Values lost to underflow and
    strings dropped by the prefix corridor |d_1+...+d_j| <= W = {dp['corridor_W']}
    only decrease the bound. The binary output (doubles = exact dyadic
    rationals) is summed exactly as Python Fractions.
  * theta: T_lower/N_upper > 1 checked exactly; ln evaluated with directed
    rounding (floor for T_lower, ceil for N_upper and q); since the numerator
    is positive, dividing its floor by an upper bound on ln q gives a valid
    lower bound on theta - 1.

VALIDATION AGAINST BRUTE FORCE (validate.py, validate_c.py)
  * Small analog mask <3,5> cap [0,7] (B=7, Q=15), m=2,3, many budgets:
    enumeration of U, U+U, U-U confirms N_upper >= |U+U|, T <= |U-U| <= D_upper,
    greedy maxU == enumerated max == reference DP; per-string witness
    realizability checked exhaustively.
  * Real mask, m=2 (L=120,189,260) and m=3 (L=130): same checks by enumeration.
  * Packed-integer polynomial powering == direct convolution reference on all
    small cases; mod-p re-verification at full scale.
  * C DP == exact int64 reference DP at m=6 (real mask), corridors W=1134 and
    W=260: float64 exact, float32 within 6e-6 below (safe direction).

PARAMETERS / PROVENANCE
  m = {con['m_digits']}, L = {con['budget_L']} (2L = {2*con['budget_L']}), Kmax = {dp['Kmax_budget_dim']}, corridor W = {dp['corridor_W']}
  limit value of this mask family: 1.1893936243 (lambda* = 0.0201794,
  mu = E[kappa] = 92.536; reproduced by mask_common.py as a sanity check)
  q = 2*maxU+1 = {cc['q_eq_2maxU_plus_1'][:60]}...({len(cc['q_eq_2maxU_plus_1'])} digits)

FILES
  mask_common.py  mask, alphabets, kappa, limit-value check
  exact1d.py      exact packed-int polynomial powering; greedy maxU
  tdp.c           certified float64 DP (FE_DOWNWARD), tdp64/tdp32 binaries
  validate.py     brute-force validation (small analog + real mask small m)
  validate_c.py   C DP vs exact reference validation
  scan.py         (m, L) scan driver
  verify.py       final self-contained verifier -> certificate_m{m}.json
  certificate_m{m}.json   machine-readable certificate (this file's source)

REPRODUCTION
  python3 mask_common.py                 # limit sanity check (1.1893936243)
  python3 validate.py && python3 validate_c.py
  clang -O2 -frounding-math -DREALT=double -o tdp64 tdp.c -lpthread -lm
  python3 verify.py {con['m_digits']} {2*con['budget_L']}      # ~{max(1, round(cert['runtime_verify_sec']/60))} min + DP time
================================================================================
"""
open(f"certificate_m{m}.txt", "w").write(txt)
print(txt[:2000])
print(f"... written to certificate_m{m}.txt ({len(txt)} chars)")
