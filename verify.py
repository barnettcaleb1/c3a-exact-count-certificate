"""verify.py — self-contained certificate builder & verifier for constant 3a.

Construction (budgeted-mask, Zheng/Kleinwaks style):
  M = numerical semigroup <24,26,36,39> intersect [0,189]   (99 elements)
  B = 189, Q = 2B+1 = 379
  U = U(m,L) = { sum_{i<m} a_i Q^i : a_i in M, sum_i a_i <= L }

Certified quantities (directions):
  N_upper  >= |U+U|   exact integer count of {(s_i) in (M+M)^m : sum s_i <= 2L}
  T_lower  <= |U-U|   float64 DP with FE_DOWNWARD (C program tdp64) counting a
                      subset of {(d_i) in (M-M)^m : sum kappa(d_i)+|sum d_i| <= 2L}
                      (corridor |prefix d-sums| <= W further restricts: safe)
  Dupper   >= |U-U|   exact integer count of {(d_i): sum kappa(d_i) <= 2L}
  maxU     exact      greedy (optimality lemma below)
  q = 2*maxU+1 exact

Mathematical facts used (proved in accompanying certificate text; validated by
brute force on small cases in validate.py):
 F1 digits of any u in U are unique (0 <= a_i <= B < Q); sums u+v are carry-free
    since a_i+b_i <= 2B = Q-1, so U+U -> (M+M)-strings with sum <= 2L injectively.
 F2 differences have unique balanced representation (|d_i| <= B, Q = 2B+1);
    u-v -> (M-M)-string; any realization satisfies sum kappa(d_i) <= 2L (Dupper);
    conversely each string with sum kappa + |sum d| <= 2L is realized by the
    minimal witness alpha_i=(kappa+d)/2, beta_i=(kappa-d)/2 in M (T_lower).
 F3 greedy maximizes sum a_i Q^i under sum a_i <= L: raising a digit at
    position i by delta gains delta*Q^i > any value delta budget can add at
    positions < i (at most delta*Q^(i-1)*Q/(Q-1)); 0 in M makes greedy feasible.

Final inequality, evaluated with decimal directed rounding:
  theta(U) >= 1 + (ln T_lower - ln N_upper)/ln q > 1.1835129324
"""
import json
import math
import struct
import subprocess
import sys
import time
from decimal import Decimal, getcontext, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction

import numpy as np

from mask_common import semigroup_mask, build_alphabets, limit_value
from exact1d import poly_pow_prefix, max_U

TARGET = Decimal("1.1835129324")

GENS = (24, 26, 36, 39)
B = 189
Q = 2 * B + 1


# ---------------------------------------------------------------- mod-p check
def poly_pow_prefix_modp(coeffs, m, Tmax, threshold, p):
    """Independent check of the packed-int poly power: same prefix count mod p,
    computed by direct numpy int64 convolution DP (binary exponentiation not
    used; plain per-position convolution, entirely different code path)."""
    cur = np.zeros(Tmax + 1, dtype=np.int64)
    cur[0] = 1
    for _ in range(m):
        nxt = np.zeros(Tmax + 1, dtype=np.int64)
        for deg, c in coeffs.items():
            if deg <= Tmax:
                nxt[deg:] = (nxt[deg:] + cur[:Tmax + 1 - deg] * c) % p
        cur = nxt
    return int(cur[:threshold + 1].sum() % p)


# ------------------------------------------------------------- ln with bounds
def ln_dir(x, direction, prec=60):
    """Certified directed ln of a positive int or Fraction via Decimal.
    direction 'floor' -> result <= ln(x);  'ceil' -> result >= ln(x).
    Note: libmpdec ln() rounds half-even at context precision regardless of
    the context rounding direction (max error 0.5 ulp); the explicit 2-ulp
    directed margin below is what provides the certified direction."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
    else:
        num, den = x, 1
    assert num > 0 and den > 0
    rnum = ROUND_FLOOR if direction == "floor" else ROUND_CEILING
    rden = ROUND_CEILING if direction == "floor" else ROUND_FLOOR
    with localcontext() as ctx:
        ctx.prec = prec
        ctx.rounding = rnum
        ln_num = Decimal(num).ln()
        ulp = ln_num.copy_abs() * Decimal(10) ** (-(prec - 2))
        with localcontext() as c2:
            c2.prec = prec
            c2.rounding = rden
            ln_den = Decimal(den).ln()
            ulp2 = ln_den.copy_abs() * Decimal(10) ** (-(prec - 2))
        ctx.rounding = ROUND_FLOOR if direction == "floor" else ROUND_CEILING
        res = ln_num - ln_den
        margin = ulp + ulp2
        if direction == "floor":
            return res - margin
        return res + margin


def run_certificate(m, twoL, Kmax, W, nthreads=8, tbin=None):
    L = twoL // 2
    t_start = time.time()
    # ---- structural facts about the mask
    M = semigroup_mask(GENS, B)
    assert len(M) == 99 and M[0] == 0 and max(M) == B
    S, D, kappa = build_alphabets(M, B)
    assert max(S) <= Q - 1, "carry-free requirement"
    assert max(abs(d) for d in D) <= B, "balanced representation requirement"
    Mset = set(M)
    for d in D:  # witness check (F2)
        al, be = (kappa[d] + d) // 2, (kappa[d] - d) // 2
        assert al in Mset and be in Mset and al - be == d and al + be == kappa[d]
        assert kappa[d] == min(a + b for a in M for b in M if a - b == d)

    # ---- T_lower from the certified C DP
    if tbin is None:
        import os
        if not os.path.exists("./tdp64"):
            r = subprocess.run(["clang", "-O2", "-frounding-math",
                                "-DREALT=double", "-o", "tdp64", "tdp.c",
                                "-lpthread", "-lm"], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        with open("pairs_primary.txt", "w") as f:
            for d in D:
                f.write(f"{d} {kappa[d]}\n")
        r = subprocess.run(["./tdp64", "pairs_primary.txt", str(m), str(Kmax),
                            str(W), str(nthreads), f"t_final_m{m}.bin"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "[selftest] FE_DOWNWARD honored" in r.stderr
        tbin = f"t_final_m{m}.bin"
    with open(tbin, "rb") as f:
        shift, n = struct.unpack("<qq", f.read(16))
        G = np.frombuffer(f.read(8 * n), dtype="<f8")
    assert n >= twoL + 1
    Tfrac = Fraction(0)
    for x in G[:twoL + 1]:
        assert x >= 0
        Tfrac += Fraction(x)          # exact: doubles are dyadic rationals
    Tfrac *= Fraction(2) ** shift
    assert Tfrac > 0

    # ---- exact N_upper and Dupper (packed big-int poly power)
    coeffs_S = {s: 1 for s in S}
    coeffs_D = {}
    for d in D:
        coeffs_D[kappa[d]] = coeffs_D.get(kappa[d], 0) + 1
    Nv = poly_pow_prefix(coeffs_S, m, twoL, [twoL])[twoL]
    Dv = poly_pow_prefix(coeffs_D, m, twoL, [twoL])[twoL]

    # independent full-scale cross-check of packed-int arithmetic, mod primes
    for p in (2147483629, 2147482951):
        chk = poly_pow_prefix_modp(coeffs_S, m, twoL, twoL, p)
        assert Nv % p == chk, "packed-int N mismatch vs mod-p reference"
        chk = poly_pow_prefix_modp(coeffs_D, m, twoL, twoL, p)
        assert Dv % p == chk, "packed-int D mismatch vs mod-p reference"

    # ---- exact maxU and q
    mx, digits = max_U(M, m, L, Q)
    assert all(a in Mset for a in digits) and sum(digits) <= L
    q = 2 * mx + 1

    # ---- verified inequalities
    assert Tfrac <= Dv, "lower bound exceeds upper bound on |U-U| (impossible)"
    assert Dv < q, "|U-U| upper bound must be < 2 maxU + 1"
    assert Tfrac > Nv, "need T_lower > N_upper for theta > 1"

    # ---- certified theta via directed decimal rounding
    lnT_lo = ln_dir(Tfrac, "floor")
    lnN_hi = ln_dir(Nv, "ceil")
    lnq_hi = ln_dir(q, "ceil")
    lnq_lo = ln_dir(q, "floor")
    assert lnT_lo - lnN_hi > 0 and lnq_hi > 0
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_FLOOR
        theta_lo = Decimal(1) + (lnT_lo - lnN_hi) / lnq_hi
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        lnT_hi = ln_dir(Tfrac, "ceil")
        lnN_lo = ln_dir(Nv, "floor")
        theta_hi = Decimal(1) + (lnT_hi - lnN_lo) / lnq_lo  # info only

    ok = theta_lo > TARGET
    elapsed = time.time() - t_start
    with localcontext() as ctx:
        ctx.prec = 100
        theta_lo_str = str(+theta_lo.quantize(Decimal("1." + "0" * 30),
                                              rounding=ROUND_FLOOR))
        theta_hi_str = str(+theta_hi.quantize(Decimal("1." + "0" * 30),
                                              rounding=ROUND_CEILING))
        margin_str = str(+(theta_lo - TARGET).quantize(
            Decimal("1." + "0" * 15), rounding=ROUND_FLOOR))

    # ---- emit certificate
    def big(x, digits=40):
        s = str(x)
        return s if len(s) <= 2 * digits + 20 else f"{s[:digits]}...{s[-digits:]}<{len(s)} digits>"

    Tnum, Tden = Tfrac.numerator, Tfrac.denominator
    cert = {
        "problem": "teorth/optimizationproblems constant 3a (exact count record)",
        "target_to_exceed": str(TARGET),
        "construction": {
            "mask_semigroup_generators": list(GENS),
            "mask_interval": [0, B],
            "mask_size": len(M),
            "base_Q": Q,
            "m_digits": m,
            "budget_L": L,
            "definition": "U = { sum a_i Q^i : a_i in mask, sum a_i <= L }",
        },
        "certified_counts": {
            "N_upper_ge_sumset": str(Nv),
            "T_lower_le_diffset_numerator": str(Tnum),
            "T_lower_le_diffset_denominator_pow2": Tden.bit_length() - 1,
            "D_upper_ge_diffset": str(Dv),
            "maxU_exact": str(mx),
            "q_eq_2maxU_plus_1": str(q),
        },
        "dp_parameters": {
            "Kmax_budget_dim": Kmax,
            "corridor_W": W,
            "float_dp": "C float64, fesetround(FE_DOWNWARD), NEON+scalar, "
                        "power-of-two rescaling (exact); see tdp.c",
        },
        "verified_inequalities": {
            "T_lower <= |U-U|": "by construction (witness realizability, F2)",
            "|U+U| <= N_upper": "by construction (carry-free digit strings, F1)",
            "|U-U| <= D_upper": "by construction (min-cost necessity, F2)",
            "D_upper < q": bool(Dv < q),
            "T_lower > N_upper": True,
            "theta_lower > target": bool(ok),
        },
        "theta_certified_lower_bound": theta_lo_str,
        "theta_upper_estimate": theta_hi_str,
        "margin_over_target": margin_str,
        "log10_maxU_approx": float(Decimal(mx).ln() / Decimal(10).ln()),
        "runtime_verify_sec": round(elapsed, 1),
    }
    with open(f"certificate_m{m}.json", "w") as f:
        json.dump(cert, f, indent=1)
    print(json.dumps({k: v for k, v in cert.items()
                      if k not in ("certified_counts",)}, indent=1))
    print("N_upper  =", big(Nv))
    print("T_lower  = (num) ", big(Tnum), " / 2^", Tden.bit_length() - 1)
    print("D_upper  =", big(Dv))
    print("maxU     =", big(mx))
    print(f"theta_lo = {theta_lo}")
    print(f"RESULT: theta_certified = {cert['theta_certified_lower_bound']} "
          f"{'> ' if ok else '<= FAIL '} target {TARGET}")
    return cert, theta_lo


if __name__ == "__main__":
    m = int(sys.argv[1])
    twoL = int(sys.argv[2])
    Kmax = int(sys.argv[3]) if len(sys.argv) > 3 else twoL
    W = int(sys.argv[4]) if len(sys.argv) > 4 else int(math.ceil(4 * 79.3426 * math.sqrt(m)))
    tbin = sys.argv[5] if len(sys.argv) > 5 else None
    run_certificate(m, twoL, Kmax, W, tbin=tbin)
