"""Validation of all DP logic against brute-force enumeration on small cases.

Cases:
  A) small analog mask  M' = <3,5> cap [0,7] = {0,3,5,6},  B'=7, Q'=15, m=2,3
  B) real mask <24,26,36,39> cap [0,189], m=2 (full) and m=3 (small L)
Checks:
  1) N_upper (packed-poly DP) == #{s-strings: sum s_i <= 2L} (enumerated)
     and >= |U+U| (enumerated)
  2) T_count (string condition sum kappa + |sum d| <= 2L) <= |U-U| (enumerated),
     realizability witness check string-by-string
  3) Dupper (packed-poly DP) == #{d-strings: sum kappa <= 2L} (enumerated)
     and >= |U-U|
  4) greedy max_U == max(U) (enumerated) == max_U_dp (reference DP)
  5) exact reference 2D T-DP (numpy int64, signed D) == enumeration of T strings
  6) packed poly pow == direct slow convolution reference
"""
import itertools
import math
import numpy as np

from mask_common import semigroup_mask, build_alphabets, primary
from exact1d import (poly_pow_prefix, poly_pow_trunc, unpack_all,
                     direct_pow_coeffs, max_U, max_U_dp)


def enumerate_U(M, m, L, Q):
    """All elements of U(m,L) as a sorted numpy int64 array (small cases)."""
    us = []
    for digits in itertools.product(M, repeat=m):
        if sum(digits) <= L:
            us.append(sum(a * Q**i for i, a in enumerate(digits)))
    u = np.array(sorted(set(us)), dtype=np.int64)
    assert len(u) == len(us), "digit strings must map injectively to integers"
    return u


def sumset_size(u):
    s = u[:, None] + u[None, :]
    return len(np.unique(s))


def diffset_size(u):
    d = u[:, None] - u[None, :]
    return len(np.unique(d))


def enumerate_T_strings(D, kappa, m, twoL, Q, M, L, check_witness=False):
    """#{(d_i) in D^m : sum kappa(d_i) + |sum d_i| <= 2L}; optionally verify the
    minimum-cost witness realizes each counted string within budget."""
    Mset = set(M)
    cnt = 0
    diffs = set()
    for ds in itertools.product(D, repeat=m):
        ksum = sum(kappa[d] for d in ds)
        dsum = sum(ds)
        if ksum + abs(dsum) <= twoL:
            cnt += 1
            if check_witness:
                alphas = [(kappa[d] + d) // 2 for d in ds]
                betas = [(kappa[d] - d) // 2 for d in ds]
                assert all(a in Mset for a in alphas)
                assert all(b in Mset for b in betas)
                assert sum(alphas) <= L and sum(betas) <= L, (ds, ksum, dsum)
                u = sum(a * Q**i for i, a in enumerate(alphas))
                v = sum(b * Q**i for i, b in enumerate(betas))
                dval = sum(d * Q**i for i, d in enumerate(ds))
                assert u - v == dval
                diffs.add(dval)
    if check_witness:
        assert len(diffs) == cnt, "counted d-strings must be distinct differences"
    return cnt


def enumerate_N_strings(S, m, twoL):
    cnt = 0
    for ss in itertools.product(S, repeat=m):
        if sum(ss) <= twoL:
            cnt += 1
    return cnt


def enumerate_D_strings(D, kappa, m, twoL):
    cnt = 0
    for ds in itertools.product(D, repeat=m):
        if sum(kappa[d] for d in ds) <= twoL:
            cnt += 1
    return cnt


def t_dp_exact_reference(D, kappa, m, Kmax, W, thresholds):
    """Exact reference 2D DP over (sum kappa, sum d), numpy int64 (small m only).

    Enforces the corridor |prefix sum of d| <= W (same as the C program).
    Returns {t: count of strings with sum kappa + |sum d| <= t}.
    """
    total = sum(1 for _ in D) ** m
    assert total < 2**62, "int64 reference only valid for tiny cases"
    # state[k, W + Dsum]
    cur = np.zeros((Kmax + 1, 2 * W + 1), dtype=np.int64)
    cur[0, W] = 1
    for _ in range(m):
        nxt = np.zeros_like(cur)
        for d in D:
            c = kappa[d]
            if c > Kmax:
                continue
            # shift k by c, D by d, clip corridor
            src_lo = max(0, -d)
            src_hi = min(2 * W + 1, 2 * W + 1 - d)
            if src_lo >= src_hi:
                continue
            nxt[c:, src_lo + d:src_hi + d] += cur[:Kmax + 1 - c, src_lo:src_hi]
        cur = nxt
    out = {}
    for t in thresholds:
        tot = 0
        for k in range(min(t, Kmax) + 1):
            dmax = t - k
            lo = max(0, W - dmax)
            hi = min(2 * W, W + dmax)
            tot += int(cur[k, lo:hi + 1].sum())
        out[t] = tot
    return out


def run_case(name, gens, B, m, Ls, enumerate_full=True):
    Q = 2 * B + 1
    M = semigroup_mask(gens, B)
    S, D, kappa = build_alphabets(M, B)
    print(f"--- case {name}: M={M if len(M)<20 else f'|M|={len(M)}'} B={B} Q={Q} m={m}")
    Kmax = max(2 * L for L in Ls)
    thresholds = sorted(set(2 * L for L in Ls))
    # packed-poly N and Dupper
    coeffs_S = {s: 1 for s in S}
    coeffs_D = {}
    for d in D:
        coeffs_D[kappa[d]] = coeffs_D.get(kappa[d], 0) + 1
    Nvals = poly_pow_prefix(coeffs_S, m, Kmax, thresholds)
    Dvals = poly_pow_prefix(coeffs_D, m, Kmax, thresholds)
    # check 6: packed pow == direct reference (N poly)
    packed, Wbits = poly_pow_trunc(coeffs_S, m, Kmax, sum(coeffs_S.values()))
    ref = direct_pow_coeffs(coeffs_S, m, Kmax)
    assert unpack_all(packed, Wbits, Kmax) == ref, "packed poly pow mismatch"
    print("   [ok] packed-int poly power == direct convolution (N alphabet)")
    packedD, WbitsD = poly_pow_trunc(coeffs_D, m, Kmax, sum(coeffs_D.values()))
    refD = direct_pow_coeffs(coeffs_D, m, Kmax)
    assert unpack_all(packedD, WbitsD, Kmax) == refD, "packed poly pow mismatch D"
    print("   [ok] packed-int poly power == direct convolution (kappa alphabet)")
    # reference exact 2D T DP with huge corridor (no corridor loss)
    Wcorr = B * m
    Tref = t_dp_exact_reference(D, kappa, m, Kmax, Wcorr, thresholds)

    for L in Ls:
        twoL = 2 * L
        if enumerate_full:
            u = enumerate_U(M, m, L, Q)
            nUU = sumset_size(u)
            nDD = diffset_size(u)
            mx = int(u.max())
            g, _ = max_U(M, m, L, Q)
            gdp = max_U_dp(M, m, L, Q)
            assert g == mx == gdp, (g, mx, gdp)
            Nstr = enumerate_N_strings(S, m, twoL)
            Tstr = enumerate_T_strings(D, kappa, m, twoL, Q, M, L,
                                       check_witness=True)
            Dstr = enumerate_D_strings(D, kappa, m, twoL)
            assert Nvals[twoL] == Nstr, (Nvals[twoL], Nstr)
            assert Dvals[twoL] == Dstr
            assert Tref[twoL] == Tstr, (Tref[twoL], Tstr)
            assert nUU <= Nvals[twoL], "N_upper must dominate |U+U|"
            assert Tstr <= nDD <= Dstr, "T_lower <= |U-U| <= Dupper violated"
            print(f"   [ok] L={L}: |U|={len(u)} |U+U|={nUU} <= N={Nvals[twoL]}; "
                  f"T={Tstr} <= |U-U|={nDD} <= D={Dvals[twoL]}; maxU={mx} (greedy==dp==enum)")
    return M, S, D, kappa


if __name__ == "__main__":
    # A) small analog, m=2 and m=3
    run_case("analog<3,5>", (3, 5), 7, 2, [4, 6, 8, 10, 12, 14])
    run_case("analog<3,5>", (3, 5), 7, 3, [6, 9, 12, 15])
    print("small-analog cases passed.\n")

    # B) real mask, m=2 (full pair enumeration is fine: |U|<=9801)
    M, S, D, kappa = primary()
    Q = 379
    for L in [120, 189, 260]:
        u = enumerate_U(M, 2, L, Q)
        nUU = sumset_size(u)
        nDD = diffset_size(u)
        mx = int(u.max())
        g, _ = max_U(M, 2, L, Q)
        assert g == mx
        twoL = 2 * L
        Kmax = twoL
        coeffs_S = {s: 1 for s in S}
        coeffs_D = {}
        for d in D:
            coeffs_D[kappa[d]] = coeffs_D.get(kappa[d], 0) + 1
        Nv = poly_pow_prefix(coeffs_S, 2, Kmax, [twoL])[twoL]
        Dv = poly_pow_prefix(coeffs_D, 2, Kmax, [twoL])[twoL]
        Tv = t_dp_exact_reference(D, kappa, 2, Kmax, 189 * 2, [twoL])[twoL]
        # T string count via vectorized enumeration
        darr = np.array(D, dtype=np.int64)
        karr = np.array([kappa[d] for d in D], dtype=np.int64)
        ks = karr[:, None] + karr[None, :]
        dsum = darr[:, None] + darr[None, :]
        Tstr = int(((ks + np.abs(dsum)) <= twoL).sum())
        assert Tv == Tstr
        assert nUU <= Nv and Tv <= nDD <= Dv
        print(f"real mask m=2 L={L}: |U|={len(u)} |U+U|={nUU}<=N={Nv} "
              f"T={Tv}<=|U-U|={nDD}<=D={Dv} maxU ok")

    # real mask, m=3, small L to keep |U| manageable
    L = 130
    u = enumerate_U(M, 3, L, Q)
    print(f"real mask m=3 L={L}: |U|={len(u)}")
    nUU = sumset_size(u)
    nDD = diffset_size(u)
    g, _ = max_U(M, 3, L, Q)
    assert g == int(u.max())
    twoL = 2 * L
    coeffs_S = {s: 1 for s in S}
    coeffs_D = {}
    for d in D:
        coeffs_D[kappa[d]] = coeffs_D.get(kappa[d], 0) + 1
    Nv = poly_pow_prefix(coeffs_S, 3, twoL, [twoL])[twoL]
    Dv = poly_pow_prefix(coeffs_D, 3, twoL, [twoL])[twoL]
    Tv = t_dp_exact_reference(D, kappa, 3, twoL, 189 * 3, [twoL])[twoL]
    darr = np.array(D, dtype=np.int64)
    karr = np.array([kappa[d] for d in D], dtype=np.int64)
    ks = (karr[:, None, None] + karr[None, :, None] + karr[None, None, :])
    dsum = (darr[:, None, None] + darr[None, :, None] + darr[None, None, :])
    Tstr = int(((ks + np.abs(dsum)) <= twoL).sum())
    assert Tv == Tstr, (Tv, Tstr)
    assert nUU <= Nv and Tv <= nDD <= Dv
    print(f"real mask m=3 L={L}: |U+U|={nUU}<=N={Nv} T={Tv}<=|U-U|={nDD}<=D={Dv}")
    print("\nALL VALIDATION CHECKS PASSED")
