"""Scan m (ladder) and L (grid near mu*m/2), computing certified theta values.

For each m:
  T_lower(2L): C program tdp64 (certified lower, directed rounding, corridor)
  N_upper(2L): exact packed-int poly power  (upper bound on |U+U|)
  Dupper(2L):  exact packed-int poly power  (upper bound on |U-U|)
  maxU(L):     exact greedy
  theta(L) = 1 + log(T_lower/N_upper)/log(2 maxU + 1)   [float here; the final
             certificate re-evaluates with directed Decimal rounding]
"""
import json
import math
import sys
import time

from mask_common import primary, limit_value, Q
from exact1d import poly_pow_prefix, max_U
from validate_c import write_pairs, run_tdp, T_lower_from_G

MU = 92.5357672  # E_nu[kappa] at lambda*; only used to center the scan grid
SIGMA_D = 79.3426

TARGET = 1.1835129324


def scan_m(m, nthreads=8, corridor_mult=4.0, grid_lo=0.86, grid_hi=1.04,
           npts=28, binary="tdp64"):
    M, S, D, kappa = primary()
    pairs = write_pairs(D, kappa)
    Kmax = int(math.ceil(grid_hi * MU * m)) + 2
    W = int(math.ceil(corridor_mult * SIGMA_D * math.sqrt(m)))
    t0 = time.time()
    shift, G, log = run_tdp(binary, pairs, m, Kmax, W, nthreads,
                            out=f"t_m{m}.bin")
    t_dp = time.time() - t0
    assert "[selftest] FE_DOWNWARD honored" in log

    # grid of L values (2L even, spanning [grid_lo, grid_hi]*mu*m)
    twoLs = sorted({2 * (int(g * MU * m) // 2)
                    for g in [grid_lo + (grid_hi - grid_lo) * i / (npts - 1)
                              for i in range(npts)]})
    twoLs = [t for t in twoLs if t <= Kmax]

    t0 = time.time()
    coeffs_S = {s: 1 for s in S}
    coeffs_D = {}
    for d in D:
        coeffs_D[kappa[d]] = coeffs_D.get(kappa[d], 0) + 1
    Nv = poly_pow_prefix(coeffs_S, m, Kmax, twoLs)
    Dv = poly_pow_prefix(coeffs_D, m, Kmax, twoLs)
    t_1d = time.time() - t0

    rows = []
    best = None
    for twoL in twoLs:
        L = twoL // 2
        T = T_lower_from_G(shift, G, twoL)
        N = Nv[twoL]
        Dup = Dv[twoL]
        mx, _ = max_U(M, m, L, Q)
        q = 2 * mx + 1
        ok_diff = Dup < q
        if T <= 0 or N <= 0:
            continue
        # float evaluation for scanning
        logT = math.log(T.numerator) - math.log(T.denominator)
        theta = 1 + (logT - math.log(N)) / math.log(q)
        rows.append({"twoL": twoL, "theta": theta, "Dup_lt_q": bool(ok_diff),
                     "logT": logT, "logN": math.log(N),
                     "logq": math.log(q)})
        if ok_diff and (best is None or theta > best["theta"]):
            best = rows[-1]
    out = {"m": m, "Kmax": Kmax, "W": W, "corridor_mult": corridor_mult,
           "t_dp_sec": t_dp, "t_1d_sec": t_1d, "rows": rows, "best": best}
    with open(f"scan_m{m}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ms = [int(x) for x in sys.argv[1:]] or [32, 64]
    for m in ms:
        r = scan_m(m)
        b = r["best"]
        print(f"m={m:4d} (dp {r['t_dp_sec']:.1f}s, 1d {r['t_1d_sec']:.1f}s) "
              f"best 2L={b['twoL']} theta={b['theta']:.10f} "
              f"margin={b['theta']-TARGET:+.7f} Dup<q={b['Dup_lt_q']}")
        for row in r["rows"]:
            flag = " <-- best" if row is b or row["twoL"] == b["twoL"] else ""
            print(f"    2L={row['twoL']:6d} theta={row['theta']:.10f}"
                  f" Dup<q={row['Dup_lt_q']}{flag}")
