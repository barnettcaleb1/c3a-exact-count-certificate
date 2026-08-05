"""Validate the C certified DP (tdp64/tdp32) against the exact int64 reference
on the real mask at m=6, with both a full corridor and a truncating corridor."""
import struct
import subprocess
import numpy as np
from fractions import Fraction

from mask_common import primary
from validate import t_dp_exact_reference


def write_pairs(D, kappa, fn="pairs_primary.txt"):
    with open(fn, "w") as f:
        for d in D:
            f.write(f"{d} {kappa[d]}\n")
    return fn


def run_tdp(binary, pairs, m, Kmax, W, nthreads=8, out="t_out.bin"):
    r = subprocess.run([f"./{binary}", pairs, str(m), str(Kmax), str(W),
                        str(nthreads), out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    with open(out, "rb") as f:
        shift, n = struct.unpack("<qq", f.read(16))
        G = np.frombuffer(f.read(8 * n), dtype="<f8")
    return shift, G, r.stderr


def T_lower_from_G(shift, G, twoL):
    """Exact rational lower bound: sum of doubles (each exact) * 2^shift."""
    s = Fraction(0)
    for x in G[:twoL + 1]:
        s += Fraction(x)
    return s * Fraction(2) ** shift


if __name__ == "__main__":
    M, S, D, kappa = primary()
    pairs = write_pairs(D, kappa)
    m = 6
    Kmax = 700
    for W in (189 * m, 260):
        print(f"--- m={m} Kmax={Kmax} corridor W={W}")
        thresholds = list(range(0, Kmax + 1, 50)) + [Kmax]
        ref = t_dp_exact_reference(D, kappa, m, Kmax, W, thresholds)
        for binary in ("tdp64", "tdp32"):
            shift, G, log = run_tdp(binary, pairs, m, Kmax, W)
            assert "[selftest] FE_DOWNWARD honored" in log
            worst = 0.0
            for t in thresholds:
                lo = T_lower_from_G(shift, G, t)
                exact = ref[t]
                assert lo <= exact, (binary, t, float(lo), exact)
                if exact:
                    rel = 1 - lo / exact
                    worst = max(worst, float(rel))
            print(f"   [ok] {binary}: certified lower <= exact at all {len(thresholds)} "
                  f"thresholds; worst relative slack = {worst:.3e}")
    print("C DP VALIDATION PASSED")
