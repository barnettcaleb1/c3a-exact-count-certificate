"""Independent replication of the T_lower difference-count DP for the 3a certificate.
Pure exact integer arithmetic (numpy int64 with certified floor-halving rescale).
Counts strings (d_1..d_m) in (M-M)^m with:
  - all prefix sums |sum_{i<=j} d_i| <= W          (corridor; dropping = undercount, safe)
  - sum kappa(d_i) <= 2L  and  sum kappa(d_i) + |sum d_i| <= 2L   (realizability)
Maintains invariant: state[c,t] <= true_count(c,t) / 2^shift  exactly (integers).
Author: independent re-implementation, written without reference to tdp.c.
"""
import numpy as np, sys, math, json
from fractions import Fraction

def semigroup(gens, cap):
    ok = np.zeros(cap+1, dtype=bool); ok[0]=True
    for x in range(cap+1):
        if ok[x]:
            for g in gens:
                if x+g <= cap: ok[x+g]=True
    return np.flatnonzero(ok).astype(np.int64)

def kappa_table(M):
    M = np.asarray(M, dtype=np.int64)
    A = np.repeat(M, len(M)); B = np.tile(M, len(M))
    du, didx = np.unique(A-B, return_inverse=True)
    kap = np.full(len(du), 1<<60, dtype=np.int64)
    np.minimum.at(kap, didx, A+B)
    return du, kap   # digits d, costs kappa(d)

def t_lower_dp(M, m, twoL, W, verbose=False, ckpt=None):
    du, kap = kappa_table(M)
    Cmax = twoL          # cost dimension 0..twoL
    # symmetric half-storage: t = |sum d| in 0..W ; state count(c, s) = count(c, -s)
    f = np.zeros((Cmax+1, W+1), dtype=np.int64)
    f[0,0] = 1
    shift = 0
    start_step = 0
    if ckpt is not None:
        import os
        if os.path.exists(ckpt):
            z = np.load(ckpt)
            f = z['f']; shift = int(z['shift']); start_step = int(z['step'])
            assert f.shape == (Cmax+1, W+1), "checkpoint shape mismatch"
            print(f"  resumed from checkpoint: step {start_step}, shift {shift}", flush=True)
    # rescale threshold: each target state accumulates <= 377 summands, each < CAP,
    # so CAP = 2^53 keeps every accumulation < 377*2^53 < 2^63 (no int64 overflow).
    CAP = np.int64(1)<<53
    kmax = int(kap.max())
    pairs = list(zip(du.tolist(), kap.tolist()))
    BLK = 4096
    block = np.zeros((BLK, W+1), dtype=np.int64)   # persistent temp (one block)
    # In-place descending-block update: row c of the new state depends only on OLD
    # rows <= c, so processing blocks from the top down and writing each block back
    # after it is fully computed uses a single state array (halves memory, no per-step
    # allocation churn).
    import time
    for step in range(start_step, m):
        mx = f.max()
        while mx >= CAP:
            f >>= 1            # floor halving: preserves lower-bound invariant
            shift += 1
            mx = f.max()
        rmax = min(Cmax, (step+1)*kmax)            # rows beyond this are zero
        c1 = rmax + 1
        while c1 > 0:
            c0 = max(0, c1 - BLK)
            nb = c1 - c0
            blk = block[:nb]
            blk[:] = 0
            for d, k in pairs:
                # dst rows r in [max(c0,k), c1), src rows r-k (old values: strictly
                # below the block for k >= nb, within the not-yet-written f otherwise)
                r0 = max(c0, k)
                if r0 >= c1: continue
                src = f[r0-k:c1-k]
                dst = blk[r0-c0:c1-c0]
                if d >= 0:
                    if d <= W:
                        dst[:, d:W+1] += src[:, 0:W+1-d]
                    ulo = max(0, d-W); uhi = min(d-1, W)
                    if ulo <= uhi:
                        dst[:, ulo:uhi+1] += src[:, d-ulo:d-uhi-1 if d-uhi-1>=0 else None:-1]
                else:
                    ad = -d
                    if ad <= W:
                        dst[:, 0:W+1-ad] += src[:, ad:W+1]
            f[c0:c1] = blk
            c1 = c0
        if verbose and (step+1) % 8 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] step {step+1}/{m} shift={shift} max={f.max()}", flush=True)
        if ckpt is not None and (step+1) % 16 == 0 and step+1 < m:
            np.savez(ckpt + '.tmp.npz', f=f, shift=shift, step=step+1)
            import os
            os.replace(ckpt + '.tmp.npz', ckpt)
    # final: admissible states c + t <= twoL
    # overflow-safe exact tally: partial sums of <=512 values (512 * 2^62 would overflow, but
    # values are < 2^62 only transiently; post-step values < 377*CAP = 2^61.6, and 512-chunk
    # partial sums could overflow int64 -> chunk at 2: instead promote to object dtype per column.
    total = 0
    for t in range(0, W+1):
        cmaxa = twoL - t
        if cmaxa < 0: break
        col = f[0:cmaxa+1, t]
        # exact: convert to Python ints in chunks (bounded memory)
        colsum = 0
        for i0 in range(0, len(col), 1_000_000):
            colsum += int(np.add.reduce(col[i0:i0+1_000_000].astype(object)))
        total += colsum * (2 if t>0 else 1)
    return Fraction(total) * (1<<shift), shift   # lower bound on true count (as integer*2^shift)

def brute_force_check():
    """Validate on small analogs by full enumeration."""
    ok_all = True
    for gens, cap, base_note, ms, Ls in [([3,5], 7, 15, [2,3], [6,9,12]), ([24,26,36,39], 189, 379, [2], [120,189])]:
        M = semigroup(gens, cap)
        du, kap = kappa_table(M)
        kd = {int(d):int(k) for d,k in zip(du,kap)}
        Q = 2*int(M[-1])+1
        for m in ms:
            for L in Ls:
                twoL = 2*L
                W = min(int(M[-1])*m, twoL)  # generous corridor for validation
                # enumerate the sufficient set directly over strings
                import itertools
                cnt = 0
                for s in itertools.product(du.tolist(), repeat=m):
                    pref=0; okp=True
                    for d in s:
                        pref+=d
                        if abs(pref)>W: okp=False; break
                    if not okp: continue
                    ck = sum(kd[d] for d in s)
                    if ck + abs(sum(s)) <= twoL: cnt+=1
                val, sh = t_lower_dp(M, m, twoL, W)
                exact_match = (val == cnt) and sh==0
                # also verify the counted set is a subset of actual U-U (realizability), on the smallest case
                print(f"gens={gens} m={m} L={L}: brute={cnt} dp={val} shift={sh} match={exact_match}")
                if not exact_match: ok_all=False
    print("VALIDATION:", "PASS" if ok_all else "FAIL")
    return ok_all

if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "validate":
        brute_force_check()

def run_full(m, twoL, W):
    M = semigroup([24,26,36,39], 189)
    val, sh = t_lower_dp(M, m, twoL, W, verbose=True, ckpt=f"ckpt_m{m}.npz")
    # accurate log via Decimal
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    lg = Decimal(val.numerator).ln() - Decimal(val.denominator).ln()
    print(f"m={m} twoL={twoL} W={W}: T_mine = {float(lg):.10f} (natural log)")
    json.dump({"m": m, "twoL": twoL, "W": W, "logT_mine": str(lg), "T_num_digits": len(str(val.numerator))},
              open(f"indep_T_m{m}.json","w"), indent=1)
    return val

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "full":
    run_full(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
