"""Exact big-integer 1D DPs via packed-integer truncated polynomial powering.

Counts are coefficients of P(x)^m truncated to degree <= Tmax, where P has
nonnegative integer coefficients. A polynomial is packed into a single Python
integer with slot width Wbits per coefficient; products of packed integers are
packed products of polynomials as long as no coefficient reaches 2^Wbits.

Used for:
  N_upper(2L)  = #{(s_i) in (M+M)^m : sum s_i <= 2L}      >= |U+U|   (exact count)
  Dupper(2L)   = #{(d_i) in (M-M)^m : sum kappa(d_i) <= 2L} >= |U-U|  (exact count)
All arithmetic is exact (Python big ints).
"""
import math


def pack_poly(coeffs, Wbits):
    """coeffs: dict degree->int (nonnegative). Returns packed integer."""
    n = 0
    for deg, c in coeffs.items():
        assert c >= 0 and c < (1 << Wbits)
        n |= c << (Wbits * deg)
    return n


def poly_pow_trunc(coeffs, m, Tmax, total_weight):
    """Compute Q(x) = P(x)^m mod x^(Tmax+1) exactly, P = sum coeffs[deg] x^deg.

    total_weight = P(1) (sum of coefficients); every coefficient of any partial
    product P^j (j <= m) is <= total_weight^j <= total_weight^m, which bounds
    slot requirements. Returns (packed_int, Wbits).
    """
    # coefficient bound for all partial products: total_weight^m
    # prefix sums later need an extra log2(total sum) margin; add 32 bits.
    Wbits = int(math.ceil(m * math.log2(total_weight))) + 32
    # round up to multiple of 8 for sanity
    Wbits = (Wbits + 7) // 8 * 8
    mask = (1 << (Wbits * (Tmax + 1))) - 1
    P = pack_poly(coeffs, Wbits) & mask
    # binary exponentiation with truncation
    result = 1  # polynomial "1"
    base = P
    e = m
    while e > 0:
        if e & 1:
            result = (result * base) & mask
        e >>= 1
        if e:
            base = (base * base) & mask
    return result, Wbits


def prefix_counts(packed, Wbits, Tmax, thresholds):
    """Given packed coefficients of Q(x) (degrees 0..Tmax), return
    {t: sum_{j<=t} coeff_j} for each t in thresholds. Exact.

    Implemented by multiplying with the all-ones polynomial 1+x+...+x^Tmax:
    coefficient of x^t in the product is the prefix sum up to t.
    Slot width must accommodate prefix sums; caller guaranteed 32 bits margin
    over total_weight^m which bounds the FULL sum, so no overflow.
    """
    ones = (1 << (Wbits * (Tmax + 1))) - 1
    # ones = sum_{j=0}^{Tmax} (2^Wbits)^j * (2^Wbits - 1)... careful: that's not
    # the all-ones polynomial. Build it properly:
    # all-ones poly packed = sum_j 1 << (Wbits*j) = (2^(Wbits*(Tmax+1)) - 1) / (2^Wbits - 1)
    ones = ((1 << (Wbits * (Tmax + 1))) - 1) // ((1 << Wbits) - 1)
    mask = (1 << (Wbits * (Tmax + 1))) - 1
    prod = (packed * ones) & mask
    out = {}
    slotmask = (1 << Wbits) - 1
    for t in thresholds:
        assert 0 <= t <= Tmax
        out[t] = (prod >> (Wbits * t)) & slotmask
    return out


def poly_pow_prefix(coeffs, m, Tmax, thresholds):
    """Convenience: exact prefix sums of coefficients of P^m at thresholds."""
    total_weight = sum(coeffs.values())
    packed, Wbits = poly_pow_trunc(coeffs, m, Tmax, total_weight)
    return prefix_counts(packed, Wbits, Tmax, thresholds)


def unpack_all(packed, Wbits, Tmax):
    """Unpack all coefficients (for validation on small cases). O(T log T)."""
    def split(n, nslots):
        if nslots == 1:
            return [n]
        half = nslots // 2
        lo = n & ((1 << (Wbits * half)) - 1)
        hi = n >> (Wbits * half)
        return split(lo, half) + split(hi, nslots - half)
    out = split(packed, Tmax + 1)
    return out


def direct_pow_coeffs(coeffs, m, Tmax):
    """Slow reference: exact coefficients of P^m truncated (list length Tmax+1)."""
    cur = [0] * (Tmax + 1)
    cur[0] = 1
    for _ in range(m):
        nxt = [0] * (Tmax + 1)
        for deg, c in coeffs.items():
            if c == 0:
                continue
            for j in range(0, Tmax + 1 - deg):
                if cur[j]:
                    nxt[j + deg] += cur[j] * c
        cur = nxt
    return cur


def max_U(M, m, L, Q):
    """Exact max of U(m,L) by greedy: most significant digit largest affordable.

    Correctness: raising a digit at position i by delta >= 1 gains delta*Q^i,
    while any budget delta spent at positions < i gains at most
    delta * Q^(i-1) * Q/(Q-1) < delta * Q^i.  So lexicographically-largest
    digit string (top down) maximizes the value.  0 in M so any remainder is fine.
    """
    Msorted = sorted(M)
    rem = L
    val = 0
    digits = []
    for i in range(m - 1, -1, -1):
        # largest a in M with a <= rem
        a = 0
        for x in Msorted:
            if x <= rem:
                a = x
            else:
                break
        rem -= a
        digits.append(a)
        val += a * (Q ** i)
    return val, digits


def max_U_dp(M, m, L, Q):
    """Reference exact maximization by DP over budget (for validation)."""
    NEG = None
    best = {0: 0}  # budget used -> max value, per position processed
    # process positions from least significant (i=0) to most significant
    cur = [0] * (L + 1)  # cur[b] = max value using budget exactly... use <= b
    for i in range(m):
        w = Q ** i
        nxt = [0] * (L + 1)
        for b in range(L + 1):
            bestv = 0
            for a in M:
                if a <= b:
                    v = a * w + cur[b - a]
                    if v > bestv:
                        bestv = v
            nxt[b] = bestv
        # enforce monotone in b (budget <= b)
        for b in range(1, L + 1):
            if nxt[b] < nxt[b - 1]:
                nxt[b] = nxt[b - 1]
        cur = nxt
    return cur[L]
