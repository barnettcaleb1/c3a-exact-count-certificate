"""Pure-integer certificate check: proves theta(U) > 1 + a/b with NO floating point.
theta > 1 + a/b  <=>  log(T/N)/log(q) > a/b  <=>  T^b > N^b * q^a   (T > N, q > 1).
Run: python3 int_check.py [a b]   (defaults 1873 10000, i.e. theta > 1.1873)
Reads certificate.json in the current directory."""
import json, sys

a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (1873, 10000)
cc = json.load(open('certificate.json'))['certified_counts']
N = int(cc['N_upper_ge_sumset'])
T = int(cc['T_lower_le_diffset_numerator'])
assert int(cc.get('T_lower_le_diffset_denominator_pow2', 0)) == 0, "dyadic T not supported here"
D = int(cc['D_upper_ge_diffset']); mx = int(cc['maxU_exact']); q = int(cc['q_eq_2maxU_plus_1'])
assert q == 2*mx + 1
assert D < q, "admissibility |U-U| < 2 maxU + 1 must hold"
assert T <= D and T > N > 0
lhs = pow(T, b)
rhs = pow(N, b) * pow(q, a)
assert lhs > rhs, f"integer check FAILED for a/b = {a}/{b}"
print(f"PURE-INTEGER CERTIFICATE OK: T^{b} > N^{b} * q^{a}")
print(f"  hence theta(U) > 1 + {a}/{b} = {1 + a/b}")
print(f"  (and D < q verified in exact integers)")
