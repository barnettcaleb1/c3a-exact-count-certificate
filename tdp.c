/* tdp.c — certified LOWER bound DP for
 *   T(t) = #{ (d_1..d_m) in (M-M)^m :  sum kappa(d_i) + |sum d_i| <= t,
 *             and all prefix sums |d_1+..+d_j| <= W (corridor) }
 * The corridor restriction only removes strings (certified-lower direction).
 *
 * State: H_j(k, D) = # length-j prefixes with sum kappa = k, |sum d| = D
 * stored for D >= 0 only (distribution symmetric in D since kappa(d)=kappa(-d)).
 * Recurrence: H_j(k,D) = sum_d H_{j-1}(k - kappa(d), |D - d|)  (drop |D-d| > W).
 *
 * ALL floating point additions are performed with fesetround(FE_DOWNWARD),
 * so every stored value is <= the exact integer count divided by 2^shift.
 * Rescaling multiplies by exact powers of two (0x1p-64): values that fall to
 * zero/denormal only lower the bound further. Output G[t] (double, rounded
 * down) with G[t] * 2^shift <= sum over {k + D = t} of (D==0 ? H : 2H).
 * So  sum_{t<=2L} G[t] * 2^shift  <=  T(2L)  for every 2L <= Kmax.
 *
 * Usage: tdp pairs.txt m Kmax W nthreads out.bin
 *   pairs.txt: lines "d kappa" for every d in M-M (signed, includes negatives)
 *   out.bin:   int64 shift, int64 n=Kmax+1, n doubles G[0..Kmax]
 *
 * Compile: clang -O2 -frounding-math -o tdp tdp.c -lpthread -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <pthread.h>
#include <stdint.h>

#pragma STDC FENV_ACCESS ON

#if defined(__ARM_NEON) && !defined(NO_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

#ifndef REALT
#define REALT double
#endif
typedef REALT real;

/* ------------ alphabet groups: distinct kappa -> positive d list ---------- */
typedef struct {
    int c;            /* kappa value */
    int nd;           /* number of positive d's */
    int *ds;          /* positive d values */
    int has_zero;     /* group contains d == 0 */
} Group;

static Group *groups; static int ngroups;
static int KAPPA_MAX = 0, D_ABS_MAX = 0;

static void load_pairs(const char *fn) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror("pairs"); exit(1); }
    int d, c, n = 0, cap = 1024;
    int (*pairs)[2] = malloc(sizeof(int[2]) * cap);
    while (fscanf(f, "%d %d", &d, &c) == 2) {
        if (n == cap) { cap *= 2; pairs = realloc(pairs, sizeof(int[2]) * cap); }
        pairs[n][0] = d; pairs[n][1] = c; n++;
        if (c > KAPPA_MAX) KAPPA_MAX = c;
        if (abs(d) > D_ABS_MAX) D_ABS_MAX = abs(d);
    }
    fclose(f);
    /* reject duplicate d entries (a duplicate would overcount: unsafe) */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (pairs[i][0] == pairs[j][0]) {
                fprintf(stderr, "duplicate alphabet entry d=%d\n", pairs[i][0]);
                exit(1);
            }
    /* verify symmetry: for every (d,c) with d>0 there is (-d,c) */
    for (int i = 0; i < n; i++) {
        if (pairs[i][0] <= 0) continue;
        int found = 0;
        for (int j = 0; j < n; j++)
            if (pairs[j][0] == -pairs[i][0] && pairs[j][1] == pairs[i][1]) found = 1;
        if (!found) { fprintf(stderr, "alphabet not symmetric at d=%d\n", pairs[i][0]); exit(1); }
    }
    /* group positive d (and the d=0 flag) by kappa */
    groups = calloc(n, sizeof(Group)); ngroups = 0;
    for (int i = 0; i < n; i++) {
        if (pairs[i][0] < 0) continue;   /* negatives folded into their +pair */
        int c2 = pairs[i][1], gi = -1;
        for (int g = 0; g < ngroups; g++) if (groups[g].c == c2) { gi = g; break; }
        if (gi < 0) { gi = ngroups++; groups[gi].c = c2; groups[gi].ds = malloc(sizeof(int) * n); }
        if (pairs[i][0] == 0) groups[gi].has_zero = 1;
        else groups[gi].ds[groups[gi].nd++] = pairs[i][0];
    }
    free(pairs);
}

/* --------------------- rounding-mode self test ---------------------------- */
static void self_test_rounding(void) {
    fesetround(FE_DOWNWARD);
    volatile float  a1 = 1.0f, b1 = 1.75f * 0x1p-24f;
    volatile double a2 = 1.0,  b2 = 1.75  * 0x1p-53;
    float  r1 = a1 + b1;   /* exact = 1 + 1.75*2^-24, down -> 1.0f      */
    double r2 = a2 + b2;   /* exact = 1 + 1.75*2^-53, down -> 1.0       */
    if (r1 != 1.0f || r2 != 1.0) { fprintf(stderr, "FE_DOWNWARD scalar FAIL\n"); exit(2); }
#if USE_NEON
    {
        volatile float av = 1.0f, bv = 1.75f * 0x1p-24f;
        float32x4_t va = vdupq_n_f32(av), vb = vdupq_n_f32(bv);
        float32x4_t vr = vaddq_f32(va, vb);
        if (vgetq_lane_f32(vr, 0) != 1.0f) { fprintf(stderr, "FE_DOWNWARD neon f32 FAIL\n"); exit(2); }
        volatile double ad = 1.0, bd = 1.75 * 0x1p-53;
        float64x2_t vad = vdupq_n_f64(ad), vbd = vdupq_n_f64(bd);
        float64x2_t vrd = vaddq_f64(vad, vbd);
        if (vgetq_lane_f64(vrd, 0) != 1.0) { fprintf(stderr, "FE_DOWNWARD neon f64 FAIL\n"); exit(2); }
    }
#endif
    fesetround(FE_TONEAREST);
    volatile float a3 = 1.0f, b3 = 1.75f * 0x1p-24f;
    float r3 = a3 + b3;    /* nearest -> 1 + 2^-23 */
    if (r3 != 1.0f + 0x1p-23f) { fprintf(stderr, "rounding self-test sanity FAIL\n"); exit(2); }
    fesetround(FE_DOWNWARD);
    fprintf(stderr, "[selftest] FE_DOWNWARD honored (scalar%s)\n", USE_NEON ? " + NEON" : "");
}

/* ------------------------------- DP ------------------------------------- */
static long Kmax, W, stride;
static real *cur, *nxt;
static int kmax_act, kmax_act_prev, dmax_act, dmax_act_prev;

typedef struct { long k0, k1; real blockmax; } Job;

static void *worker(void *arg) {
    Job *jb = (Job *)arg;
    fesetround(FE_DOWNWARD);           /* per-thread FPCR */
    real bm = 0;
    for (long k = jb->k0; k < jb->k1; k++) {
        real *out = nxt + k * stride;
        memset(out, 0, sizeof(real) * (W + 1));
        long Dact = dmax_act;          /* write extent this position */
        for (int g = 0; g < ngroups; g++) {
            long c = groups[g].c;
            if (c > k) continue;
            if (k - c > kmax_act_prev) continue;   /* source row is all zero */
            const real *r = cur + (k - c) * stride;
            if (groups[g].has_zero) {
                long D = 0;
#if USE_NEON
                if (sizeof(real) == 8) {
                    for (; D + 2 <= Dact + 1; D += 2) {
                        float64x2_t vo = vld1q_f64((const double *)(out + D));
                        float64x2_t vr = vld1q_f64((const double *)(r + D));
                        vst1q_f64((double *)(out + D), vaddq_f64(vo, vr));
                    }
                }
#endif
                for (; D <= Dact; D++) out[D] += r[D];
            }
            for (int t = 0; t < groups[g].nd; t++) {
                long d = groups[g].ds[t];
                /* region A: 0 <= D <= min(d-1, Dact): r[d-D] (+ r[D+d] if <=W) */
                long hiA = d - 1 < Dact ? d - 1 : Dact;
                for (long D = 0; D <= hiA; D++) {
                    real v = 0;
                    if (d - D <= W) v = r[d - D];        /* reflected read */
                    if (D + d <= W) v += r[D + d];
                    out[D] += v;
                }
                /* region B: d <= D <= min(W-d, Dact): r[D-d] + r[D+d] */
                long loB = d, hiB = (W - d < Dact ? W - d : Dact);
                long D = loB;
#if USE_NEON
                if (sizeof(real) == 8) {
                    for (; D + 2 <= hiB + 1; D += 2) {
                        float64x2_t vo = vld1q_f64((const double *)(out + D));
                        float64x2_t va = vld1q_f64((const double *)(r + D - d));
                        float64x2_t vb = vld1q_f64((const double *)(r + D + d));
                        vo = vaddq_f64(vo, vaddq_f64(va, vb));
                        vst1q_f64((double *)(out + D), vo);
                    }
                } else if (sizeof(real) == 4) {
                    for (; D + 4 <= hiB + 1; D += 4) {
                        float32x4_t vo = vld1q_f32((const float *)(out + D));
                        float32x4_t va = vld1q_f32((const float *)(r + D - d));
                        float32x4_t vb = vld1q_f32((const float *)(r + D + d));
                        vo = vaddq_f32(vo, vaddq_f32(va, vb));
                        vst1q_f32((float *)(out + D), vo);
                    }
                }
#endif
                for (; D <= hiB; D++) out[D] += r[D - d] + r[D + d];
                /* region C: max(d, W-d+1) <= D <= Dact: r[D-d] only */
                long loC = (d > W - d + 1 ? d : W - d + 1);
                for (D = loC; D <= Dact; D++) out[D] += r[D - d];
            }
        }
        for (long D = 0; D <= Dact; D++) if (out[D] > bm) bm = out[D];
    }
    jb->blockmax = bm;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s pairs.txt m Kmax W nthreads out.bin\n", argv[0]);
        return 1;
    }
    load_pairs(argv[1]);
    long m = atol(argv[2]);
    Kmax = atol(argv[3]);
    W = atol(argv[4]);
    int nthreads = atoi(argv[5]);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 256) nthreads = 256;   /* fixed th[256]/jobs[256] arrays */
    const char *outfn = argv[6];
    self_test_rounding();
    fprintf(stderr, "[tdp] m=%ld Kmax=%ld W=%ld ngroups=%d kappa_max=%d dmax=%d real=%zuB\n",
            m, Kmax, W, ngroups, KAPPA_MAX, D_ABS_MAX, sizeof(real));

    stride = ((W + 1) + 15) / 16 * 16;
    size_t layer = (size_t)(Kmax + 1) * stride;
    cur = calloc(layer, sizeof(real));
    nxt = calloc(layer, sizeof(real));
    if (!cur || !nxt) { fprintf(stderr, "alloc failed\n"); return 1; }
    cur[0] = 1.0;
    long shift = 0;
    kmax_act_prev = 0; dmax_act_prev = 0;

    pthread_t th[256]; Job jobs[256];
    for (long j = 1; j <= m; j++) {
        kmax_act = (long)j * KAPPA_MAX < Kmax ? j * KAPPA_MAX : Kmax;
        dmax_act = (long)j * D_ABS_MAX < W ? j * D_ABS_MAX : W;
        long rows = kmax_act + 1;
        int nt = nthreads;
        for (int t = 0; t < nt; t++) {
            jobs[t].k0 = rows * t / nt;
            jobs[t].k1 = rows * (t + 1) / nt;
            if (pthread_create(&th[t], NULL, worker, &jobs[t]) != 0) {
                fprintf(stderr, "pthread_create failed\n"); exit(1);
            }
        }
        real gmax = 0;
        for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); if (jobs[t].blockmax > gmax) gmax = jobs[t].blockmax; }
        if (gmax > 0x1p64) {           /* rescale by exact power of two */
            for (long k = 0; k <= kmax_act; k++) {
                real *row = nxt + k * stride;
                for (long D = 0; D <= dmax_act; D++) row[D] *= 0x1p-64;
            }
            shift += 64;
        }
        real *tmp = cur; cur = nxt; nxt = tmp;
        kmax_act_prev = kmax_act; dmax_act_prev = dmax_act;
        if (j % 32 == 0 || j == m) fprintf(stderr, "[tdp] pos %ld/%ld shift=%ld\n", j, m, shift);
    }

    /* G[t] = sum over k + D = t (t <= Kmax) of (D==0 ? H : 2H), rounded down */
    double *G = calloc(Kmax + 1, sizeof(double));
    fesetround(FE_DOWNWARD);
    for (long k = 0; k <= kmax_act_prev && k <= Kmax; k++) {
        const real *row = cur + k * stride;
        long dtop = Kmax - k < dmax_act_prev ? Kmax - k : dmax_act_prev;
        G[k] += (double)row[0];
        for (long D = 1; D <= dtop; D++) G[k + D] += 2.0 * (double)row[D];
    }
    FILE *f = fopen(outfn, "wb");
    int64_t hdr[2] = { shift, Kmax + 1 };
    fwrite(hdr, sizeof(int64_t), 2, f);
    fwrite(G, sizeof(double), Kmax + 1, f);
    fclose(f);
    double tot = 0; for (long t = 0; t <= Kmax; t++) tot += G[t];
    fprintf(stderr, "[tdp] done. shift=%ld  log2(T(Kmax)) ~= %.3f\n",
            shift, log2(tot) + (double)shift);
    return 0;
}
