"""ESTIMATED: position-0 acceptance under live-like rounds vs the g1.py proxy (window 5 committed trajectory).
Two-regime Markov token model; per-position draft-match probability a[regime(t+j)] * decay**j.
Live: window w = clamp(ceil(EMA+1.0), 2, cap=4) (AK_MIN 2, margin 1.0, 3 streams -> cap 4), conf depth policy
picks depth 0 when predicted c0 <= 0.0925. Proxy: window 5, depth 5, from the same committed trajectory."""
import random, math, sys
def run(pe, pstay_e, pstay_h, ae, ah, decay, conf_noise, n=400000, seed=1):
    rng = random.Random(seed)
    reg = []; r = 0
    for _ in range(n + 10):
        reg.append(r)
        stay = pstay_e if r == 0 else pstay_h
        if rng.random() > stay: r = 1 - r
    # match indicator of draft j at round-start t: depends on regime at t+j (independent draw per (t,j))
    def acc(t, j): 
        a = (ae if reg[t + j] == 0 else ah) * decay ** j
        return a
    # precompute a deterministic "truth": draw uniform per (t,j) so proxy and live see the same committed trajectory
    U = [[rng.random() for _ in range(5)] for _ in range(n + 10)]
    def ml(t, w):
        k = 0
        while k < w and U[t][k] < acc(t, k): k += 1
        return k
    # proxy
    t = 0; R = 0; H = 0; A = 0
    while t < n:
        m = ml(t, 5); R += 1; H += m > 0; A += m; t += m + 1
    proxy = (H / R, A / R)
    # live: adaptive window, depth-0 censoring
    t = 0; R = 0; H = 0; A = 0; ema = None; d0 = 0
    while t < n:
        w = 4 if ema is None else max(2, min(4, math.ceil(ema + 1.0)))
        c0 = min(1, max(0, acc(t, 0) + rng.gauss(0, conf_noise)))
        depth = 0 if c0 <= 0.0925 else w
        m = ml(t, min(w, depth)); R += 1; H += m > 0; A += m; d0 += depth == 0
        x = m + (1 if m >= w else 0); ema = x if ema is None else 0.7 * ema + 0.3 * x
        t += m + 1
    live = (H / R, A / R, d0 / R)
    return proxy, live
print('pe/stay_e/stay_h/ae/ah/decay/noise | proxy p0, apr | live p0, apr, frac_d0 | live-proxy p0')
for args in [(0, .5, .5, .70, .70, .97, .05), (0, .9, .8, .9, .35, .97, .1), (0, .95, .9, .92, .3, .97, .1),
             (0, .98, .95, .95, .25, .97, .1), (0, .97, .9, .9, .05, .97, .05), (0, .99, .97, .95, .2, .95, .15)]:
    p, l = run(*args)
    print(args[1:], '| %.4f %.3f | %.4f %.3f %.3f | %+.4f' % (p[0], p[1], l[0], l[1], l[2], l[0] - p[0]))
