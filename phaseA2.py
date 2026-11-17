"""Phase A2b: hysteresis at sigma=0 with perturbation-seeded sync init + correct edges."""
import json, time, statistics
import torch
from diva import adiabatic_sweep, geo_grid

THR = 0.05

def cross_dn(eta, D, thr=THR):   # ordered(above) -> disordered(below) while scanning in list order
    for i in range(len(eta)-1):
        if D[i] > thr and D[i+1] < thr: return eta[i+1]

def cross_up(eta, D, thr=THR):
    for i in range(len(eta)-1):
        if D[i] < thr and D[i+1] > thr: return eta[i+1]

def main():
    dev, N, P = "cuda", 64, 8
    g = geo_grid(1e-4, 12.0, 60)
    out = {}
    for (k, d) in [(1, 2), (2, 4), (2, 8), (1, 4)]:
        out[f"{k},{d}"] = {}
        for name, kw in {
            "haar_up":   dict(init="haar", sched=g, pre=None),
            "sync_up":   dict(init="sync", sched=g, pre=None),
            "sync_down": dict(init="sync", sched=g[::-1], pre=None),
            "haar_down": dict(init="haar", sched=g[::-1], pre=5.0),
        }.items():
            t0 = time.time()
            res = adiabatic_sweep([0.0]*P, [0.0]*P, [0.0]*P, kw["sched"],
                                  N=N, d=d, k=k, steps_per_eta=150, T_drift=100,
                                  init=kw["init"], device=dev, seed=7,
                                  pre_eta=kw["pre"], pre_steps=400 if kw["pre"] else 0,
                                  perturb_init=1e-4)
            out[f"{k},{d}"][name] = {"eta": res["eta_sched"],
                                     "Dchord": res["Delta_chord"].tolist(),
                                     "R": res["R"].tolist()}
            print(f"(k={k},d={d}) {name}: {time.time()-t0:5.1f}s")
    json.dump({"N": N, "results": out}, open("results/phaseA2_hysteresis.json", "w"))
    med = lambda L: statistics.median([x for x in L if x is not None]) if any(x is not None for x in L) else float('nan')
    print(f"\n{'cfg':8s} {'lo_fwd(haar_up D down)':>22s} {'lo_bwd(sync_down D down)':>24s} {'hi_fwd(sync_up D up)':>22s} {'hi_bwd(haar_down D down)':>24s}")
    for cfg, sw in out.items():
        lo_f = [cross_dn(sw["haar_up"]["eta"], r) for r in sw["haar_up"]["Dchord"]]
        lo_b = [cross_dn(sw["sync_down"]["eta"], r) for r in sw["sync_down"]["Dchord"]]
        hi_f = [cross_up(sw["sync_up"]["eta"], r) for r in sw["sync_up"]["Dchord"]]
        hi_b = [cross_dn(sw["haar_down"]["eta"], r) for r in sw["haar_down"]["Dchord"]]
        print(f"{cfg:8s} {med(lo_f):22.4f} {med(lo_b):24.4f} {med(hi_f):22.4f} {med(hi_b):24.4f}")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
