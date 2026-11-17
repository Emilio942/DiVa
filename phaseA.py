"""Phase A: baseline Delta(eta) at (sigma, lam_W) = 0, several (k,d)."""
import json, math, sys, time
import torch
from diva import run_curve

def main():
    dev = "cuda"
    N = 64
    configs = [(2, 8), (1, 4), (1, 3), (2, 4), (1, 2)]
    etas = [round(x, 6) for x in torch.logspace(-3, 1, 26).tolist()]
    out = {}
    for (k, d) in configs:
        t0 = time.time()
        res = run_curve(etas, N=N, d=d, k=k, T=2500, burn=1800, meas_every=50,
                        sigma=0.0, lam_W=0.0, p_fix=0.0, replicates=3, seed=1, device=dev)
        out[f"{k},{d}"] = {
            "eta": res["eta"].tolist(),
            "Delta": res["Delta"].tolist(),
            "Delta_std": res["Delta_std"].tolist(),
            "R": res["R"].tolist(),
        }
        print(f"(k={k},d={d})  {time.time()-t0:6.1f}s")
        for e, dl, R in zip(res["eta"], res["Delta"], res["R"]):
            print(f"   eta={e:9.4f}  Delta={dl:8.4f} +- {0:6.4f}  R={R:7.4f}")
    with open("results/phaseA.json", "w") as f:
        json.dump({"N": N, "etas": etas, "configs": out}, f, indent=1)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
