"""Phase B: lower & upper edges vs sigma (NESS at sigma>0), exact Delta + zeta1 at every point."""
import json, time
import torch
from diva import adiabatic_sweep, geo_grid

def main():
    dev, N, P = "cuda", 64, 8
    g_up = geo_grid(2e-4, 12.0, 72)          # lower-edge sweep (disordered NESS at start)
    out = {"N": N, "sweeps": {}}
    for (k, d) in [(2, 8), (1, 2)]:
        sigmas = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
        t0 = time.time()
        res = adiabatic_sweep(sigmas, [0.0]*len(sigmas), [0.0]*len(sigmas), g_up,
                              N=N, d=d, k=k, steps_per_eta=150, T_drift=100,
                              init="haar", device=dev, seed=11,
                              pre_eta=1e-4, pre_steps=800,
                              exact_at=list(range(len(g_up))))
        out["sweeps"][f"{k},{d}_lower"] = {
            "sigma": sigmas, "eta": res["eta_sched"],
            "Delta": res["exact"][0][0].tolist(),  # placeholder overwritten below
        }
        # collect exact Delta at every point: exact[li] = (Delta, zeta_raw, zeta_corr, kv)
        Ds, Zr, Zc, Kv = [], [], [], []
        for li in sorted(res["exact"].keys()):
            eD, zr, zc, kv = res["exact"][li]
            Ds.append(eD.tolist()); Zr.append(zr.tolist()); Zc.append(zc.tolist()); Kv.append(kv.tolist())
        out["sweeps"][f"{k},{d}_lower"] = {
            "sigma": sigmas, "eta": res["eta_sched"],
            "Delta": Ds, "zeta_raw": Zr, "zeta_corr": Zc, "kvar": Kv,
            "Dchord": res["Delta_chord"].tolist(), "R": res["R"].tolist(),
        }
        print(f"(k={k},d={d}) lower sweep: {time.time()-t0:6.1f}s")
    json.dump(out, open("results/phaseB_sigma.json", "w"))
    print("saved results/phaseB_sigma.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
