"""Verification pass V1+V2: threshold & T robustness of edges, replicate error bars.
For (2,8) and (1,2) at sigma in {0.05, 0.1}:
  - dense local eta grid, T in {4000, 12000}, P=12 replicates (batched)
  - edges at thresholds 0.3/0.5/0.7 of plateau
  - report a_sigma = edge/sigma^2 with std across replicates.
"""
import json, math, time
import torch
from diva import Swarm

@torch.no_grad()
def edge_multithresh(k, d, sigma, asig, N=64, T=8000, burn=3000, meas=400, n_eta=16,
                     P=12, span=(1/2.8, 2.8), device="cuda", dtype=torch.float32, seed=101):
    dev = device
    torch.manual_seed(seed)
    est = asig*sigma*sigma
    lo, hi = est/span[1], est*span[1]
    etas = [math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/(n_eta-1)) for i in range(n_eta)]
    # batch = (eta, replicate) with elementwise sigma
    jobs = [(e, r) for e in etas for r in range(P)]
    etas_b = torch.tensor([e for e, _ in jobs], device=dev, dtype=dtype)
    sw = Swarm(len(jobs), N, d, k, dev, dtype)
    sw.set_params(sigma=sigma)
    Ds = []   # each: (n_jobs,)
    for t in range(T):
        sw.step(etas_b)
        if t >= burn and (t-burn) % meas == 0:
            Ds.append(sw.Delta())
    Dm = torch.stack(Ds).mean(0)            # time-mean per job
    Dsd = torch.stack(Ds).std(0)
    # reshape: (n_eta, P)
    Dm = Dm.reshape(n_eta, P); Dsd = Dsd.reshape(n_eta, P)
    out = {}
    for thrfrac in (0.3, 0.5, 0.7):
        edges = []
        for r in range(P):
            dd = Dm[:, r]; plateau = dd[0]
            thr = thrfrac*plateau
            e = None
            for i in range(n_eta-1):
                if dd[i] > thr >= dd[i+1]:
                    lg = lambda x: math.log(max(float(x), 1e-6))
                    f = (lg(dd[i])-lg(thr))/(lg(dd[i])-lg(dd[i+1]))
                    e = math.exp(math.log(etas[i])+f*(math.log(etas[i+1])-math.log(etas[i])))
                    break
            edges.append(e)
        vals = [e for e in edges if e is not None]
        out[thrfrac] = {
            "mean": sum(vals)/len(vals) if vals else None,
            "std": (sum((x-sum(vals)/len(vals))**2 for x in vals)/(len(vals)-1))**0.5 if len(vals) > 1 else None,
            "n_ok": len(vals), "edges": edges,
        }
    return {"sigma": sigma, "etas": etas, "D_mean": Dm.mean(1).tolist(),
            "D_std_across_rep": Dsd.mean(1).tolist(), "edges_by_thr": out}

def main():
    dev = "cuda"
    out = {}
    for (k, d, asig) in [(2, 8, 25.7), (1, 2, 7.0)]:
        for s in (0.05, 0.1):
            for T in (4000, 12000):
                t0 = time.time()
                r = edge_multithresh(k, d, s, asig, T=T, P=12, device=dev)
                key = f"{k},{d}_s={s}_T={T}"
                out[key] = r
                line = f"{key}: "
                for thr, v in r["edges_by_thr"].items():
                    m, sd = v["mean"], v["std"]
                    a = m/s**2 if m else None
                    line += f"thr={thr}: edge={m and round(m,5)}±{sd and round(sd,5)} (a={a and round(a,2)})  "
                print(line + f" [{time.time()-t0:5.1f}s]")
    json.dump(out, open("results/V1_threshold_T_robustness.json", "w"))
    print("saved results/V1_threshold_T_robustness.json")

if __name__ == "__main__":
    main()
