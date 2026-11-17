"""Phase B2: precision lower-edge eta_lo(sigma) via fixed-eta steady states."""
import json, time, math
import torch
from diva import Swarm

def steady_eta_grid(sigmas, N=64, d=8, k=2, T=8000, burn=3000, meas_every=400,
                    device="cuda", dtype=torch.float32, seed=3, base=None, asig=22.0,
                    span=(1/3.0, 3.0), n_eta=16):
    """For each sigma, sweep eta over a local grid; batch ALL (sigma,eta) pairs."""
    dev = device
    torch.manual_seed(seed)
    pairs = []   # (sigma, eta)
    for s in sigmas:
        est = max(base, asig * s * s)
        for i in range(n_eta):
            f = math.exp(math.log(span[0]) + (math.log(span[1])-math.log(span[0]))*i/(n_eta-1))
            pairs.append((s, est*f))
    P = len(pairs)
    sigs = torch.tensor([p[0] for p in pairs], device=dev)
    etas = torch.tensor([p[1] for p in pairs], device=dev)
    sw = Swarm(P, N, d, k, dev, dtype)
    sw.set_params(sigma=sigs)
    Ds, Rs, Zc, Zr = [], [], [], []
    for t in range(T):
        sw.step(etas)
        if t >= burn and (t - burn) % meas_every == 0:
            Ds.append(sw.Delta()); Rs.append(sw.R_order())
            zr, zc, kv = sw.hoeffding()
            Zc.append(zc); Zr.append(zr)
    D = torch.stack(Ds).mean(0); Dstd = torch.stack(Ds).std(0)
    R = torch.stack(Rs).mean(0)
    Z = torch.stack(Zc).mean(0)
    return {
        "pairs": pairs, "sigma": sigs.cpu(), "eta": etas.cpu(),
        "Delta": D.cpu(), "Delta_std": Dstd.cpu(), "R": R.cpu(), "zeta1": Z.cpu(),
    }

def extract_edge(res, frac=0.5):
    """per sigma: eta where Delta crosses frac * Delta_dis (linear in log eta)."""
    import collections
    by_sigma = collections.defaultdict(list)
    for (s, e), D, R, Z in zip(res["pairs"], res["Delta"], res["R"], res["zeta1"]):
        by_sigma[float(s)].append((float(e), float(D), float(R), float(Z)))
    out = {}
    for s, lst in by_sigma.items():
        lst.sort()
        Ddis = lst[0][1]           # disordered plateau = lowest eta value
        thr = frac * Ddis
        edge = None
        for i in range(len(lst)-1):
            e0, D0, _, _ = lst[i]; e1, D1, _, _ = lst[i+1]
            if D0 > thr >= D1:     # linear interp in log eta
                f = (math.log(D0) - math.log(thr)) / (math.log(D0) - math.log(D1))
                edge = math.exp(math.log(e0) + f*(math.log(e1)-math.log(e0)))
                break
        out[s] = {"edge": edge, "Ddis": Ddis, "curve": lst}
    return out

def main():
    dev = "cuda"
    all_out = {}
    for (k, d), base, sigmas, asig in [
        ((2, 8), 0.006, [0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.28], 22.0),
        ((1, 2), 0.004, [0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.4], 6.5),
    ]:
        t0 = time.time()
        res = steady_eta_grid(sigmas, N=64, d=d, k=k, T=8000, burn=3000,
                              device=dev, base=base, asig=asig)
        edges = extract_edge(res)
        all_out[f"{k},{d}"] = {
            "sigma": [float(s) for s in res["sigma"]],
            "eta": [float(e) for e in res["eta"]],
            "Delta": [float(x) for x in res["Delta"]],
            "Delta_std": [float(x) for x in res["Delta_std"]],
            "R": [float(x) for x in res["R"]],
            "zeta1": [float(x) for x in res["zeta1"]],
            "edges": {str(s): v["edge"] for s, v in edges.items()},
        }
        print(f"(k={k},d={d}) {time.time()-t0:6.1f}s  edges:",
              {s: (round(v["edge"],5) if v["edge"] else None) for s, v in edges.items()})
    json.dump(all_out, open("results/phaseB2_edges_sigma.json", "w"))
    print("saved results/phaseB2_edges_sigma.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
