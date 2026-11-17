"""Phase B3+C: a_sigma(k,d) scaling and anchor surface eta_lo(sigma, lam_W, p_fix)."""
import json, time, math
import torch
from diva import Swarm

@torch.no_grad()
def edge_scan(items, N=64, T=6000, burn=2500, meas_every=350, span=(1/3.5, 3.5),
              n_eta=14, device="cuda", dtype=torch.float32, seed=5):
    """
    items: list of dicts {k,d,sigma,lam,pfix,asig,base}
    For each item, eta grid = est * geo_span; batch all (item,eta) pairs.
    Returns per-item Delta(eta) curves (mean over replicates within batch).
    """
    dev = device
    torch.manual_seed(seed)
    jobs = []   # (item_idx, eta)
    for ii, it in enumerate(items):
        est = max(it.get("base", 1e-3), it.get("asig", 10.0)*it["sigma"]**2
                  + it.get("aW", 0.0)*it["lam"]**2*it["pfix"])
        for i in range(n_eta):
            f = math.exp(math.log(span[0]) + (math.log(span[1])-math.log(span[0]))*i/(n_eta-1))
            jobs.append((ii, est*f))
    B = len(jobs)
    # group by (k,d) to build swarms of matching geometry? -- need one Swarm per geometry.
    from collections import defaultdict
    geo_jobs = defaultdict(list)
    for ji, (ii, e) in enumerate(jobs):
        it = items[ii]
        geo_jobs[(it["k"], it["d"])].append((ji, ii, e))
    results = {}
    for (k, d), lst in geo_jobs.items():
        Qs = torch.tensor([[it["sigma"], it["lam"], it["pfix"]] for _, ii, _ in lst for it in [items[ii]]])
        sigs = torch.tensor([items[ii]["sigma"] for _, ii, _ in lst], device=dev)
        lams = torch.tensor([items[ii]["lam"] for _, ii, _ in lst], device=dev)
        pfxs = torch.tensor([items[ii]["pfix"] for _, ii, _ in lst], device=dev)
        etas = torch.tensor([e for _, _, e in lst], device=dev)
        sw = Swarm(len(lst), N, d, k, dev, dtype)
        sw.set_params(sigma=sigs, lam_W=lams, p_fix=pfxs)
        Ds, Zs = [], []
        for t in range(T):
            sw.step(etas)
            if t >= burn and (t - burn) % meas_every == 0:
                Ds.append(sw.Delta())
                zr, zc, kv = sw.hoeffding()
                Zs.append(zc)
        D = torch.stack(Ds).mean(0); Dsd = torch.stack(Ds).std(0)
        Z = torch.stack(Zs).mean(0)
        for (ji, ii, e), Dv, Dsv, Zv in zip(lst, D, Dsd, Z):
            results[ji] = (ii, e, float(Dv), float(Dsv), float(Zv))
    return jobs, results

def extract_edges(jobs, results, thr_frac=0.5):
    import collections
    per_item = collections.defaultdict(list)
    for ji, (ii, e, D, Dsd, Z) in results.items():
        per_item[ii].append((e, D, Z))
    out = {}
    for ii, lst in per_item.items():
        lst.sort()
        Ddis = lst[0][1]
        thr = thr_frac * Ddis
        edge, z1_at_edge = None, None
        for i in range(len(lst)-1):
            e0, D0, Z0 = lst[i]; e1, D1, Z1 = lst[i+1]
            if D0 > thr >= D1:
                f = (math.log(D0)-math.log(thr))/(math.log(D0)-math.log(D1))
                edge = math.exp(math.log(e0)+f*(math.log(e1)-math.log(e0)))
                z1_at_edge = 0.5*(Z0+Z1)
                break
        out[ii] = {"edge": edge, "Ddis": Ddis, "z1_at_edge": z1_at_edge,
                   "curve": [(e, D, Z) for e, D, Z in lst]}
    return out

def main():
    dev = "cuda"
    t00 = time.time()

    # ---- Phase B3: a_sigma(k,d) at sigma in {0.05, 0.1}
    cfgs = [(1,2),(1,3),(1,4),(1,8),(2,4),(2,8),(3,6),(4,8)]
    asig_guess = {(1,2):7.0,(1,3):9.0,(1,4):12.0,(1,8):25.0,(2,4):15.0,(2,8):25.7,(3,6):20.0,(4,8):35.0}
    items = []
    for (k,d) in cfgs:
        for s in (0.05, 0.1):
            items.append(dict(k=k, d=d, sigma=s, lam=0.0, pfix=0.0, asig=asig_guess[(k,d)], base=1e-3))
    jobs, results = edge_scan(items, T=6000, burn=2500, device=dev)
    edges = extract_edges(jobs, results)
    print("== a_sigma(k,d) at sigma=0.05, 0.1 ==")
    a_out = {}
    for ii, it in enumerate(items):
        e = edges[ii]["edge"]
        a = (e / it["sigma"]**2) if e else None
        key = f"{it['k']},{it['d']}"
        a_out.setdefault(key, {})[str(it["sigma"])] = {"edge": e, "a_from_edge": a,
                                                       "z1": edges[ii]["z1_at_edge"]}
        print(f"  (k={it['k']},d={it['d']}) sigma={it['sigma']}: edge={e if e is None else round(e,5)}  a=e/s2={None if a is None else round(a,3)}  z1*={None if edges[ii]['z1_at_edge'] is None else round(edges[ii]['z1_at_edge'],4)}")
    json.dump({"a_sigma": a_out}, open("results/phaseB3_asigma_kd.json", "w"))

    # ---- Phase C: anchors at (2,8) and (1,2), sigma in {0.05,0.1}
    items2 = []
    combos = [(0.05,0.5),(0.1,0.5),(0.2,0.5),(0.4,0.5),(0.1,1.0),(0.2,1.0),(0.4,1.0),(0.2,0.25)]
    for (k,d) in [(2,8),(1,2)]:
        for s in (0.05, 0.1):
            for (l, p) in combos:
                items2.append(dict(k=k, d=d, sigma=s, lam=l, pfix=p, asig=asig_guess[(k,d)],
                                   aW=6.0, base=1e-3))
    jobs2, results2 = edge_scan(items2, T=6000, burn=2500, device=dev)
    edges2 = extract_edges(jobs2, results2)
    print("== anchor surface ==")
    c_out = {}
    for ii, it in enumerate(items2):
        e = edges2[ii]["edge"]
        key = f"{it['k']},{it['d']}"
        c_out.setdefault(key, {}).setdefault(f"s={it['sigma']}", {})[f"l={it['lam']},p={it['pfix']}"] = {
            "edge": e, "z1": edges2[ii]["z1_at_edge"],
            "curve": edges2[ii]["curve"]}
        print(f"  (k={it['k']},d={it['d']}) s={it['sigma']} l={it['lam']} p={it['pfix']}: edge={None if e is None else round(e,5)} z1*={None if edges2[ii]['z1_at_edge'] is None else round(edges2[ii]['z1_at_edge'],4)}")
    json.dump(c_out, open("results/phaseC_anchors.json", "w"))
    print(f"total: {time.time()-t00:6.1f}s")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
