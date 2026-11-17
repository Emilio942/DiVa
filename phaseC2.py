"""Phase C2: two-pass edge localization -> precise eta_lo(sigma, lam_W, p_fix) surface."""
import json, time, math
import torch
from diva import Swarm, gs_orth, haar_Q

@torch.no_grad()
def two_pass_edges(items, N=64, device="cuda", dtype=torch.float32, seed=5,
                   T1_steps=110, T1_pts=26, span1=(1e-4, 20.0),
                   T2=6000, burn2=2500, meas2=350, n2=15, span2=(1/4.0, 2.5)):
    """
    Pass 1: batched adiabatic sweep (up, from disordered) over span1 -> rough edge per item.
    Pass 2: fixed-eta local steady-state grid around the rough edge -> precise edge.
    items: dicts with k,d,sigma,lam,pfix.
    """
    dev = device
    torch.manual_seed(seed)
    P = len(items)
    # ---- pass 1 (batched over all items, per geometry)
    from collections import defaultdict
    geo = defaultdict(list)
    for ii, it in enumerate(items):
        geo[(it["k"], it["d"])].append(ii)
    rough = {}
    sched1 = [math.exp(math.log(span1[0]) + (math.log(span1[1])-math.log(span1[0]))*i/(T1_pts-1))
              for i in range(T1_pts)]
    for (k, d), idxs in geo.items():
        sigs = torch.tensor([items[i]["sigma"] for i in idxs], device=dev)
        lams = torch.tensor([items[i]["lam"] for i in idxs], device=dev)
        pfxs = torch.tensor([items[i]["pfix"] for i in idxs], device=dev)
        sw = Swarm(len(idxs), N, d, k, dev, dtype)
        sw.set_params(sigma=sigs, lam_W=lams, p_fix=pfxs)
        # settle disordered NESS at tiny eta
        et0 = torch.full((len(idxs),), 1e-4, device=dev, dtype=dtype)
        for _ in range(800):
            sw.step(et0)
        Ds = torch.empty(len(idxs), T1_pts)
        for li, e in enumerate(sched1):
            et = torch.full((len(idxs),), e, device=dev, dtype=dtype)
            for _ in range(T1_steps):
                sw.step(et)
            Ds[:, li] = sw.Delta_chord()
        for jj, i in enumerate(idxs):
            rough[i] = [float(x) for x in Ds[jj]]
    # rough edge from chordal curve: crossing below 0.35*plateau
    rough_edge = {}
    for i, curve in rough.items():
        thr = 0.35 * curve[0]
        e = None
        for li in range(T1_pts-1):
            if curve[li] > thr >= curve[li+1]:
                f = (math.log(curve[li])-math.log(thr))/(math.log(curve[li])-math.log(curve[li+1]))
                e = math.exp(math.log(sched1[li]) + f*(math.log(sched1[li+1])-math.log(sched1[li])))
                break
        rough_edge[i] = e

    # ---- pass 2 (per geometry, local grids)
    precise = {}
    for (k, d), idxs in geo.items():
        jobs = []
        for i in idxs:
            base = rough_edge[i] if rough_edge[i] else 0.1
            for j in range(n2):
                f = math.exp(math.log(span2[0]) + (math.log(span2[1])-math.log(span2[0]))*j/(n2-1))
                jobs.append((i, base*f))
        sigs = torch.tensor([items[i]["sigma"] for i, _ in jobs], device=dev)
        lams = torch.tensor([items[i]["lam"] for i, _ in jobs], device=dev)
        pfxs = torch.tensor([items[i]["pfix"] for i, _ in jobs], device=dev)
        etas = torch.tensor([e for _, e in jobs], device=dev)
        sw = Swarm(len(jobs), N, d, k, dev, dtype)
        sw.set_params(sigma=sigs, lam_W=lams, p_fix=pfxs)
        Ds, Zs, Rs = [], [], []
        for t in range(T2):
            sw.step(etas)
            if t >= burn2 and (t - burn2) % meas2 == 0:
                Ds.append(sw.Delta())
                zr, zc, kv = sw.hoeffding()
                Zs.append(zc); Rs.append(sw.R_order())
        D = torch.stack(Ds).mean(0); Dsd = torch.stack(Ds).std(0)
        Z = torch.stack(Zs).mean(0); R = torch.stack(Rs).mean(0)
        for (i, e), Dv, Dsv, Zv, Rv in zip(jobs, D, Dsd, Z, R):
            precise.setdefault(i, []).append((e, float(Dv), float(Dsv), float(Zv), float(Rv)))
    # extract precise edges per item: threshold = midpoint between disordered (max D at low eta end)
    # and ordered floor (min D); works even with anchor-deformed floors.
    out = {}
    for i, lst in precise.items():
        lst.sort()
        Ds_ = [x[1] for x in lst]
        Dhi, Dlo = max(Ds_), min(Ds_)
        thr = 0.5*(Dhi + Dlo)
        edge, z1e = None, None
        for j in range(len(lst)-1):
            e0, D0, _, Z0, _ = lst[j]
            e1, D1, _, Z1, _ = lst[j+1]
            if D0 >= thr > D1:
                f = (math.log(D0)-math.log(thr))/(math.log(D0)-math.log(D1))
                edge = math.exp(math.log(e0)+f*(math.log(e1)-math.log(e0)))
                z1e = 0.5*(Z0+Z1)
                break
        out[i] = {"edge": edge, "Dhi": Dhi, "Dlo": Dlo, "z1_at_edge": z1e,
                  "curve": lst, "rough": rough_edge.get(i), "rough_curve": rough.get(i)}
    return out


def main():
    dev = "cuda"
    t00 = time.time()
    items = []
    tags = []
    combos = [(0.025,0.5),(0.05,0.5),(0.1,0.5),(0.2,0.5),(0.4,0.5),
              (0.05,1.0),(0.1,1.0),(0.2,1.0),(0.4,1.0),
              (0.1,0.25),(0.2,0.25)]
    for (k, d) in [(2,8),(1,2)]:
        for s in (0.05, 0.1):
            for (l, p) in combos:
                items.append(dict(k=k, d=d, sigma=s, lam=l, pfix=p))
                tags.append((k, d, s, l, p))
    res = two_pass_edges(items, N=64, device=dev)
    out = {}
    print(f"{'(k,d)':8s} {'s':>5s} {'lam':>6s} {'p':>5s} | {'edge':>9s} {'Dhi':>7s} {'Dlo':>7s} {'z1*':>8s}")
    for (t, i) in zip(tags, sorted(res.keys())):
        r = res[i]
        k, d, s, l, p = t
        out[str(t)] = r
        e = r["edge"]
        print(f"({k},{d})   {s:5.2f} {l:6.3f} {p:5.2f} | " +
              f"{'None':>9s}" if e is None else
              f"({k},{d})   {s:5.2f} {l:6.3f} {p:5.2f} | {e:9.5f} {r['Dhi']:7.3f} {r['Dlo']:7.3f} {r['z1_at_edge']:8.4f}")
    json.dump(out, open("results/phaseC2_anchor_surface.json", "w"))
    print(f"total {time.time()-t00:6.1f}s")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
