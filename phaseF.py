"""Phase F: (1) conditional zeta1 test, (2) two-init bistability scan (transition
character), (3) extra (k,d) points for the a_sigma ~ 3d+k law."""
import json, math, time
import torch
from diva import Swarm, gs_orth, haar_Q

# ---------- (1) conditional zeta1 ----------
@torch.no_grad()
def zeta_conditional(k, d, sigma, lam, pfix, eta, N=64, T=4000, burn=1500,
                     device="cuda", dtype=torch.float32, seed=31):
    dev = device
    torch.manual_seed(seed)
    sw = Swarm(4, N, d, k, dev, dtype)   # 4 replicates
    sw.set_params(sigma=sigma, lam_W=lam, p_fix=pfix)
    et = torch.full((4,), eta, device=dev, dtype=dtype)
    acc = torch.zeros(4, device=dev, dtype=dtype); accw = torch.zeros(4, device=dev, dtype=dtype)
    accz = torch.zeros(4, device=dev, dtype=dtype); kv_acc = torch.zeros(4, device=dev, dtype=dtype)
    n = 0
    for t in range(T):
        u = torch.rand(4, N, 1, 1, device=dev, dtype=dtype)
        fresh = haar_Q((4, N, d, k), dev, dtype) if lam > 0 else None
        sw.step(et, anchor_u=u, anchor_fresh=fresh)
        if t >= burn:
            h = sw.pairwise_geo()
            s = sw.last_anchor_indicator.squeeze(-1).squeeze(-1)   # (4,N)
            mask = (1.0 - torch.eye(N, device=dev, dtype=dtype))
            r = (h*mask).sum(-1)/(N-1)
            ztot = r.var(dim=-1, unbiased=False)
            # within-group variance (remove group means)
            rm = (r*s).sum(-1)/s.sum(-1).clamp_min(1)             # mean over fixed group
            rf = (r*(1-s)).sum(-1)/(1-s).sum(-1).clamp_min(1)
            r_within = r - s*rm.unsqueeze(-1) - (1-s)*rf.unsqueeze(-1)
            zwithin = r_within.var(dim=-1, unbiased=False)
            hmask = h*mask
            kh_mean = hmask.sum(dim=(-2,-1), keepdim=True)/(N*(N-1))
            kv = ((hmask-kh_mean)**2*mask).sum(dim=(-2,-1))/(N*(N-1))
            acc += ztot; accw += zwithin; kv_acc += kv
            accz += ((ztot - kv/(N-1)).clamp_min(0)/kv.clamp_min(1e-12))
            n += 1
    return {"sigma":sigma,"lam":lam,"pfix":pfix,"eta":eta,
            "z_total_raw":(acc/n).tolist(), "z_within_raw":(accw/n).tolist(),
            "kvar":(kv_acc/n).tolist(), "floor":(kv_acc/n/(N-1)).tolist(),
            "z_total_corr":(accz/n).tolist()}

# ---------- (2) two-init bistability ----------
@torch.no_grad()
def bistability(k, d, sigma, eta_grid, N=64, T=15000, burn=5000, meas=500,
                device="cuda", dtype=torch.float32, seed=41):
    dev = device
    torch.manual_seed(seed)
    B = len(eta_grid)
    etas = torch.tensor(eta_grid, device=dev, dtype=dtype)
    out = {}
    for init in ("dis", "ord"):
        sw = Swarm(B, N, d, k, dev, dtype)
        sw.set_params(sigma=sigma)
        if init == "ord":
            base = haar_Q((B,1,d,k), dev, dtype)
            sw.Q = base.unsqueeze(2).expand(B,1,N,d,k).reshape(B,N,d,k).contiguous()
        else:
            et0 = torch.full((B,), float(eta_grid[0]*0.3), device=dev, dtype=dtype)
            for _ in range(2000): sw.step(et0)
        Ds = []
        for t in range(T):
            sw.step(etas)
            if t >= burn and (t-burn) % meas == 0:
                Ds.append(sw.Delta())
        out[init] = torch.stack(Ds).mean(0).cpu()
    return {"eta": eta_grid, "dis": out["dis"].tolist(), "ord": out["ord"].tolist()}

# ---------- (3) extra (k,d) for a_sigma law ----------
@torch.no_grad()
def edge_at(k, d, sigma, asig_guess, N=64, T=6000, burn=2500, meas=350, n_eta=14,
            device="cuda", dtype=torch.float32, seed=5):
    dev = device
    torch.manual_seed(seed)
    est = asig_guess*sigma*sigma
    lo, hi = est/3.5, est*3.5
    etas = [math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/(n_eta-1)) for i in range(n_eta)]
    etas_t = torch.tensor(etas, device=dev, dtype=dtype)
    sw = Swarm(n_eta, N, d, k, dev, dtype)
    sw.set_params(sigma=sigma)
    Ds = []
    for t in range(T):
        sw.step(etas_t)
        if t >= burn and (t-burn) % meas == 0:
            Ds.append(sw.Delta())
    D = torch.stack(Ds).mean(0)
    Ddis = float(D[0]); thr = 0.5*Ddis
    edge = None
    for i in range(n_eta-1):
        if float(D[i]) > thr >= float(D[i+1]):
            lg = lambda x: math.log(max(x,1e-6))
            f = (lg(float(D[i]))-lg(thr))/(lg(float(D[i]))-lg(float(D[i+1])))
            edge = math.exp(math.log(etas[i])+f*(math.log(etas[i+1])-math.log(etas[i])))
            break
    return edge, D.tolist(), etas

def main():
    dev = "cuda"
    t00 = time.time()
    out = {}
    # (1) conditional zeta1
    print("== conditional zeta1 (ordered anchored states) ==")
    cz = []
    for (lam, p, eta) in [(0.05,0.5,0.3),(0.1,0.5,0.3),(0.1,1.0,0.3),(0.2,0.25,0.3),(0.2,0.5,0.3),(0.05,0.5,1.0),(0.1,0.5,1.0)]:
        r = zeta_conditional(2, 8, 0.05, lam, p, eta, device=dev)
        cz.append(r)
        print(f"  l={lam:4.2f} p={p:3.2f} eta={eta}: z_total={sum(r['z_total_raw'])/4:.2e} "
              f"z_within={sum(r['z_within_raw'])/4:.2e} floor={sum(r['floor'])/4:.2e} "
              f"z_corr_total={sum(r['z_total_corr'])/4:.4f}")
    out["conditional_zeta"] = cz
    # (2) bistability: matched eta_lo ~ 0.064: (2,8) sigma=0.05, (1,2) sigma=sqrt(0.064/7)
    print("== two-init bistability ==")
    s12 = math.sqrt(0.064/7.0)
    bis = {}
    for (k,d,s,label) in [(2,8,0.05,"2,8"),(1,2,s12,"1,2")]:
        g = [0.064*f for f in (0.5,0.7,0.85,1.0,1.2,1.5,2.0)]
        r = bistability(k, d, s, g, device=dev)
        bis[label] = {"sigma": s, **r}
        print(f"  ({k},{d}) sigma={s:.4f} eta={['%.4f'%e for e in g]}")
        print(f"     dis: {['%.3f'%x for x in r['dis']]}")
        print(f"     ord: {['%.3f'%x for x in r['ord']]}")
    out["bistability"] = bis
    # (3) extra (k,d)
    print("== extra (k,d) a_sigma ==")
    ex = {}
    for (k,d,guess) in [(1,5,16.0),(2,3,11.0),(3,9,30.0),(2,5,17.0)]:
        e, D, etas = edge_at(k, d, 0.05, guess, device=dev)
        a = e/0.0025 if e else None
        ex[f"{k},{d}"] = {"edge": e, "a_from_edge": a}
        print(f"  (k={k},d={d}): edge={None if e is None else round(e,5)} -> a={None if a is None else round(a,2)} (3d+k={3*d+k})")
    out["extra_kd"] = ex
    json.dump(out, open("results/phaseF.json", "w"))
    print(f"saved results/phaseF.json  ({time.time()-t00:6.1f}s)")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
