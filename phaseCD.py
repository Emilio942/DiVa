"""Phase CD: dense (eta, lam, p) NESS map, up- and down-swept, Delta + R + zeta1.
At fixed sigma. Gives edge shift vs anchors AND hysteresis (transition character)."""
import json, time, math
import torch
from diva import Swarm, gs_orth, haar_Q, geo_grid

@torch.no_grad()
def map_surface(k, d, sigma, combos, etas, N=64, T=6000, burn=3000, meas=300,
                direction="up", device="cuda", dtype=torch.float32, seed=9,
                perturb=1e-4):
    dev = device
    torch.manual_seed(seed)
    B = len(combos)
    sigs = torch.tensor([sigma]*B, device=dev, dtype=dtype).reshape(-1)
    lams = torch.tensor([c[0] for c in combos], device=dev, dtype=dtype).reshape(-1)
    pfxs = torch.tensor([c[1] for c in combos], device=dev, dtype=dtype).reshape(-1)
    sw = Swarm(B, N, d, k, dev, dtype)
    sw.set_params(sigma=sigs, lam_W=lams, p_fix=pfxs)
    if direction == "up":
        et0 = torch.full((B,), float(etas[0]), device=dev, dtype=dtype)
        for _ in range(1200):
            sw.step(et0)
    else:
        base = haar_Q((B, 1, d, k), dev, dtype)
        sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
        Q3 = sw.Q.reshape(B*N, d, k)
        z = torch.randn(B*N, d, k, device=dev, dtype=dtype)
        z = z - torch.bmm(Q3, torch.bmm(Q3.transpose(1,2), z))
        z = z / z.norm(dim=(-2,-1), keepdim=True)
        sw.Q = gs_orth(Q3 + perturb*z).reshape(B, N, d, k)
        et0 = torch.full((B,), float(etas[-1]), device=dev, dtype=dtype)
        for _ in range(1200):
            sw.step(et0)
    order = list(etas) if direction == "up" else list(etas)[::-1]
    Ds, Rs, Zs = [], [], []
    for e in order:
        et = torch.full((B,), float(e), device=dev, dtype=dtype)
        Ds_, Zs_ = [], []
        for _ in range(6):
            for _ in range(meas):
                sw.step(et)
            Ds_.append(sw.Delta())
            zr, zc, kv = sw.hoeffding()
            Zs_.append(zc)
        Ds.append(torch.stack(Ds_).mean(0))
        Zs.append(torch.stack(Zs_).mean(0))
        Rs.append(sw.R_order())
    D = torch.stack(Ds); R = torch.stack(Rs); Z = torch.stack(Zs)
    if direction == "down":
        D, R, Z = D.flip(0), R.flip(0), Z.flip(0)
    return {"combos": combos, "etas": list(etas),
            "Delta": D.cpu(), "R": R.cpu(), "zeta1": Z.cpu()}


def main():
    dev = "cuda"
    t00 = time.time()
    etas = geo_grid(1e-3, 5.0, 20)
    combos = [(0.0,0.5),(0.05,0.5),(0.1,0.5),(0.2,0.5),(0.4,0.5),
              (0.1,1.0),(0.2,1.0),(0.4,1.0),(0.1,0.25),(0.2,0.25)]
    try:
        out = json.load(open("results/phaseCD_surface.json"))
    except Exception:
        out = {}
    for (k, d) in [(2,8),(1,2)]:
        for s in (0.05,):
            for direction in ("up","down"):
                key = f"{k},{d}_s={s}_{direction}"
                if key in out:
                    print(f"{key}: cached")
                    continue
                t0 = time.time()
                res = map_surface(k, d, s, combos, etas, direction=direction,
                                  T=6000, burn=3000, meas=300, device=dev)
                out[key] = {"combos": combos, "etas": res["etas"],
                            "Delta": res["Delta"].tolist(), "R": res["R"].tolist(),
                            "zeta1": res["zeta1"].tolist()}
                json.dump(out, open("results/phaseCD_surface.json", "w"))
                print(f"{key}: {time.time()-t0:6.1f}s (saved)")
    print(f"total {time.time()-t00:6.1f}s")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
