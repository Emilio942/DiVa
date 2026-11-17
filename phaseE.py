"""Phase E: transverse Lyapunov exponent of the anchored ordered state (Benettin pairs,
shared noise & anchor draws) -> sharp eta_hi(sigma, lam, p); plus conditional zeta1."""
import json, math, time
import torch
from diva import Swarm, gs_orth, haar_Q, proj_tan

@torch.no_grad()
def lyap_curve(k, d, sigma, lams, pfixs, eta_grid, N=64, T=4000, tau=20,
               delta=1e-5, device="cuda", dtype=torch.float32, seed=17):
    """Batch: elements = (lam,p) x eta_grid. Reference swarm driven to anchored
    ordered state; Benettin pair with shared randomness; returns top Lyapunov exponents."""
    dev = device
    torch.manual_seed(seed)
    Bc = len(lams)
    Be = len(eta_grid)
    B = Bc * Be
    lams_b = torch.tensor([l for l in lams for _ in range(Be)], device=dev, dtype=dtype)
    pfxs_b = torch.tensor([p for p in pfixs for _ in range(Be)], device=dev, dtype=dtype)
    sig_b = torch.full((B,), sigma, device=dev, dtype=dtype)
    etas_b = torch.tensor([e for _ in lams for e in eta_grid], device=dev, dtype=dtype)
    sw = Swarm(B, N, d, k, dev, dtype)
    sw.set_params(sigma=sig_b, lam_W=lams_b, p_fix=pfxs_b)
    # ordered init
    base = haar_Q((B, 1, d, k), dev, dtype)
    sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
    # settle to anchored ordered steady state
    for _ in range(2500):
        sw.step(etas_b)
    # perturbed clone
    sw2 = Swarm(B, N, d, k, dev, dtype)
    sw2.W_fix = sw.W_fix.clone()
    sw2.set_params(sigma=sig_b, lam_W=lams_b, p_fix=pfxs_b)
    Q3 = sw.Q.reshape(B*N, d, k)
    g = torch.Generator(device=dev); g.manual_seed(seed+1)
    z = torch.randn(B*N, d, k, device=dev, dtype=dtype, generator=g)
    z = proj_tan(Q3, z)
    z = z / z.norm(dim=(-2,-1), keepdim=True)
    sw2.Q = gs_orth(Q3 + delta*z).reshape(B, N, d, k)
    logsum = torch.zeros(B, device=dev, dtype=dtype)
    n_samples = 0
    for t in range(T):
        G = torch.randn(B*N, d, k, device=dev, dtype=dtype)
        u = torch.rand(B, N, 1, 1, device=dev, dtype=dtype)
        fresh = None
        if torch.any(sw.lam_W != 0):
            fresh = haar_Q((B, N, d, k), dev, dtype)
        sw.step(etas_b, noise_G=G, anchor_u=u, anchor_fresh=fresh)
        sw2.step(etas_b, noise_G=G, anchor_u=u, anchor_fresh=fresh)
        if (t+1) % tau == 0:
            diff = (sw2.Q - sw.Q).reshape(B, N, d, k)
            # project out the Goldstone mode (common tangent shift of the whole swarm)
            mean_diff = diff.mean(dim=1, keepdim=True)
            diff = diff - mean_diff
            diff3 = diff.reshape(B*N, d, k)
            rho = diff3.norm(dim=(-2,-1))
            logsum += torch.log((rho/delta).clamp_min(1e-30)).reshape(B, N).sum(-1) / N
            n_samples += 1
            U = diff3 / rho.clamp_min(1e-30).view(-1, 1, 1)
            sw2.Q = gs_orth(sw.Q.reshape(B*N, d, k) + delta*U).reshape(B, N, d, k)
    lyap = logsum / (n_samples * tau)
    return {"eta_grid": eta_grid, "lams": lams, "pfixs": pfixs,
            "lyap": lyap.reshape(Bc, Be).cpu(), "sigma": sigma}

def main():
    dev = "cuda"
    t0 = time.time()
    out = {}
    for (k, d) in [(2,8),(1,2)]:
        for s in (0.05, 0.1):
            lams  = [0.0, 0.05, 0.1, 0.2, 0.4]
            pfixs = [0.5]*len(lams)
            etas = [math.exp(math.log(0.02)+(math.log(6.0)-math.log(0.02))*i/23) for i in range(24)]
            res = lyap_curve(k, d, s, lams, pfixs, etas, N=64, T=4000, tau=20, device=dev)
            key = f"{k},{d}_s={s}_p=0.5"
            out[key] = res
            print(f"== {key}  ({time.time()-t0:6.1f}s)")
            for j, l in enumerate(lams):
                row = [f"{x:+8.4f}" for x in res["lyap"][j]]
                print(f"  lam={l:4.2f}: " + " ".join(row))
            print("  eta grid :", " ".join(f"{e:8.3f}" for e in etas))
    json.dump(out, open("results/phaseE_lyapunov.json", "w"))
    print("saved.")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
