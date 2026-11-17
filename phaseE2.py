"""Phase E2: deterministic one-step Lyapunov exponent along the stochastic NESS.
lambda_det(eta, sigma, lam, p) = < log ||F_det(Q+delta*U) - F_det(Q)|| / delta >
with U evolved by the same finite-difference (no noise in the perturbation channel,
anchor draws shared between the two evaluations). Threshold-free stability boundary.
"""
import json, math, time
import torch
from diva import Swarm, gs_orth, haar_Q, proj_tan

@torch.no_grad()
def det_step_increment(sw, Q, W_draw):
    """deterministic part of one update for state Q (B,N,d,k), given anchor draws W_draw."""
    B, N, d, k = sw.B, sw.N, sw.d, sw.k
    Q3 = Q.reshape(B*N, d, k)
    Q2 = Q.transpose(1, 2).reshape(B, d, N * k)
    Sbar = (Q2 @ Q2.transpose(1, 2)) / N
    Sx = Sbar.unsqueeze(1).expand(B, N, d, d).reshape(B * N, d, d)
    drift = torch.bmm(Sx, Q3)
    drift = proj_tan(Q3, drift)
    upd = sw.eta4 * drift.reshape(B, N, d, k)
    if torch.any(sw.lam_W != 0):
        W3 = W_draw.reshape(B * N, d, k)
        a = proj_tan(Q3, torch.bmm(W3, torch.bmm(W3.transpose(1, 2), Q3)))
        upd = upd + sw.lam_W * a.reshape(B, N, d, k)
    return upd

@torch.no_grad()
def lyap_det(k, d, sigma, lam_p_list, eta_grid, N=64, T=3000, burn=1500,
             delta=1e-6, device="cuda", dtype=torch.float32, seed=23):
    dev = device
    torch.manual_seed(seed)
    Bc = len(lam_p_list)
    Be = len(eta_grid)
    B = Bc * Be
    sw = Swarm(B, N, d, k, dev, dtype)
    lams_b = torch.tensor([lp[0] for lp in lam_p_list for _ in range(Be)], device=dev, dtype=dtype)
    pfxs_b = torch.tensor([lp[1] for lp in lam_p_list for _ in range(Be)], device=dev, dtype=dtype)
    sig_b = torch.full((B,), sigma, device=dev, dtype=dtype)
    etas_b = torch.tensor([e for _ in lam_p_list for e in eta_grid], device=dev, dtype=dtype)
    sw.set_params(sigma=sig_b, lam_W=lams_b, p_fix=pfxs_b)
    sw.eta4 = etas_b.reshape(B, 1, 1, 1)
    # ordered init + settle into NESS
    base = haar_Q((B, 1, d, k), dev, dtype)
    sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
    for _ in range(burn):
        sw.step(etas_b)
    # unit perturbation directions (tangent, per agent)
    g = torch.Generator(device=dev); g.manual_seed(seed + 1)
    z = torch.randn(B, N, d, k, device=dev, dtype=dtype, generator=g)
    Qr = sw.Q.reshape(B*N, d, k)
    zz = proj_tan(Qr, z.reshape(B*N, d, k))
    U = zz / zz.norm(dim=(-2,-1), keepdim=True)
    logsum = torch.zeros(B, device=dev, dtype=dtype)
    n = 0
    for t in range(burn, burn + T):
        # draw anchors once (shared by both evaluations); noise only for reference walk
        G = torch.randn(B*N, d, k, device=dev, dtype=dtype)
        u = torch.rand(B, N, 1, 1, device=dev, dtype=dtype)
        fresh = haar_Q((B, N, d, k), dev, dtype) if torch.any(sw.lam_W != 0) else None
        # reference stochastic step (updates sw.Q)
        sw.step(etas_b, noise_G=G, anchor_u=u, anchor_fresh=fresh)
        # deterministic finite difference along the *new* reference state
        Qn = sw.Q
        W_draw, _ = sw.anchors(u=u, fresh=fresh)
        inc1 = det_step_increment(sw, Qn, W_draw).reshape(B*N, d, k)
        Qp3 = gs_orth(Qn.reshape(B*N, d, k) + delta * U)
        inc2 = det_step_increment(sw, Qp3.reshape(B, N, d, k), W_draw).reshape(B*N, d, k)
        dQ = ((Qp3 + inc2) - (Qn.reshape(B*N, d, k) + inc1))
        # project difference to tangent of reference (removes radial component)
        dQ = proj_tan(Qn.reshape(B*N, d, k), dQ)
        ndq = dQ.norm(dim=(-2,-1))
        logsum += torch.log((ndq / delta).clamp(1e-30, 1e30)).reshape(B, N).sum(-1) / N
        U = dQ / ndq.clamp_min(1e-30).view(-1, 1, 1)
        n += 1
    lyap = (logsum / n).reshape(Bc, Be)
    return {"sigma": sigma, "lam_p": lam_p_list, "eta_grid": eta_grid,
            "lyap": lyap.cpu()}

def main():
    dev = "cuda"
    t0 = time.time()
    out = {}
    for (k, d) in [(2,8),(1,2)]:
        for s in (0.05, 0.1):
            lam_p = [(0.0,0.0),(0.05,0.5),(0.1,0.5),(0.2,0.5),(0.1,1.0),(0.2,0.25)]
            etas = [math.exp(math.log(0.02)+(math.log(8.0)-math.log(0.02))*i/27) for i in range(28)]
            res = lyap_det(k, d, s, lam_p, etas, N=64, T=3000, burn=1500, device=dev)
            key = f"{k},{d}_s={s}"
            out[key] = {**res, "lyap": res["lyap"].tolist()}
            print(f"== {key} ({time.time()-t0:6.1f}s)")
            for j, (l, p) in enumerate(lam_p):
                row = [f"{x:+8.4f}" for x in res["lyap"][j]]
                print(f"  l={l:4.2f},p={p:3.2f}: " + " ".join(row))
            print("  eta      :", " ".join(f"{e:8.3f}" for e in etas))
    json.dump(out, open("results/phaseE2_lyapdet.json", "w"))
    print("saved results/phaseE2_lyapdet.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
