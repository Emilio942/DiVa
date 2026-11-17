"""Phase G: (a) deterministic spinodal (sigma=0 lambda_det) -> transition character;
(b) extra (k,d) edges with printed curves; (c) conditional zeta1 v2 (cloud coordinate)."""
import json, math, time
import torch
from diva import Swarm, gs_orth, haar_Q, proj_tan
from phaseE2 import det_step_increment, lyap_det
from phaseF import edge_at, zeta_conditional

@torch.no_grad()
def spinodal(k, d, N=64, T=2500, burn=1000, delta=1e-6, device="cuda", dtype=torch.float32,
             seed=51, eta_lo=0.002, eta_hi=2.0, n=20):
    """sigma=0, lam=0: lambda_det(eta) of the exactly synchronized state."""
    dev = device
    torch.manual_seed(seed)
    etas = [math.exp(math.log(eta_lo)+(math.log(eta_hi)-math.log(eta_lo))*i/(n-1)) for i in range(n)]
    etas_t = torch.tensor(etas, device=dev, dtype=dtype)
    B = n
    sw = Swarm(B, N, d, k, dev, dtype)
    sw.set_params(sigma=0.0)
    sw.eta4 = etas_t.reshape(B,1,1,1)
    base = haar_Q((B,1,d,k), dev, dtype)
    sw.Q = base.unsqueeze(2).expand(B,1,N,d,k).reshape(B,N,d,k).contiguous()
    g = torch.Generator(device=dev); g.manual_seed(seed+1)
    z = torch.randn(B*N, d, k, device=dev, dtype=dtype, generator=g)
    zz = proj_tan(sw.Q.reshape(B*N,d,k), z)
    U = zz / zz.norm(dim=(-2,-1), keepdim=True)
    logsum = torch.zeros(B, device=dev, dtype=dtype); cnt = 0
    for t in range(T):
        if t < burn:
            sw.step(etas_t)
            continue
        Qn = sw.Q
        inc1 = det_step_increment(sw, Qn, None).reshape(B*N, d, k)
        Qp3 = gs_orth(Qn.reshape(B*N,d,k) + delta*U)
        inc2 = det_step_increment(sw, Qp3.reshape(B,N,d,k), None).reshape(B*N, d, k)
        dQ = proj_tan(Qn.reshape(B*N,d,k), (Qp3+inc2) - (Qn.reshape(B*N,d,k)+inc1))
        ndq = dQ.norm(dim=(-2,-1))
        logsum += torch.log((ndq/delta).clamp(1e-30,1e30)).reshape(B,N).sum(-1)/N
        U = dQ / ndq.clamp_min(1e-30).view(-1,1,1)
        sw.step(etas_t)
        cnt += 1
    return etas, (logsum/cnt).tolist()

def main():
    dev = "cuda"
    out = {}
    print("== (a) deterministic spinodal of the synchronized state (sigma=0) ==")
    sp = {}
    for (k,d) in [(1,2),(1,4),(2,4),(2,8)]:
        etas, ld = spinodal(k, d, device=dev)
        sp[f"{k},{d}"] = {"etas": etas, "lyap": ld}
        # find upper spinodal: first eta where lyap > 0 going up
        e_up = next((etas[i] for i in range(len(ld)) if ld[i] > 0 and i > 0 and ld[i-1] <= 0), None)
        print(f"  ({k},{d}): upper spinodal eta* = {e_up}")
        print("     ", " ".join(f"{x:+8.4f}" for x in ld))
        print("      eta:", " ".join(f"{e:8.4f}" for e in etas))
    out["spinodal"] = sp
    print("== (b) extra (k,d) edges, printed curves ==")
    ex = {}
    for (k,d,guess) in [(1,5,16.0),(2,3,11.0),(3,9,30.0)]:
        e, D, etas = edge_at(k, d, 0.05, guess, device=dev, T=8000, burn=3500)
        ex[f"{k},{d}"] = {"edge": e, "curve": D, "etas": etas}
        print(f"  ({k},{d}): edge={e}  D={['%.3f'%x for x in D]}")
        print(f"        etas={['%.4f'%x for x in etas]}")
    out["extra_kd"] = ex
    print("== (c) conditional zeta1 v2: regress r_i on distance-to-W_fix ==")
    # reuse zeta_conditional but with regression residual
    @torch.no_grad()
    def zeta_reg(k, d, sigma, lam, pfix, eta, N=64, T=4000, burn=1500, device="cuda", seed=61):
        dvt = device
        torch.manual_seed(seed)
        sw = Swarm(4, N, d, k, dvt)
        sw.set_params(sigma=sigma, lam_W=lam, p_fix=pfix)
        et = torch.full((4,), eta, device=dvt)
        accz = torch.zeros(4, device=dvt); acczw = torch.zeros(4, device=dvt)
        acc0 = torch.zeros(4, device=dvt); kv0 = torch.zeros(4, device=dvt)
        n = 0
        for t in range(T):
            u = torch.rand(4, N, 1, 1, device=dvt)
            fresh = haar_Q((4,N,d,k), dvt) if lam>0 else None
            sw.step(et, anchor_u=u, anchor_fresh=fresh)
            if t >= burn:
                h = sw.pairwise_geo()
                mask = (1.0-torch.eye(N, device=dvt))
                r = (h*mask).sum(-1)/(N-1)
                # cloud coordinate: geodesic distance of agent i to W_fix (replicate-specific)
                Wf = sw.W_fix.squeeze(1)                       # (4,d,k)
                c = torch.acos(torch.linalg.svdvals(
                    torch.einsum("bndk,bdj->bnjk", sw.Q, Wf.transpose(1,2)).transpose(2,3)
                    if False else torch.matmul(sw.Q.transpose(-1,-2), Wf.unsqueeze(1)).squeeze(-2).transpose(-1,-2)
                ).clamp(-1,1))
                dist = c.pow(2).sum(-1).sqrt()                  # (4,N)
                ztot = r.var(dim=-1, unbiased=False)
                # linear fit r_i = a + b*dist_i per replicate; residual variance
                x = dist - dist.mean(-1, keepdim=True)
                y = r - r.mean(-1, keepdim=True)
                b = (x*y).sum(-1)/x.pow(2).sum(-1).clamp_min(1e-12)
                res = y - b.unsqueeze(-1)*x
                zres = res.var(dim=-1, unbiased=False)
                hmask = h*mask
                kh_mean = hmask.sum(dim=(-2,-1), keepdim=True)/(N*(N-1))
                kv = ((hmask-kh_mean)**2*mask).sum(dim=(-2,-1))/(N*(N-1))
                accz += ztot; acczw += zres; acc0 += ((ztot-kv/(N-1)).clamp_min(0)/kv.clamp_min(1e-12))
                kv0 += kv; n += 1
        return {"z_total_raw": (accz/n).tolist(), "z_resid_raw": (acczw/n).tolist(),
                "z_total_corr": (acc0/n).tolist(), "kvar": (kv0/n).tolist()}
    rz = []
    for (lam,p,eta) in [(0.05,0.5,0.3),(0.1,0.5,0.3),(0.2,0.5,0.3),(0.1,1.0,0.3)]:  # noqa
        r = zeta_reg(2, 8, 0.05, lam, p, eta, device=dev)
        rz.append({"lam":lam,"p":p,"eta":eta, **r})
        print(f"  l={lam} p={p} eta={eta}: z_total={sum(r['z_total_raw'])/4:.2e} "
              f"z_resid={sum(r['z_resid_raw'])/4:.2e} z_corr={sum(r['z_total_corr'])/4:.3f}")
    out["zeta_reg"] = rz
    json.dump(out, open("results/phaseG.json", "w"))
    print("saved results/phaseG.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
