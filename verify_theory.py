"""
Theory Verification Suite for HERLEITUNG.md (Problems 1-5 of DiVa).

Reproduces all analytical and numerical verification checks:
  1. Problem 4 (k=1): exact 1D Boltzmann free energy, a_sp, a_1st, a_half_eq, a_lin,
     plus Kramers nucleation barrier collapse N * dF_bar(a) against V8 data (F8).
  2. Problem 4 (general k,d): exact Grassmannian duality test (k,d) <-> (d-k,d)
     and unmeasured predictions (1,6), (2,6) vs (5,6), (4,6) and (1,4) vs (3,4).
  3. Problem 1: exact Boltzmann zeta_1^inf at the (2,8) edge and exact finite-N
     estimator bias formula against V7 data.
  4. Problem 3: smoking-gun test on (1,3) (D=2 -> critical eta = 2D/(D+1) = 4/3)
     and global vs per-agent normalization on (1,2).
  5. Problem 2: phaseE2.py raw vs gs_orth diagnosis of epsilon(eta; sigma)
     and flip-boundary invariant eta*R + x along V9 points.

Output: results/theory_verification.json (runtime ~1 min on RTX 3060).
"""
import json
import math
import os
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

from diva import Swarm, gs_orth, haar_Q, proj_tan
from phaseBC import edge_scan, extract_edges
from phaseE2 import det_step_increment


def verify_problem4_k1_and_f8():
    """Exact 1D Boltzmann quadrature for k=1 and Kramers barrier collapse for F8."""
    x_gl, w_gl = leggauss(300)
    th = 0.5 * (x_gl + 1.0) * (np.pi / 2.0)
    wth = 0.5 * w_gl * (np.pi / 2.0)
    c2 = np.cos(th) ** 2

    def barrier_and_eq(d):
        log_jac = (d - 2) * np.log(np.sin(th))
        log_jac -= log_jac.max()
        base_w = wth * np.exp(log_jac)
        base_w /= base_w.sum()

        hs = np.concatenate([np.linspace(1e-3, 1.0, 150), np.linspace(1.0, 12 * d, 1500)])
        a_v, q_v, dF_v = [], [], []
        for h in hs:
            lw = h * c2
            lwm = lw.max()
            Z = np.sum(base_w * np.exp(lw - lwm))
            logZ = lwm + np.log(Z)
            w = base_w * np.exp(lw - lwm) / Z
            m = np.sum(w * c2)
            q = (m - 1.0 / d) / (1.0 - 1.0 / d)
            a = h / q
            dF = 0.5 * a * (1.0 - 1.0 / d) * q ** 2 + h / d - logZ
            a_v.append(a)
            q_v.append(q)
            dF_v.append(dF)
        a_v, q_v, dF_v = map(np.array, (a_v, q_v, dF_v))
        idx_sp = int(np.argmin(a_v))
        a_sp = float(a_v[idx_sp])
        a_lin = float(d * (d + 2) / 2.0)
        if d == 2:
            a_1st = a_lin
        else:
            cross = int(np.where(dF_v[idx_sp:] <= 0)[0][0])
            a_1st = float(a_v[idx_sp + cross])
        return a_sp, a_1st, a_lin, a_v[: idx_sp + 1], dF_v[: idx_sp + 1]

    eq_summary = {}
    curves = {}
    for d in [2, 3, 4, 5, 6, 8, 16, 24, 32]:
        a_sp, a_1st, a_lin, au, dfu = barrier_and_eq(d)
        eq_summary[str(d)] = {"a_sp": a_sp, "a_1st": a_1st, "a_lin": a_lin}
        curves[d] = (au, dfu)

    v8_pts = [
        (16, 64, 52.4),
        (16, 128, 1.32 * 49),
        (16, 256, 1.57 * 49),
        (32, 32, 105.3),
        (32, 64, 123.5),
        (32, 128, 1.60 * 97),
        (32, 256, 2.02 * 97),
    ]
    kramers = []
    for d, N, a_meas in v8_pts:
        au, dfu = curves[d]
        idx = int(np.argmin(np.abs(au - a_meas)))
        df = float(dfu[idx])
        kramers.append({"d": d, "N": N, "a_meas": a_meas, "dF_bar": df, "N_dF_bar": N * df})
    return {"k1_eq": eq_summary, "f8_kramers": kramers}


@torch.no_grad()
def verify_problem4_duality(device="cuda"):
    """Unmeasured (1,6),(2,6) and Grassmannian duals (5,6),(4,6),(3,4) vs (1,4)."""
    items = [
        dict(k=1, d=6, sigma=0.05, lam=0.0, pfix=0.0, asig=18.5, base=1e-3),
        dict(k=5, d=6, sigma=0.05, lam=0.0, pfix=0.0, asig=18.5, base=1e-3),
        dict(k=2, d=6, sigma=0.05, lam=0.0, pfix=0.0, asig=19.5, base=1e-3),
        dict(k=4, d=6, sigma=0.05, lam=0.0, pfix=0.0, asig=19.5, base=1e-3),
        dict(k=1, d=4, sigma=0.05, lam=0.0, pfix=0.0, asig=13.0, base=1e-3),
        dict(k=3, d=4, sigma=0.05, lam=0.0, pfix=0.0, asig=13.0, base=1e-3),
    ]
    jobs, results = edge_scan(
        items, N=64, T=6000, burn=2500, n_eta=16, span=(1 / 2.5, 2.5), device=device, seed=77
    )
    edges = extract_edges(jobs, results)
    out = {}
    for ii, it in enumerate(items):
        k, d = it["k"], it["d"]
        e = edges[ii]["edge"]
        a = e / (it["sigma"] ** 2)
        out[f"{k},{d}"] = {
            "a_meas": a,
            "pred_3dk": 3 * d + k,
            "pred_3d_mink": 3 * d + min(k, d - k),
            "z1_at_edge": edges[ii]["z1_at_edge"],
        }
    return out


@torch.no_grad()
def verify_problem1_zeta1(device="cuda"):
    """Exact Boltzmann zeta_1^inf at (2,8) edge and finite-N bias check vs V7."""
    torch.manual_seed(0)
    M, d, k, a_target, sigma = 4096, 8, 2, 25.1, 0.04
    Q = torch.zeros(M, d, k, device=device)
    Q[:, :k, :k] = torch.eye(k, device=device)
    Q = gs_orth(Q + 0.2 * torch.randn(M, d, k, device=device))
    eta = a_target * sigma * sigma
    for _ in range(3000):
        Q2 = Q.permute(1, 0, 2).reshape(d, M * k)
        Sbar = (Q2 @ Q2.T) / M
        drift = proj_tan(Q, torch.matmul(Sbar, Q))
        z = proj_tan(Q, torch.randn(M, d, k, device=device))
        Q = gs_orth(Q + eta * drift + sigma * z)

    M2 = 2048
    Qs = Q[:M2]
    Mmat = torch.einsum("idk,jdl->ijkl", Qs, Qs)
    sv = torch.linalg.svdvals(Mmat)
    h = torch.acos(sv.clamp(-1.0, 1.0)).pow(2).sum(-1).sqrt()
    mask = 1.0 - torch.eye(M2, device=device)
    hm = h * mask
    msum = M2 * (M2 - 1)
    km = hm.sum() / msum
    kv = ((hm - km).pow(2) * mask).sum() / msum
    r = hm.sum(-1) / (M2 - 1)
    zraw = r.var(unbiased=False)
    z_exact = (
        (zraw / kv - (M2 - 2) / (M2 * (M2 - 1)))
        / ((M2 - 2) * (M2 - 4) / (M2 * (M2 - 1)))
    ).item()

    meas_v7 = [(32, 0.217), (64, 0.239), (128, 0.246), (256, 0.252), (512, 0.2531), (1024, 0.2558)]
    bias_table = []
    for N, m in meas_v7:
        pred = z_exact * (1.0 - (5 * N - 8) / (N * (N - 1))) - 2.0 / (N * (N - 1))
        bias_table.append({"N": N, "pred": pred, "meas": m, "diff": pred - m})
    return {"zeta1_inf_28": z_exact, "Delta_28": km.item(), "bias_vs_V7": bias_table}


@torch.no_grad()
def verify_problem3_dichotomy(device="cuda"):
    """Smoking-gun test on (1,3) (eta*=4/3) and global norm on (1,2)."""
    def run_spinodal(k, d, etas, per_agent_norm=True, N=64, T=1500, burn=300, delta=1e-7):
        B = len(etas)
        etas_t = torch.tensor(etas, device=device, dtype=torch.float64)
        sw = Swarm(B, N, d, k, device, torch.float64)
        sw.set_params(sigma=0.0)
        sw.eta4 = etas_t.reshape(B, 1, 1, 1)
        torch.manual_seed(51)
        base = haar_Q((B, 1, d, k), device, torch.float64)
        sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
        z = torch.randn(B * N, d, k, device=device, dtype=torch.float64)
        zz = proj_tan(sw.Q.reshape(B * N, d, k), z)
        if per_agent_norm:
            U = zz / zz.norm(dim=(-2, -1), keepdim=True)
        else:
            Uf = zz.reshape(B, -1)
            U = (Uf / Uf.norm(dim=-1, keepdim=True)).reshape(B * N, d, k)
        logsum = torch.zeros(B, device=device, dtype=torch.float64)
        cnt = 0
        for t in range(T):
            Qn = sw.Q
            inc1 = det_step_increment(sw, Qn, None).reshape(B * N, d, k)
            Qp3 = gs_orth(Qn.reshape(B * N, d, k) + delta * U)
            inc2 = det_step_increment(sw, Qp3.reshape(B, N, d, k), None).reshape(B * N, d, k)
            dQ = proj_tan(Qn.reshape(B * N, d, k), (Qp3 + inc2) - (Qn.reshape(B * N, d, k) + inc1))
            if per_agent_norm:
                ndq = dQ.norm(dim=(-2, -1))
                if t >= burn:
                    logsum += torch.log((ndq / delta).clamp(1e-30, 1e30)).reshape(B, N).sum(-1) / N
                    cnt += 1
                U = dQ / ndq.clamp_min(1e-30).view(-1, 1, 1)
            else:
                ndq = dQ.reshape(B, -1).norm(dim=-1)
                if t >= burn:
                    logsum += torch.log((ndq / delta).clamp(1e-30, 1e30))
                    cnt += 1
                U = (dQ.reshape(B, -1) / ndq.clamp_min(1e-30).view(B, 1)).reshape(B * N, d, k)
            sw.step(etas_t)
        return (logsum / cnt).tolist()

    etas = [0.2, 0.8, 1.2, 1.30, 1.38, 1.5, 1.8, 2.5]
    return {
        "etas": etas,
        "1,3_per_agent": run_spinodal(1, 3, etas, per_agent_norm=True),
        "1,2_per_agent": run_spinodal(1, 2, etas, per_agent_norm=True),
        "1,2_global": run_spinodal(1, 2, etas, per_agent_norm=False),
        "ln_abs_1_minus_eta": [math.log(abs(1.0 - e)) for e in etas],
    }


@torch.no_grad()
def verify_problem2_anchors(device="cuda"):
    """Compare raw vs gs_orth in phaseE2.py and verify eta*R + x along V9 (p=1)."""
    def diag_lyap(etas, lam, pfix, sigma=0.05, k=2, d=8, N=64, T=1500, burn=800, delta=1e-5, use_gs_out=False):
        torch.manual_seed(23)
        B = len(etas)
        sw = Swarm(B, N, d, k, device, torch.float64)
        sw.set_params(sigma=sigma, lam_W=lam, p_fix=pfix)
        etas_b = torch.tensor(etas, device=device, dtype=torch.float64)
        sw.eta4 = etas_b.reshape(B, 1, 1, 1)
        base = haar_Q((B, 1, d, k), device, torch.float64)
        sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
        for _ in range(burn):
            sw.step(etas_b)
        z = torch.randn(B * N, d, k, device=device, dtype=torch.float64)
        zz = proj_tan(sw.Q.reshape(B * N, d, k), z)
        U = zz / zz.norm(dim=(-2, -1), keepdim=True)
        logsum = torch.zeros(B, device=device, dtype=torch.float64)
        Rsum = torch.zeros(B, device=device, dtype=torch.float64)
        cnt = 0
        for t in range(T):
            G = torch.randn(B * N, d, k, device=device, dtype=torch.float64)
            u = torch.rand(B, N, 1, 1, device=device, dtype=torch.float64)
            fresh = haar_Q((B, N, d, k), device, torch.float64) if lam > 0 else None
            sw.step(etas_b, noise_G=G, anchor_u=u, anchor_fresh=fresh)
            Qn = sw.Q
            W_draw, _ = sw.anchors(u=u, fresh=fresh) if lam > 0 else (None, None)
            inc1 = det_step_increment(sw, Qn, W_draw).reshape(B * N, d, k)
            Qp3 = gs_orth(Qn.reshape(B * N, d, k) + delta * U)
            inc2 = det_step_increment(sw, Qp3.reshape(B, N, d, k), W_draw).reshape(B * N, d, k)
            if use_gs_out:
                Y1 = gs_orth(Qn.reshape(B * N, d, k) + inc1)
                Y2 = gs_orth(Qp3 + inc2)
                dQ = proj_tan(Y1, Y2 - Y1)
            else:
                dQ = proj_tan(Qn.reshape(B * N, d, k), (Qp3 + inc2) - (Qn.reshape(B * N, d, k) + inc1))
            ndq = dQ.norm(dim=(-2, -1))
            if t >= 200:
                logsum += torch.log((ndq / delta).clamp(1e-30, 1e30)).reshape(B, N).sum(-1) / N
                Rsum += sw.R_order()
                cnt += 1
            U = dQ / ndq.clamp_min(1e-30).view(-1, 1, 1)
        return (logsum / cnt).tolist(), (Rsum / cnt).tolist()

    etas_u = [0.3, 1.0, 1.5, 1.65, 1.8, 2.0, 2.1, 2.2]
    l_raw, R_u = diag_lyap(etas_u, 0.0, 0.0, use_gs_out=False)
    l_gs, _ = diag_lyap(etas_u, 0.0, 0.0, use_gs_out=True)

    v9_p1 = [(0.02, 1.645), (0.05, 2.091), (0.10, 2.145), (0.20, 2.026), (0.40, 1.796), (0.60, 1.573)]
    anchored = []
    for x, e_b in v9_p1:
        lr, Rv = diag_lyap([e_b], x, 1.0, use_gs_out=False)
        lg, _ = diag_lyap([e_b], x, 1.0, use_gs_out=True)
        anchored.append({
            "x": x,
            "B_meas": e_b,
            "R": Rv[0],
            "eta_R_plus_x": e_b * Rv[0] + x,
            "lyap_raw": lr[0],
            "lyap_gs": lg[0],
        })
    return {
        "unanchored": {"etas": etas_u, "R": R_u, "lyap_raw": l_raw, "lyap_gs": l_gs},
        "anchored_p1": anchored,
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    out = {}

    print("[1/5] Problem 4 (k=1 Boltzmann + F8 Kramers barrier)...")
    out["problem4_k1_f8"] = verify_problem4_k1_and_f8()
    for row in out["problem4_k1_f8"]["f8_kramers"]:
        print(f"  d={row['d']:2d}, N={row['N']:3d}, a={row['a_meas']:5.1f} -> N*dF_bar={row['N_dF_bar']:5.2f}")

    print("[2/5] Problem 4 (unmeasured points & Grassmannian duality k <-> d-k)...")
    out["problem4_duality"] = verify_problem4_duality(dev)
    for cfg, v in out["problem4_duality"].items():
        print(f"  ({cfg}): a_meas={v['a_meas']:5.2f} | 3d+k={v['pred_3dk']:2d} | 3d+min(k,d-k)={v['pred_3d_mink']:2d}")

    print("[3/5] Problem 1 (Boltzmann zeta_1^inf + exact finite-N bias vs V7)...")
    out["problem1_zeta1"] = verify_problem1_zeta1(dev)
    print(f"  zeta1_inf(2,8) = {out['problem1_zeta1']['zeta1_inf_28']:.5f}")
    for row in out["problem1_zeta1"]["bias_vs_V7"]:
        print(f"  N={row['N']:4d}: pred={row['pred']:.4f} vs meas={row['meas']:.4f} (diff={row['diff']:+.4f})")

    print("[4/5] Problem 3 ((1,3) eta*=4/3 smoking gun & (1,2) global norm)...")
    out["problem3_dichotomy"] = verify_problem3_dichotomy(dev)
    for e, a, b, c in zip(
        out["problem3_dichotomy"]["etas"],
        out["problem3_dichotomy"]["1,3_per_agent"],
        out["problem3_dichotomy"]["1,2_per_agent"],
        out["problem3_dichotomy"]["1,2_global"],
    ):
        print(f"  eta={e:4.2f}: (1,3)={a:+7.4f} | (1,2)pa={b:+7.4f} | (1,2)gl={c:+7.4f}")

    print("[5/5] Problem 2 (raw vs gs_orth & flip invariant eta*R + x)...")
    out["problem2_anchors"] = verify_problem2_anchors(dev)
    for row in out["problem2_anchors"]["anchored_p1"]:
        print(f"  x={row['x']:4.2f}, B={row['B_meas']:5.3f} -> R={row['R']:.4f}, eta*R+x={row['eta_R_plus_x']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/theory_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/theory_verification.json ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
