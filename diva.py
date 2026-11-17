"""
DiVa swarm simulator v3: elementwise parameters + adiabatic sweeps.

Elementwise per batch element: sigma, lam_W, p_fix (shape (B,1,1) tensors).
Sweep: shared eta schedule (geometric), state carried across eta values
(adiabatic) -> hysteresis loops; exact geodesic U-statistic measured on demand.
"""
import torch

torch.set_float32_matmul_precision("high")


def gs_orth(X):
    """Gram-Schmidt orthonormalization of (M,d,k), k<=4."""
    k = X.shape[2]
    U = torch.empty_like(X)
    for c in range(k):
        v = X[:, :, c]
        for cc in range(c):
            u = U[:, :, cc]
            v = v - (u * v).sum(-1, keepdim=True) * u
        U[:, :, c] = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return U


def haar_Q(shape, device, dtype=torch.float32):
    G = torch.randn(shape, device=device, dtype=dtype)
    return gs_orth(G.reshape(-1, shape[-2], shape[-1])).reshape(shape)


def proj_tan(Q, X):
    return X - torch.bmm(Q, torch.bmm(Q.transpose(1, 2), X))


class Swarm:
    """B parallel swarms of N agents on Gr(k,d); elementwise (sigma, lam_W, p_fix)."""

    def __init__(self, B, N, d, k, device="cuda", dtype=torch.float32):
        self.B, self.N, self.d, self.k = B, N, d, k
        self.device, self.dtype = device, dtype
        self.Q = haar_Q((B, N, d, k), device, dtype)
        self.W_fix = haar_Q((B, 1, d, k), device, dtype)
        self.sigma = torch.zeros(B, 1, 1, 1, device=device, dtype=dtype)
        self.lam_W = torch.zeros(B, 1, 1, 1, device=device, dtype=dtype)
        self.p_fix = torch.zeros(B, 1, 1, 1, device=device, dtype=dtype)
        self.t = 0

    def set_params(self, sigma=None, lam_W=None, p_fix=None):
        for name, val in (("sigma", sigma), ("lam_W", lam_W), ("p_fix", p_fix)):
            if val is None:
                continue
            t = torch.as_tensor(val, device=self.device, dtype=self.dtype).reshape(-1)
            if t.numel() == 1:
                t = t.expand(self.B, 1, 1, 1)
            elif t.numel() == self.B:
                t = t.view(self.B, 1, 1, 1)
            else:
                raise ValueError(f"param size {t.numel()} != B={self.B}")
            setattr(self, name, t)

    def anchors(self, u=None, fresh=None):
        B, N, d, k = self.B, self.N, self.d, self.k
        if u is None:
            u = torch.rand(B, N, 1, 1, device=self.device, dtype=self.dtype)
        if fresh is None:
            fresh = haar_Q((B, N, d, k), self.device, self.dtype)
        fixed = self.W_fix.expand(B, N, d, k)
        return torch.where(u < self.p_fix, fixed, fresh), (u < self.p_fix).to(self.dtype)

    def step(self, eta, noise_G=None, anchor_u=None, anchor_fresh=None):
        B, N, d, k = self.B, self.N, self.d, self.k
        if not torch.is_tensor(eta):
            eta = torch.tensor(float(eta), device=self.device, dtype=self.dtype)
        if eta.dim() == 0:
            eta4 = eta.view(1, 1, 1, 1)
        elif eta.numel() == B:
            eta4 = eta.reshape(B, 1, 1, 1)
        elif eta.numel() == B * N:
            eta4 = eta.reshape(B, N, 1, 1)
        else:
            raise ValueError("eta shape")
        Q = self.Q
        Q3 = Q.reshape(B * N, d, k)
        Q2 = Q.transpose(1, 2).reshape(B, d, N * k)
        Sbar = (Q2 @ Q2.transpose(1, 2)) / N
        Sx = Sbar.unsqueeze(1).expand(B, N, d, d).reshape(B * N, d, d)
        drift = torch.bmm(Sx, Q3)
        drift = proj_tan(Q3, drift)
        upd = eta4 * drift.reshape(B, N, d, k)                  # (B,N,d,k)
        if torch.any(self.lam_W != 0):
            W, indicator = self.anchors(u=anchor_u, fresh=anchor_fresh)
            self.last_anchor_indicator = indicator                     # (B,N,1,1)
            W3 = W.reshape(B * N, d, k)
            a = proj_tan(Q3, torch.bmm(W3, torch.bmm(W3.transpose(1, 2), Q3)))
            upd = upd + self.lam_W * a.reshape(B, N, d, k)
        if torch.any(self.sigma != 0):
            G = noise_G if noise_G is not None else torch.randn(
                B * N, d, k, device=self.device, dtype=self.dtype)
            z = proj_tan(Q3, G)
            upd = upd + self.sigma * z.reshape(B, N, d, k)
        self.Q = gs_orth((Q + upd).reshape(B * N, d, k)).reshape(B, N, d, k)
        self.t += 1

    def R_order(self):
        B, N, d, k = self.B, self.N, self.d, self.k
        Q = self.Q
        Q2 = Q.transpose(1, 2).reshape(B, d, N * k)
        Sbar = (Q2 @ Q2.transpose(1, 2)) / N
        return torch.linalg.eigvalsh(Sbar)[..., -1]

    def _pairM(self):
        """(B,N,N,k,k): Q_i^T Q_j."""
        B, N, d, k = self.B, self.N, self.d, self.k
        Q = self.Q
        Qt = Q.transpose(-1, -2)
        A = Qt.unsqueeze(2).expand(B, N, N, k, d).reshape(B * N * N, k, d)
        Bm = Q.unsqueeze(1).expand(B, N, N, d, k).reshape(B * N * N, d, k)
        return torch.bmm(A, Bm).reshape(B, N, N, k, k)

    def pairwise_geo(self):
        s = torch.linalg.svdvals(self._pairM())
        ang = torch.acos(s.clamp(-1.0, 1.0))
        return ang.pow(2).sum(-1).sqrt()

    def Delta(self):
        """exact geodesic U-statistic."""
        h = self.pairwise_geo()
        return h.sum(dim=(-2, -1)) / (self.N * self.N)

    def Delta_chord(self):
        """chordal proxy: sqrt(mean_pairs (k - tr(P_i P_j)))."""
        M = self._pairM()
        tr = M.pow(2).sum(dim=(-2, -1))                          # tr(P_iP_j)
        v = (self.k - tr).clamp_min(0)
        return v.mean(dim=(-2, -1)).sqrt()

    def hoeffding(self):
        s = torch.linalg.svdvals(self._pairM())
        ang = torch.acos(s.clamp(-1.0, 1.0))
        h = ang.pow(2).sum(-1).sqrt()
        B, n = self.B, self.N
        mask = (1.0 - torch.eye(n, device=h.device, dtype=h.dtype))
        hmask = h * mask
        msum = n * (n - 1)
        kh_mean = hmask.sum(dim=(-2, -1), keepdim=True) / msum
        kh_var = ((hmask - kh_mean).pow(2) * mask).sum(dim=(-2, -1)) / msum
        r = hmask.sum(-1) / (n - 1)
        zeta_raw = r.var(dim=-1, unbiased=False)
        floor = kh_var / (n - 1)
        zeta_corr = (zeta_raw - floor).clamp_min(0.0) / kh_var.clamp_min(1e-12)
        return zeta_raw, zeta_corr, kh_var

    def clone_perturbed(self, delta=1e-5, seed=None):
        """return a copy of self with tiny tangential perturbations (Benettin pair)."""
        import copy
        s2 = copy.copy(self)
        s2.Q = self.Q.clone()
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device); g.manual_seed(seed)
        z = torch.randn(self.Q.shape, device=self.device, dtype=self.dtype,
                        generator=g)
        z = z - torch.bmm(self.Q.reshape(-1, self.d, self.k),
                          torch.bmm(self.Q.reshape(-1, self.d, self.k).transpose(1, 2),
                                    z.reshape(-1, self.d, self.k))).reshape(self.Q.shape)
        z = z / z.norm(dim=(-2, -1), keepdim=True)
        Qp = gs_orth(self.Q.reshape(-1, self.d, self.k) + delta * z.reshape(-1, self.d, self.k))
        s2.Q = Qp.reshape(self.Q.shape)
        return s2


@torch.no_grad()
def adiabatic_sweep(sigmas, lams, pfixs, eta_sched, N=64, d=8, k=2,
                    steps_per_eta=200, T_drift=200, init="haar",
                    device="cuda", dtype=torch.float32, seed=0,
                    exact_at=None, pre_eta=None, pre_steps=0, perturb_init=0.0):
    """
    Batched adiabatic sweep with elementwise (sigma, lam_W, p_fix).
    eta_sched: list of eta values (shared, geometric progression recommended).
    init: 'haar' or 'sync' (all agents one random subspace).
    exact_at: list of indices into eta_sched where exact Delta/hoeffding is measured.
    Returns dict with per-element arrays over the schedule.
    """
    dev, dt = device, dtype
    torch.manual_seed(seed)
    P = len(sigmas)
    B = P
    assert len(lams) == P and len(pfixs) == P
    sw = Swarm(B, N, d, k, dev, dt)
    sw.set_params(sigma=sigmas, lam_W=lams, p_fix=pfixs)
    if init == "sync":
        base = haar_Q((B, 1, d, k), dev, dt)
        sw.Q = base.unsqueeze(2).expand(B, 1, N, d, k).reshape(B, N, d, k).contiguous()
        if perturb_init > 0:
            Q3 = sw.Q.reshape(B * N, d, k)
            z = torch.randn(B * N, d, k, device=dev, dtype=dt)
            z = z - torch.bmm(Q3, torch.bmm(Q3.transpose(1, 2), z))
            z = z / z.norm(dim=(-2, -1), keepdim=True)
            sw.Q = gs_orth(Q3 + perturb_init * z).reshape(B, N, d, k)
    if pre_eta is not None and pre_steps > 0:
        et0 = torch.full((B,), float(pre_eta), device=dev, dtype=dt)
        for _ in range(pre_steps):
            sw.step(et0)

    L = len(eta_sched)
    dc = torch.empty(B, L)
    R = torch.empty(B, L)
    exact = {}
    for li, eta in enumerate(eta_sched):
        et = torch.full((B,), float(eta), device=dev, dtype=dt)
        for _ in range(steps_per_eta):
            sw.step(et)
        for _ in range(T_drift):
            sw.step(et)
        dc[:, li] = sw.Delta_chord()
        R[:, li] = sw.R_order()
        if exact_at is not None and li in exact_at:
            eD = sw.Delta()
            zr, zc, kv = sw.hoeffding()
            exact[li] = (eD.cpu(), zr.cpu(), zc.cpu(), kv.cpu())
    out = {
        "params": {"sigma": list(map(float, sigmas)), "lam": list(map(float, lams)),
                   "pfix": list(map(float, pfixs))},
        "eta_sched": list(map(float, eta_sched)),
        "Delta_chord": dc.cpu(), "R": R.cpu(), "exact": exact,
    }
    return out


def geo_grid(lo, hi, n):
    import math
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]
