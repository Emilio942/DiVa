# DiVa — Dichotomy under Anchor Interpolation

**Synchronization of subspace swarms on Grassmannians: numerical verification of a
conjectured critical surface.**

DiVa studies a mean-field swarm of $N$ agents living on the Grassmannian
$\mathrm{Gr}(k,d)$ (the manifold of $k$-dimensional subspaces of $\mathbb{R}^d$).
Each agent is pulled toward the swarm's mean projector, receives tangential
Gaussian noise, and is — with probability $p_{\mathrm{fix}}$ — additionally pulled
toward a common quenched anchor subspace:

$$
Q_i^{(t+1)} = \mathrm{orth}\!\Big(
Q_i^{(t)} + \eta\,(I - Q_i Q_i^{\top})\,\bar{S}\,Q_i^{(t)}
+ \lambda_W\,(I - Q_i Q_i^{\top})\,W_i^{(t)} W_i^{(t)\top} Q_i^{(t)}
+ \sigma\,(I - Q_i Q_i^{\top})\,G_i^{(t)} \Big),
$$

with $\bar{S} = \tfrac1N\sum_j Q_j Q_j^{\top}$, $G_i \sim \mathcal N(0, I_{d\times k})$
i.i.d., and $W_i = W_{\mathrm{fix}}$ w.p. $p_{\mathrm{fix}}$, else Haar on
$\mathrm{Gr}(k,d)$.

The starting conjecture was that the critical coupling of the synchronization
dichotomy shifts as
$\eta_c(\sigma,\lambda_W,p_{\mathrm{fix}}) = \eta_c(0) - a_\sigma\sigma^2 - a_W\lambda_W^2 p_{\mathrm{fix}}$,
with a symmetry-protected "marginal degeneracy" preserved across the shifted
surface, and a transition character that is continuous for $d=2$ / subcritical for
$d\ge4$.

## Key results (measured, with error bars)

| # | Law | Value |
|---|-----|-------|
| F1 | Lower critical edge | $\eta_{\mathrm{lo}} = a_\sigma\,\sigma^2$, zero intercept |
| F2 | Noise coefficient | $a_\sigma = 3d+k$ (±5 %, 12 configurations; float64: $7.00\pm0.14$ at $(k,d)=(1,2)$) — **not** proportional to the conjectured $D\,d(d{+}2)/2$ |
| F3 | Linear shape-mode rate | $\gamma \approx 1.0$, geometry-independent → naive AR(1) rate-balance heuristic refuted |
| F4 | Upper window edge | $\eta_{\mathrm{hi}} \approx 2.036$, **universal for $k\ge2$**; $k=1$ is metastable (2.07–2.55, 16 seeds) |
| F5 | Marginal degeneracy | top deterministic transverse exponent $\lambda_{\mathrm{det}}\equiv 0$ across the whole window for $d\ge4$ (direct Lyapunov-level proof of $h_1\equiv0$); contracting ($\gamma\approx1$) for $d=2$; abrupt loss $0\to+0.4$ at $\eta_{\mathrm{hi}}$ |
| F6 | Anchor boundary | non-monotone in $x=\lambda_W p_{\mathrm{fix}}$: no crossing for $x\lesssim0.015$, peak $B_{\max}\approx2.10$ at $x^\ast\approx0.075$, decay $\approx-1.1/x$ — the conjectured $\lambda_W^2 p_{\mathrm{fix}}$ term refuted at 25σ |
| F7 | Hoeffding signature | $\zeta_1^\ast \to 1/4$ at the edge ($0.2558$ at $N=1024$); $\zeta_1=0$ in both pure phases |
| F8 | Validity region of F2 | holds for $d\le16$ at $N\ge64$; at $d\ge24$ a positive $N$-coupling appears ($N$ and $d\to\infty$ do not commute) |
| F9 | Transition character | continuous in the noise-sustained NESS at all measured $\sigma>0$ (both $d=2$ and $d\ge4$); bistability/spinodal exists only deterministically ($\sigma\to0$) |

**Net verdict on the conjecture:** the quadratic-in-$\sigma^2$ *form* is confirmed;
both coefficient laws are refuted and replaced; the marginal-degeneracy class
survives on the noise-shifted sheet ($\lambda_Wp_{\mathrm{fix}}=0$) but is lifted by
anchors; the $d=2$ vs $d\ge4$ dichotomy is confirmed at $\sigma\to0$ via a sharp
Lyapunov signature.

## Repository layout

```
├── diva.py                  # GPU simulator: swarm dynamics, order parameters,
│                            #   exact geodesic U-statistic, Hoeffding machinery,
│                            #   deterministic-Jacobian (marginality) exponents
├── phaseA.py ... phaseG.py  # numbered research phases (each standalone)
├── V1_robustness.py         # verification pass: threshold/T-robustness, error bars
├── analysis_final.py        # figures (results/figs/)
├── REPORT.md                # full results report (13 sections, English)
├── diva_mathematik.tex      # self-contained mathematical write-up (German,
│                            #   compiles with pdflatex)
├── PROBLEMSTELLUNG.md       # the five open theory problems (German), with time
│                            #   estimates, entry points and success criteria
├── HERLEITUNG.md            # complete analytical derivations solving all five
│                            #   problems (German)
├── verify_theory.py         # standalone GPU/CPU verification of the analytical
│                            #   predictions (results/theory_verification.json)
├── DiVa                     # raw research log of the autonomous session
│                            #   (includes raw pipeline output; unpolished)
└── results/
    ├── *.json               # all raw measurement data
    └── figs/                # generated figures
```

## Requirements

- Python 3.11+
- PyTorch with CUDA (developed on torch 2.15.0+cu132; CPU should also work at
  reduced speed — set `device="cpu"`)
- numpy, matplotlib (for `analysis_final.py`)

Hardware used for all documented results: one RTX 3060 (12 GB); total compute
≈ 1.5 h. All scripts hard-code their seeds (see `seed=` arguments), so figures
and tables are reproducible bit-for-bit up to GPU nondeterminism.

## Quickstart

```python
import torch
from diva import Swarm

dev = "cuda"
sw = Swarm(B=8, N=64, d=8, k=2, device=dev)      # 8 parallel swarms on Gr(2,8)
sw.set_params(sigma=0.05)                        # tangential noise strength
eta = torch.full((8,), 0.1, device=dev)          # coupling per swarm
for t in range(2000):
    sw.step(eta)
print("mean-field order R =", sw.R_order().mean().item())   # ~1.0 (ordered)
print("Delta_T =", sw.Delta().mean().item())                # ~0 (synchronized)
zr, zc, kv = sw.hoeffding()                                 # first Hoeffding coefficient
```

Reproduce the full study:

```bash
python phaseA.py      # baseline dichotomy map
python phaseA2.py     # hysteresis loops
python phaseB2.py     # eta_lo(sigma) precision edges
python phaseBC.py     # a_sigma(k,d) on 12 configurations + anchor surface
python phaseCD.py     # dense (eta, lambda_W, p_fix) NESS surfaces, up/down
python phaseE2.py     # deterministic one-step Lyapunov exponents
python phaseF.py      # conditional zeta1, two-init bistability, extra (k,d)
python phaseG.py      # float64 spinodals, validity probes
python V1_robustness.py
python analysis_final.py
python verify_theory.py   # analytical derivations & smoking-gun checks (Problems 1–5)
```

Results are written to `results/` as JSON.

## Documentation

| Document | Language | Content |
|---|---|---|
| `REPORT.md` | EN | complete results: 13 sections incl. verification table and verdict per conjecture element |
| `diva_mathematik.tex` | DE | self-contained mathematical development: model, Hoeffding theorem + estimator proof, empirical theorems, analytical derivations |
| `PROBLEMSTELLUNG.md` | DE | the five open theory problems (now solved in `HERLEITUNG.md`) |
| `HERLEITUNG.md` | DE | complete analytical derivations solving all five problems (Boltzmann equilibrium, Weingarten instability, Hoeffding $1/4$ & finite-$N$ bias, $d$-dichotomy, anchor boundary $B(x)$) |
| `DiVa` | EN/DE | raw research log of the autonomous session |

## Status and solved theory problems

Both the numerical program and the analytical derivations for all five problems in `PROBLEMSTELLUNG.md` are complete (see `HERLEITUNG.md`, Section 12 of `diva_mathematik.tex`, and `verify_theory.py`):

1. **$\zeta_1^\ast \to 1/4$ (solved):** Exact isotropic tangent-Gaussian identity $\zeta_1(D) = 1/4 - O(D^{-1})$ (plus compact Boltzmann tail $\zeta_1^\infty = 0.2588$ at $(2,8)$), with the $N$-trend ($0.217 \to 0.2558$) explained by the exact $O(N^{-1})$ estimator bias $-5\zeta_1^\infty/N$ of `zeta_corr`.
2. **Anchor boundary $B(x)$ (solved):** Lower envelope of the rising branch $\varepsilon(\eta;\sigma) = x$ (from omitted `gs_orth` on the output in `phaseE2.py`, giving $x_c \approx 0.015$) and the discrete flip-instability branch $\eta R_{\mathrm{eff}} + x = 2 \Rightarrow B(x) = (2-x)/R_{\mathrm{eff}}$ ($B' \approx -1.10$, peak $B_{\max}\approx 2.10$ at $x^\ast \approx 0.075$).
3. **$d$-dichotomy & $\eta_{\mathrm{hi}} = 2$ (solved):** Exact linearization $U_i' = (1-\eta)U_i + \eta\bar{U}$ is $(k,d)$-independent (giving $\gamma=1$ and $\eta_{\mathrm{hi}}=2$); the apparent $d$-dichotomy in `phaseG.py` arises from per-agent unit normalization on $S^{D-1}$ (disconnected for $D=1$, alignment threshold $\eta_\ast(D)=2D/(D+1)$ for $D\ge 2$).
4. **Origin of $a_\sigma \approx 3d+k$ & F8 (solved):** Exact small-step Gibbs–Boltzmann equilibrium $\rho_N \propto \exp(\frac{a}{2N}\sum_{i,j}\mathrm{tr}(P_i P_j))$ gives $q(1-q) = d/(2a)$ ($D$ cancels identically); $a_\sigma$ is the half-plateau equilibrium $a_{1/2}^{\mathrm{eq}}(k,d)$, strictly symmetric under Grassmannian duality $k \leftrightarrow d-k$ (approximated by $3d+\min(k,d-k)$), and F8 is Kramers nucleation over the first-order barrier.
5. **Source-theory constant $D\,d(d+2)/2$ (solved):** Exact Weingarten linear instability of the Haar state is $a_{\mathrm{lin}}(k,d) = d(d-1)(d+2)/(2D)$, which equals $d(d+2)/2$ at $k=1$; the source constant labels the linear spinodal $a_{\mathrm{lin}}$, not $\eta_{\mathrm{lo}}$ or $\eta_{\mathrm{hi}}$.

## Citation

If you use this code or the measured laws, please cite this repository:

```
Emilio, "DiVa: Dichotomy under Anchor Interpolation — numerical verification
of a conjectured critical surface for subspace-swarm synchronization",
https://github.com/Emilio942/DiVa, 2026.
```

## License

Apache-2.0 — see `LICENSE`.
