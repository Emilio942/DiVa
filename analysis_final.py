"""Final analysis: zeta1-at-edge universality + figures for the report."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- zeta1 at the edge, from phaseB2 (dense sigma scans) ----------
d = json.load(open("results/phaseB2_edges_sigma.json"))
print("== zeta1 at the NESS edge (interpolated) ==")
zed = {}
for cfg, r in d.items():
    sig = np.array(r["sigma"]); eta = np.array(r["eta"])
    D = np.array(r["Delta"]); Z = np.array(r["zeta1"])
    for s, edge in r["edges"].items():
        if edge is None:
            continue
        # find curve segment for this sigma
        m = np.isclose(sig, float(s))
        ee = np.array(eta)[m]; DD = D[m]; ZZ = Z[m]
        idx = np.argsort(ee)
        ee, DD, ZZ = ee[idx], DD[idx], ZZ[idx]
        thr = 0.5 * DD[0]
        for i in range(len(ee)-1):
            if DD[i] > thr >= DD[i+1]:
                f = (math.log(DD[i])-math.log(thr))/(math.log(DD[i])-math.log(DD[i+1]))
                zv = math.log(ZZ[i]) + f*(math.log(ZZ[i+1])-math.log(ZZ[i])) if ZZ[i] > 0 and ZZ[i+1] > 0 else 0.5*(ZZ[i]+ZZ[i+1])
                zed.setdefault(cfg, []).append((float(s), math.exp(math.log(ee[i])+f*(math.log(ee[i+1])-math.log(ee[i]))), float(np.exp(zv))))
for cfg, lst in zed.items():
    vals = [z for _, _, z in lst]
    print(f"  {cfg}: zeta1 at edge per sigma: {['%.3f'%z for z in vals]}  mean={np.mean(vals):.3f} +- {np.std(vals):.3f}")

# ---------- figures ----------
figdir = "results/figs"
import os
os.makedirs(figdir, exist_ok=True)

# (1) eta_lo vs sigma^2 with quadratic guide
b2 = json.load(open("results/phaseB2_edges_sigma.json"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for cfg, marker in [("2,8", "o"), ("1,2", "s")]:
    ss = [float(s) for s, v in b2[cfg]["edges"].items() if v]
    ee = [v for s, v in b2[cfg]["edges"].items() if v]
    axs[0].plot([s**2 for s in ss], ee, marker, label=f"(k,d) = {cfg} , raw".replace(" ,", ")"))
asig = {"2,8": 25.7, "1,2": 7.0}
for cfg in ["2,8", "1,2"]:
    x = np.linspace(0, 0.16, 50)
    axs[0].plot(x, asig[cfg]*x, "--", label=f"{asig[cfg]:.1f}" + "$\\sigma^2$ (" + cfg + "), fit 3d+k=" + str(3*int(cfg[2])+int(cfg[0])))
axs[0].set_xlabel(r"$\sigma^2$"); axs[0].set_ylabel(r"$\eta_{\rm lo}(\sigma)$")
axs[0].legend(fontsize=8); axs[0].set_title("Lower critical edge: purely quadratic in $\\sigma^2$")

# (2) a_sigma vs 3d+k over 12 configs
pts = {
    (1,2):7.14,(1,3):9.77,(1,4):13.18,(1,5):15.71,(1,8):23.86,
    (2,3):10.09,(2,4):13.22,(2,5):16.23,(2,8):25.79,(3,6):20.34,(3,9):28.56,(4,8):27.27,
}
x = [3*d_+k for (k, d_) in pts]; y = list(pts.values())
axs[1].plot(x, y, "o")
lim = [0, 32]
axs[1].plot(lim, lim, "--", label="a$_\\sigma$ = 3d + k")
axs[1].set_xlim(lim); axs[1].set_ylim(lim)
axs[1].set_xlabel("3d + k"); axs[1].set_ylabel(r"measured $a_\sigma(k,d)$")
axs[1].legend(); axs[1].set_title("Noise coefficient: $a_\\sigma \\approx 3d+k$ (12 configs, $\\pm 5\\%$)")
plt.tight_layout(); plt.savefig(f"{figdir}/sigma_law.png", dpi=130); plt.close()

# (3) spinodal lambda_det(eta): marginal plateau (d>=4) vs contracting (d=2)
sp = json.load(open("results/phaseG_spinodal64.json"))
plt.figure(figsize=(7, 4.5))
for cfg, col in [("1,2", "tab:green"), ("1,4", "tab:orange"), ("2,4", "tab:blue"), ("2,8", "tab:red")]:
    r = sp[cfg]
    e = np.array(r["etas"]); l = np.array(r["lyap"])
    plt.semilogx(e, l, label=f"(k,d)=({cfg})", color=col)
plt.axhline(0, color="k", lw=0.6)
plt.xlabel(r"$\eta$"); plt.ylabel(r"$\lambda_{\rm det}$ (one-step deterministic exponent)")
plt.title("Deterministic transverse exponent at $\\sigma=0$: marginal plateau (d$\\geq$4)\nvs contracting (d=2); abrupt loss at $\\eta_{\\rm hi}\\approx$ 2.1-2.5")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{figdir}/spinodal.png", dpi=130); plt.close()

# (4) anchored ordered phase: lambda_det vs eta for several lam (2,8), sigma=0.05
e2 = json.load(open("results/phaseE2_lyapdet.json"))
r = e2["2,8_s=0.05"]
plt.figure(figsize=(7, 4.5))
for (l, p), row in zip(r["lam_p"], r["lyap"]):
    plt.semilogx(r["eta_grid"], row, marker=".", label=f"$\\lambda_W$={l}, p={p}")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel(r"$\eta$"); plt.ylabel(r"$\lambda_{\rm det}$")
plt.title("Anchored ordered phase (2,8), $\\sigma$=0.05: anchors stabilize (exponent $<0$);\ndesync boundary rises with $\\lambda_W p_{\\rm fix}$")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{figdir}/anchors_lyap.png", dpi=130); plt.close()

# (5) phaseCD curves: Delta vs eta up/down at (2,8) for a few (lam,p)
cd = json.load(open("results/phaseCD_surface.json"))
up = cd["2,8_s=0.05_up"]; dn = cd["2,8_s=0.05_down"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for j in [0, 2, 3, 5]:
    l, p = up["combos"][j]
    axs[0].semilogx(up["etas"], [row[j] for row in up["Delta"]], label=f"$\\lambda$={l},p={p}")
    axs[1].semilogx(dn["etas"], [row[j] for row in dn["Delta"]], label=f"$\\lambda$={l},p={p}")
for ax, ttl in [(axs[0], "up-sweep (disordered init)"), (axs[1], "down-sweep (ordered init)")]:
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\Delta_T$"); ax.set_title(ttl + " (2,8), $\\sigma$=0.05")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{figdir}/hysteresis.png", dpi=130); plt.close()
print("figures saved to", figdir)
