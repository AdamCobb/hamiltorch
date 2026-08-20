import csv, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots 1,2,3,7 (all-pairs: CVD dE 9.2, normal 16.3)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.5))

# ── (a) GP benchmark: best ESS/s per model, validity distinguished ────────────
gp = list(csv.DictReader(open("experiments/diagnostic_results_gp.csv")))
best = {}
for x in gp:
    m, rev, ps = x["model"], float(x["reversibility_error"]), float(x["ess"])/float(x["time"])
    valid = rev < 1e-6
    cur = best.get(m)
    if cur is None or ps > cur[0]:
        best[m] = (ps, valid)
label = {"RevGSymplecticNNgHMC": "Rev-SympNet", "RevGradGSymplecticNNgHMC": "Rev-SympNet\n(+grad)",
         "GSymplecticNNgHMC": "SympNet\n(unsymmetrized)", "HMC": "HMC (exact)",
         "NNgHMC": "NNgHMC", "NNODEgHMC": "NNODEgHMC", "Explicit NNODEgHMC": "HNNODEgHMC"}
order = sorted(best, key=lambda m: best[m][0])
ys = np.arange(len(order))
for y, m in zip(ys, order):
    ps, valid = best[m]
    is_flow = m.startswith("Rev")
    col = BLUE if is_flow else (MUTED if m == "HMC" else ORANGE)
    if valid:
        axA.barh(y, ps, 0.62, color=col, edgecolor="white", linewidth=1.2)
    else:
        axA.barh(y, ps, 0.62, facecolor="white", edgecolor=AQUA, linewidth=1.4,
                 hatch="////")
    txt = f"{ps:.0f}" + ("" if valid else "  invalid")
    axA.text(ps + 4, y, txt, va="center", ha="left", fontsize=8,
             color=INK if valid else AQUA)
axA.set_yticks(ys); axA.set_yticklabels([label[m] for m in order])
hmc_ps = best["HMC"][0]
axA.axvline(hmc_ps, color=INK2, ls="--", lw=1.0, zorder=0)
axA.text(hmc_ps, len(order)-0.3, " HMC baseline", fontsize=7.5, color=INK2, va="top")
axA.set_xlabel("ESS / second   (higher is better)")
axA.set_title("(a) GP hyperparameters, $n=500$", loc="left")
axA.set_xlim(0, 215); axA.grid(axis="x", color="#e8e7e3", lw=0.7); axA.set_axisbelow(True)
for s in ("top", "right", "left"): axA.spines[s].set_visible(False)

# ── (b) scaling with trajectory length ───────────────────────────────────────
tl = list(csv.DictReader(open("experiments/trajectory_length.csv")))
Ls = sorted({int(x["L"]) for x in tl})
series = [("RevSympNet[endpoint]", "Rev-SympNet (flow map)", BLUE, "o", "-"),
          ("HMC", "HMC (exact)", VIOLET, "s", "--"),
          ("NNODEgHMC", "NNODEgHMC (integrator)", ORANGE, "^", "-."),
          ("RevSympNet[all]", "Rev-SympNet, $O(L^2)$ pairs", AQUA, "D", ":")]
for key, lab, col, mk, ls in series:
    ys_ = [next((float(x["ess_per_sec"]) for x in tl
                 if int(x["L"]) == L and x["model"] == key), np.nan) for L in Ls]
    axB.plot(Ls, ys_, color=col, marker=mk, ls=ls, lw=2.0, ms=6.5,
             markeredgecolor="white", markeredgewidth=1.0, label=lab)
axB.set_yscale("log"); axB.set_xlabel("trajectory length $L$")
axB.set_ylabel("ESS / second")
axB.set_title("(b) Cost scaling with trajectory length", loc="left")
axB.set_xticks(Ls); axB.set_xticklabels(Ls)
axB.grid(color="#e8e7e3", lw=0.7); axB.set_axisbelow(True)
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.legend(frameon=False, loc="lower left", handlelength=2.6)

plt.tight_layout()
# Write next to the paper when building it, otherwise into figures/ — the code
# archive ships without the paper directory.
outdir = os.environ.get("NEURREPS_FIGDIR") or (
    "paper_neurreps" if os.path.isdir("paper_neurreps") else "figures")
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "results.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(outdir, "results.png"), dpi=200, bbox_inches="tight")
print(f"wrote {outdir}/results.pdf (+ .png preview)")
