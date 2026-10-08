"""Regenerate the three stale journal figures from current results.

Quantities span many orders of magnitude, so these are dot plots on a log axis
rather than bars: a bar encodes magnitude by length from zero, and a log axis
has no zero, which makes the fill length arbitrary.
"""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated slots 1,2,3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e8e7e3"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})

rows = list(csv.DictReader(open("experiments/diagnostic_results.csv")))
DISTS = ["gaussian", "normal_normal", "high_dimensional_gaussian", "banana"]
NICE = {"gaussian": "Gaussian", "normal_normal": "Normal-Gamma",
        "high_dimensional_gaussian": "High-Dim Gaussian", "banana": "Banana"}
MODELS = ["HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
          "SymplecticNNgHMC", "GSymplecticNNgHMC",
          "GradSymplecticNNgHMC", "GradGSymplecticNNgHMC"]
LABEL = {"HMC": "HMC (exact)", "NNgHMC": "NNgHMC", "NNODEgHMC": "NNODEgHMC",
         "Explicit NNODEgHMC": "HNNODEgHMC", "SymplecticNNgHMC": "LA-SympNet",
         "GSymplecticNNgHMC": "G-SympNet", "GradSymplecticNNgHMC": "LA-SympNet+grad",
         "GradGSymplecticNNgHMC": "G-SympNet+grad"}
def family(m):
    if m == "HMC": return "exact"
    return "flow" if "Symplectic" in m else "integrator"
FCOL = {"exact": MUTED, "integrator": ORANGE, "flow": AQUA}

def _num(s):
    """Some columns store a stringified numpy array (per-sample values)."""
    s = s.strip()
    if s.startswith("["):
        v = [float(t) for t in s.strip("[]").replace("\n", " ").split()]
        return float(np.mean(v)) if v else np.nan
    try: return float(s)
    except ValueError: return np.nan

def at_full(m, d, col):
    v = [_num(x[col]) for x in rows if x["model"] == m and x["distribution"] == d
         and abs(float(x["training_size"]) - 1.0) < 1e-9]
    return v[0] if v else np.nan

def dotfig(value_of, xlabel, title, outfile, floor):
    fig, axarr = plt.subplots(2, 2, figsize=(9.6, 5.4), sharey=True)
    axes = axarr.ravel()
    ys = np.arange(len(MODELS))
    for ax, d in zip(axes, DISTS):
        vals = [max(value_of(m, d), floor) for m in MODELS]
        lo = min(v for v in vals if np.isfinite(v))
        for y, (m, v) in enumerate(zip(MODELS, vals)):
            invalid = family(m) == "flow" and at_full(m, d, "reversibility_error") > 1e-6
            col = FCOL[family(m)]
            ax.plot([lo, v], [y, y], color=GRID, lw=1.0, zorder=1)   # leader, not a fill
            if invalid:
                ax.plot(v, y, "o", ms=7, mfc="white", mec=col, mew=1.8, zorder=2)
            else:
                ax.plot(v, y, "o", ms=7, color=col, mec="white", mew=1.0, zorder=2)
        ax.set_title(NICE[d], fontsize=9); ax.set_xlabel(xlabel); ax.set_xscale("log")
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(MODELS)-0.3)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    for a in (axes[0], axes[2]):
        a.set_yticks(ys); a.set_yticklabels([LABEL[m] for m in MODELS])
    h = [plt.Line2D([],[], marker="o", ls="", ms=7, color=FCOL[k], mec="white")
         for k in ("exact","integrator","flow")]
    h.append(plt.Line2D([],[], marker="o", ls="", ms=7, mfc="white", mec=AQUA, mew=1.8))
    fig.legend(h, ["exact HMC","integrator-based surrogate","learned flow map",
                   "flow map: invalid proposal (rev. err $>10^{-6}$)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.06))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    plt.tight_layout(rect=[0,0.04,1,0.94])
    plt.savefig(outfile, bbox_inches="tight", dpi=200); plt.close(fig)
    print("wrote", outfile)

dotfig(lambda m,d: at_full(m,d,"ess")/at_full(m,d,"time"),
       "ESS / second (log)",
       "Effective samples per second at the full training budget",
       "experiments/ess_stats.png", 1e-3)
dotfig(lambda m,d: at_full(m,d,"reversibility_error"),
       "reversibility error (log)",
       "Reversibility: integrator-based methods are exact by construction; unsymmetrized flow maps are not",
       "experiments/reversibility_stats.png", 1e-17)
dotfig(lambda m,d: abs(at_full(m,d,"hamiltonian_error")),
       "mean relative $|\\Delta H|$ (log)",
       "Hamiltonian conservation along proposal trajectories",
       "experiments/hamiltonian_conservation.png", 1e-6)
