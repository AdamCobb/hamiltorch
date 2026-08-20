# Scripts

Figure generation and one-off diagnostics. All are run inside the GPU container
(see `HANDOFF.md` for the invocation); paths are relative to the repo root.

| script | purpose |
|---|---|
| `make_neurreps_figure.py` | builds `paper_neurreps/results.pdf` from `experiments/diagnostic_results_gp.csv` and `experiments/trajectory_length.csv` |
| `make_journal_figures.py` | rebuilds `experiments/{ess_stats,reversibility_stats,hamiltonian_conservation}.png` from `experiments/diagnostic_results.csv` |
| `validate_palette.py` | Python port of the dataviz palette validator (no `node` on this machine). Usage: `python scripts/validate_palette.py "#hex,#hex,..." light all` |

`diagnostics/` holds the scripts behind specific findings, kept so the numbers
quoted in the papers can be reproduced:

- `cost.py` — per-proposal cost, flow map vs integrator surrogate (O(1) vs O(L))
- `drift.py` — RMHMC energy error vs step size (shows the resonance stall at eps=0.025)
- `funnelcal.py` — RMHMC acceptance on the funnel across (eps, softabs)
- `rmhmc_gauss.py` — recovers a known Gaussian; the test that settled the Hbar/2 question
- `rmhmc_verify.py`, `rmhmc_verify2.py` — involutivity / volume preservation of the extended map
