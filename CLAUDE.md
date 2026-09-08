# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Analysis tools for a Master's thesis studying finite-size scaling of a 2D bilayer spin system (quantum Monte Carlo simulations) with power-law-decaying interactions parameterized by `alpha` (equivalently `sigma = alpha - 2`). The raw simulation output is produced by an external, non-Python QMC code and lives outside this repo (e.g. under `/home/daniel/Master_thesis/Data/...`, see `pathLeo4` in `parameters.py`) — this repo only loads, fits, and plots that data.

## Environment & commands

- A venv already exists at `.venv/`. Activate with `source .venv/bin/activate`; install deps with `pip install -r requirements.txt` (numpy, scipy, pandas, matplotlib, sympy, jupyter/jupyterlab).
- There is no lint or test tooling configured (no pyproject.toml, no pytest/ruff config). `test_anything.py` is a scratch file, not a real test suite — don't treat it as one or try to "run the tests".
- Most work happens in Jupyter notebooks (`analysis.ipynb`, `plotting.ipynb`, `FiniteSizeScaling.ipynb`, `Data_analysis.ipynb`, `k2realspace.ipynb`, `Plots_Data.ipynb`) that import the `.py` modules in this directory. The `.py` files are libraries, not entry points — only `FS3.py` and `lattice2D.py` have `__main__` guards, and those just print a version banner.
- To execute a notebook headlessly: `jupyter nbconvert --to notebook --execute <name>.ipynb`.
- Notebooks typically start with `%load_ext autoreload` / `%autoreload 2`, so edits to the `.py` modules take effect without restarting the kernel.

## Architecture

### Data flow

1. Simulation output on disk is organized as `<run_dir>/L<L>/<g_param>/`, where `g_param` is a temperature- or `J_perp`-labeled folder containing `config.toml`, `output_files/output.csv` (binned time series: energy, mag, mag2, mag4, spin_stiffness, susceptibility, ...), `output_files/time.csv`, and optionally `output_files/spin_corrX.csv` / `spin_corrY.csv`.
2. `Collected_runs.py`'s `single_run` loads one such run directory, computes jackknife mean/error/bias (`jackknife()`) per observable, and derives the Binder cumulant `U2 = 5/2 * (1 - mag4/(3*mag2**2))`.
3. `collected_runs` aggregates many `single_run`s (globbing `L*/*` under one or more root paths) and indexes them by `L` and by the scan variable `g_param` (`T` if `T0=False`, `J_perp` if `T0=True`). Key methods:
   - `get_x_vs_y_data(xname, yname, xfunc=, yfunc=, ...)` — build a per-`L` x-vs-y table, with sympy-based Gaussian error propagation when `xfunc`/`yfunc` expressions are supplied, and optional plotting.
   - `write_to_FS3_file()` — export to the flat `#<Lindex>` / columnar text format consumed by `FS3.load()`.
   - `select(LRange=, gRange=)` — restrict the collection to a size/parameter window.

### Finite-size scaling toolkit (FS3.py)

General-purpose FSS toolkit, independent of the `Collected_runs` data model: consistent per-`L` plot styling (`setStyle`, `plotStyle`, `errorbarStyle` — `setStyle(Llist=...)` must be called before the others are used with a given `L`), `.dat` file I/O (`load`/`select`), symbolic error propagation (`errorPropagation`, `rescaleAxis`), least-squares/BFGS fitting (`fit_leastsq`, `fit_minimize`, `fit_precond`) against a fit-function object, and crossing-point/data-collapse helpers.

Fit-function classes (in `Collected_runs.py`, mirrored in `Plot_definitions.py`: `f0`, `f1`, `f_fixed_nu`, `Rg0`, `Rg1`, `f_beta_nu`, `f0_dyn_z`, `f_general`) all follow the same informal protocol: `self.vars` (free parameter names), `self.polyOrder`, `nparams()`, `unpack(params)`, `func(x, L, params)`. `fitfunc` in `Collected_runs.py` generalizes this to an arbitrary sympy-string scaling ansatz with optionally-fixed parameters — prefer extending/using that over adding another hardcoded class.

### Duplicated modules — check both before editing shared logic

- `Collected_runs_copy.py` is a **diverged fork**, not a stale backup, of `Collected_runs.py` (e.g. its `scatter_histograms` differs). `Binder_Hist.py`, `eta_hist.py`, and `z_Hist.py` import the copy (`import Collected_runs_copy as cr`), while `plotting.ipynb` and others import the original (`import Collected_runs as cr`). When fixing shared logic (`jackknife`, `single_run`, `collected_runs`), check whether the same fix is needed in the copy.
- `Plot_definitions.py` is a legacy predecessor of `Collected_runs.py`'s fit-function/plotting code. The actual plotting routines (`plot_Binder`, `Binder_Hist`, `plot_Spin_stiffness`, `mag2_U`, `Eta_Hist`, `data_collapse`, `datacollapse`, ...) sit inside one large triple-quoted string at the end of the file and are not executable — treat this as reference/dead code, not a live API.

### Other modules

- `parameters.py` hardcodes experiment metadata: the data root (`pathLeo4`), per-alpha fit windows (`gRange`, `LRange`), and critical-exponent seed estimates (`gc_estimate`, `nu_estimate`, `z_estimate`, `twoBetaNuInv_estimate`) for the `T`-driven and `J_perp`-driven data sets, exposed as `SELECTPARAMS_T` / `SELECTPARAMS_J` plus `HISTPARAMS_*` dataclasses. Extend this file (rather than hardcoding numbers in notebooks) when adding a new `alpha` or data set.
- `lattice2D.py` is a standalone 2D Bravais-lattice/Brillouin-zone geometry module (Wigner-Seitz cell, reciprocal lattice, bond lists for chain/square/honeycomb/triangular lattices) — unrelated to the QMC data pipeline, used by `k2realspace.ipynb`.
- `Binder_Hist.py`, `eta_hist.py`, `z_Hist.py`, `Histograms_Thomas.py` implement the bootstrap-histogram workflow: refit a scaling function many times over `resample()`-perturbed data to build a distribution of critical exponents (`gc`, `nu`, `eta`, `z`, `omega`, ...). Results are cached as `.pkl` (via `Histograms_Thomas.save`/`load`, pickle-based) and `.csv` (e.g. `bootstrap_binderJ_alpha*.csv`) for reuse across notebooks; these result files are not covered by `.gitignore`, so check before committing them.
- `visualize_connections.py`, `visualize_dispersion.py`, `viusalize_loop_update.py` (filename typo — leave as-is unless asked to rename) are standalone illustrative scripts (long-range bond visualization, dispersion relation, SSE loop-update cartoon) with no dependencies on the rest of the codebase.

## Conventions

- `g_param` denotes whichever variable is being scanned (`T` or `J_perp`, chosen via the `T0` flag on `single_run`/`collected_runs`); `alpha = sigma + 2` is the power-law exponent; `N = (1 + bilayer) * Lx * Ly`.
- `*.csv`, `*.pdf`, and `*.txt` are gitignored (see `.gitignore`), so generated plots, data tables, and even `requirements.txt` itself are untracked local files, not part of the repo history — don't assume `git log`/`git diff` will show changes to them.
