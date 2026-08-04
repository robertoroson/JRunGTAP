# JRunGTAP

A Julia reimplementation of the **GEMPACK/RunGTAP** workflow for the standard **GTAP CGE model**, supporting both **version 6.2** (the traditional model) and **version 7** (Corong et al. 2017).

The models are described in:
- Hertel & Tsigas (1997), "Structure of the Standard GTAP Model", Ch. 2 in Hertel (ed.) *Global Trade Analysis*, Cambridge University Press. *(v6.2)*
- McDougall (2003), "A New Regional Household Demand System for GTAP", GTAP Technical Paper No. 20. *(CDE demand system, both versions)*
- Corong, E., Hertel, T., McDougall, R., Tsigas, M. & van der Mensbrugghe, D. (2017), "The Standard GTAP Model, Version 7", *Journal of Global Economic Analysis* 2(1): 1–119. *(v7)*

---

## Features

- Supports **GTAP model v6.2 and v7** — select via `model = v62` or `model = v7` in the config file.
- Reads data directly from a **GTAPAgg3** zip archive (`.har` files).
- Four solution methods:
  - **Johansen** — single linearisation step (fast; accurate for small shocks).
  - **Euler** — multi-step, rebuilds the Jacobian at each sub-step (1st-order accuracy).
  - **Gragg** — modified midpoint method (2nd-order accuracy; ~2× the cost of Euler).
  - **Gragg + Richardson extrapolation** — automatic when `steps` is a multiple of 3 and ≥ 6; runs three Gragg passes at [n, 2n, 3n] steps and extrapolates h²→0 (polynomial order 2, substantially higher effective accuracy).
- Flexible **closure** via variable swaps in the experiment config file.
- Exports results to **CSV**.
- On-screen summary with welfare table (utility, income, CPI, equivalent variation in million USD) and validation checks (Walras' law, income–expenditure balance, factor market clearing).

### GTAP v7 structural additions
Version 7 introduces several key model extensions relative to v6.2:
- **Activities separate from commodities**: production activities (ACTS) are distinct from traded commodities (COMM), linked by a CET/CES *make matrix* (MAKES/MAKEB).
- **Three endowment types**: mobile, sluggish, and fixed endowments, controlled by the ENDOWFLAG matrix.
- **Additional nesting**: composite intermediates with ESUBC (substitution among intermediates) and ESUBT (VA vs. intermediates).
- **Activity-specific taxes**: output (`to`), endowment income (`tinc`), and firm-demand (`tfe`) taxes.
- **Government CES demand**: substitution among government expenditure items with elasticity ESUBG(r).

---

## Repository contents

| File | Description |
|---|---|
| `jrungtap.jl` | Main entry point — load this file in Julia |
| **v6.2 model** | |
| `gtap_v62.jl` | Sets, data structs, derived coefficients, equation residuals (v6.2) |
| `gtap_pack.jl` | Pack/unpack the endogenous variable vector; `F_core` residual function |
| `gtap_analytical.jl` | Analytical sparse Jacobian builder (`build_A_analytical`) |
| `gtap_euler.jl` | Level-update for multi-step Euler/Gragg integration (v6.2) |
| `load_gtapAgg3.jl` | HAR file loader for GTAPAgg3 v6.2 zip archives |
| **v7 model** | |
| `gtap_v7.jl` | Sets, data structs, derived coefficients, equation residuals (v7) |
| `gtap_pack_v7.jl` | Pack/unpack for v7; `F_core_v7` residual function |
| `gtap_analytical_v7.jl` | Cached sparsity-pattern + compressed finite-difference Jacobian builder (v7) |
| `gtap_jacobian_analytical_v7.jl` | Direct analytical Jacobian builder for v7 (< 1 s; used by default) |
| `gtap_euler_v7.jl` | Level-update for multi-step integration (v7) |
| `load_gtapAgg3_v7.jl` | HAR file loader for GTAPv7 zip archives |
| **Shared** | |
| `gtap_solver.jl` | Solution accessor (`get_variable`) |
| `gtap_names.jl` | Name-to-index resolution; config parser (shared) |
| `gtap_names_v7.jl` | v7-specific overloads for name resolution |
| `har_reader.py` | Pure-Python HAR reader (called via PyCall) |
| `template.cfg` | Template experiment config file with documentation |
| `variables.txt` | Reference list of all endogenous and exogenous variables (v6.2) |

---

## Requirements

### Julia packages

JRunGTAP uses only **one external Julia package**. Standard-library packages (`SparseArrays`, `LinearAlgebra`, `Printf`, `Statistics`) are included with Julia and require no installation.

```julia
using Pkg
Pkg.add("PyCall")
```

### Python

`PyCall` bridges Julia to a small Python helper (`har_reader.py`) that reads GEMPACK HAR binary files. You need:

- **Python 3** (any recent version) accessible to PyCall.
- The **NumPy** package:

```bash
pip install numpy
```

If PyCall is not yet configured to use your Python installation, set the environment variable before installing:

```julia
ENV["PYTHON"] = "path/to/python"   # e.g. "C:/Users/you/AppData/Local/Programs/Python/Python313/python.exe"
Pkg.build("PyCall")
```

### Data

JRunGTAP reads data from a **GTAPAgg3** zip archive. This archive is produced by the GTAPAgg3 aggregation software and must contain:

- `sets.har` — set definitions
- `basedata.har` — benchmark flow data
- `default.prm` — elasticity parameters

GTAP data are proprietary and distributed separately by the [GTAP Center](https://www.gtap.agecon.purdue.edu/).

---

## Quick start

```julia
include("jrungtap.jl")
run_gtap_interactive()
```

The interactive runner will prompt you for:
1. The path to your GTAPAgg3 zip file.
2. The name of an experiment (reads `<name>.cfg`, writes `<name>.csv`).

---

## Experiment config file

Each experiment is described by a plain-text `.cfg` file. Copy and edit `template.cfg`:

```
# Method and accuracy
method = euler        # johansen | euler | gragg
steps  = 10           # number of sub-steps (euler and gragg only)

# Shocks: variable[index1, index2, ...] = percentage-point change
shock  tm[rice, Japan] = 50.0       # 50 pp tariff on rice imports into Japan

# Closure swaps (optional): endogenous <-> exogenous
# swap   rental[Japan] <-> qo_endw[Capital, Japan]   # long-run capital

# User-defined sets (optional)
# set agri = [rice, wheat, grains]
```

Variable names, index conventions, and common closure swaps are documented in `variables.txt`.

---

## Programmatic use — v7

```julia
include("jrungtap.jl")

# Load GTAPv7 data
s7, d7, C7 = load_data_v7("mydata_v7.zip")

# Build a simple tariff experiment
exp = gtap_experiment("tariff_v7"; shocks = Dict("tm[1,1]" => 10.0), method = :euler, steps = 10)
sol7 = run_gtap_v7(exp, "mydata_v7.zip")

# Extract results
pds = get_result_v7(sol7, :pds)   # domestic commodity prices (nC × nR)
qo  = get_result_v7(sol7, :qo)    # activity output (nA × nR)
```

Or via config file with `model = v7`:
```
model  = v7
method = euler
steps  = 10
shock  tm[1,1] = 10.0
```

## Programmatic use — v6.2

```julia
include("jrungtap.jl")

# Load data
s, d, C = load_data("mydata.zip")

# Parse a config file
method, steps, raw_shocks, raw_swaps, user_sets = parse_config("myexp.cfg")
shocks, swaps = resolve_config(raw_shocks, raw_swaps, user_sets, s)

# Run
exp = gtap_experiment("mydata.zip"; shocks=shocks, swaps=swaps, method=method, steps=steps)
sol = run_gtap(exp)

# Inspect and export
show_results(sol)
export_results(sol, "myexp.csv")
```

---

## Solution methods

| Method | Config keyword | `steps` | Accuracy | Jacobian builds |
|---|---|---|---|---|
| Johansen | `johansen` | — | O(shock) | 1 (total) |
| Euler | `euler` | N | O(1/N) | N |
| Gragg (modified midpoint) | `gragg` | N | O(1/N²) | 2N |
| Gragg + Richardson | `gragg` | 6, 9, 12, … | O(1/N⁴)+ | 2(n+2n+3n)=12n |

**Richardson extrapolation** is activated automatically when `steps` is a multiple of 3 and ≥ 6. The solver runs three Gragg passes at n, 2n, and 3n sub-steps (where n = steps/3) and combines them via barycentric Lagrange interpolation in 1/n² to eliminate the leading O(h²) error term. With `steps = 6` (passes at 2, 4, 6 steps) the result is generally more accurate than plain Gragg with `steps = 12`.

For small shocks (< 5 pp), Johansen is adequate. For large shocks, `gragg` with `steps = 6` or `steps = 9` (Richardson) is recommended.

### v7 Jacobian: direct analytical builder

The default Jacobian builder (`gtap_jacobian_analytical_v7.jl`) constructs the GTAPv7 system matrix analytically by reading each coefficient directly from the data. Build time is under 1 second for any dataset size. The result is cached to disk next to the data zip and reused on subsequent runs.

`gtap_analytical_v7.jl` provides an alternative builder that caches the sparsity pattern on the first call and subsequently uses **greedy column colouring** with compressed finite differences. Columns that share no common row are probed simultaneously in a single residual evaluation. Because the model is exactly linear in percentage changes, compressed FD is numerically exact. Typical GTAP datasets reach 20–50 colour groups, giving a 2000–5000× speedup over column-by-column probing for subsequent Euler/Gragg sub-steps.

---

## Welfare output

The on-screen summary reports, for each region:

- `welfare(u)` — percentage change in utility.
- `income(y)` — percentage change in regional household income.
- `CPI(p)` — percentage change in the consumer price index.
- `EV (mn USD)` — equivalent variation in million USD at benchmark prices, computed as `EV[r] = INCOME[r] × u[r] / 100` (exact for Johansen; a good approximation for Euler/Gragg).

A **WORLD** row shows the sum of regional EVs (global welfare effect).

---

## Data loading notes

### HAR reader: 14-element arrays

`har_reader.py` reads GEMPACK binary HAR files via PyCall. A subtle bug was fixed: GEMPACK data records whose payload is exactly 56 bytes (14 × float32) produce a total record length of 64 bytes, which coincides with the length of GEMPACK inter-page header records. The old reader skipped all 64-byte records unconditionally. The fix parses the page-structure record to determine `elements_per_page` and skips 64-byte records **only at genuine page boundaries**. Arrays affected in standard GTAP datasets with 14 regions: `VKB`, `VDEP`, `SAVE`, `RFLX`, `RDLT`.

### RORFLEX / RORDELTA

The standard GTAP default is `RORFLEX = 10.0` (rate-of-return flexibility) and `RORDELTA = 1` (ROR equalization). Both are read directly from `default.prm` in the data zip when present; the fallback is 10.0 / 1 respectively.

### VKB, VDEP, SAVE

Capital stock (`VKB`), depreciation (`VDEP`), and net savings (`SAVE`) are read from `basedata.har` when available. Estimated fallbacks are used only when these headers are absent from the data file.

---

## License

This code is released for academic and research use. The underlying GTAP model structure and data are subject to the terms of the [GTAP Data Base license](https://www.gtap.agecon.purdue.edu/databases/default.asp).
