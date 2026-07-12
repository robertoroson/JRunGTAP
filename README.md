# JRunGTAP

A Julia reimplementation of the **GEMPACK/RunGTAP** workflow for the standard **GTAP v6.2** linearised computable general equilibrium (CGE) model.

The model is described in:
- Hertel & Tsigas (1997), "Structure of the Standard GTAP Model", Ch. 2 in Hertel (ed.) *Global Trade Analysis*, Cambridge University Press.
- McDougall (2003), "A New Regional Household Demand System for GTAP", GTAP Technical Paper No. 20.

---

## Features

- Reads data directly from a **GTAPAgg3** zip archive (`.har` files).
- Three solution methods:
  - **Johansen** — single linearisation step (fast; accurate for small shocks).
  - **Euler** — multi-step, rebuilds the Jacobian at each sub-step (1st-order accuracy).
  - **Gragg** — modified midpoint method (2nd-order accuracy; ~2× the cost of Euler).
- Flexible **closure** via variable swaps in the experiment config file.
- Exports results to **CSV**.
- On-screen summary with welfare table (utility, income, CPI, equivalent variation in million USD) and validation checks (Walras' law, income–expenditure balance, factor market clearing).

---

## Repository contents

| File | Description |
|---|---|
| `jrungtap.jl` | Main entry point — load this file in Julia |
| `gtap_v62.jl` | Sets, data structs, derived coefficients, equation residuals |
| `gtap_pack.jl` | Pack/unpack the endogenous variable vector; `F_core` residual function |
| `gtap_analytical.jl` | Analytical sparse Jacobian builder (`build_A_analytical`) |
| `gtap_solver.jl` | Solution accessor (`get_variable`) |
| `gtap_euler.jl` | Level-update for multi-step Euler/Gragg integration |
| `gtap_names.jl` | Name-to-index resolution; config parser |
| `load_gtapAgg3.jl` | HAR file loader for GTAPAgg3 zip archives |
| `har_reader.py` | Pure-Python HAR reader (called via PyCall) |
| `template.cfg` | Template experiment config file with documentation |
| `variables.txt` | Reference list of all endogenous and exogenous variables |

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

## Programmatic use

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

| Method | Config keyword | Accuracy | Jacobian builds per step |
|---|---|---|---|
| Johansen | `johansen` | O(shock) | 1 (total) |
| Euler | `euler` | O(1/N) | 1 per step |
| Gragg (modified midpoint) | `gragg` | O(1/N²) | 2 per step |

For small shocks (< 5 pp), Johansen is adequate. For large shocks, Euler with 10–20 steps or Gragg with 5–10 steps is recommended.

---

## Welfare output

The on-screen summary reports, for each region:

- `welfare(u)` — percentage change in utility.
- `income(y)` — percentage change in regional household income.
- `CPI(p)` — percentage change in the consumer price index.
- `EV (mn USD)` — equivalent variation in million USD at benchmark prices, computed as `EV[r] = INCOME[r] × u[r] / 100` (exact for Johansen; a good approximation for Euler/Gragg).

A **WORLD** row shows the sum of regional EVs (global welfare effect).

---

## License

This code is released for academic and research use. The underlying GTAP model structure and data are subject to the terms of the [GTAP Data Base license](https://www.gtap.agecon.purdue.edu/databases/default.asp).
