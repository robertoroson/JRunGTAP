# JRunGTAP

A Julia reimplementation of the **GEMPACK/RunGTAP** workflow for the **GTAP CGE model**, supporting both the standard **v6.2** and the extended **v7** model structures.

The models are described in:
- Hertel & Tsigas (1997), "Structure of the Standard GTAP Model", Ch. 2 in Hertel (ed.) *Global Trade Analysis*, Cambridge University Press.
- McDougall (2003), "A New Regional Household Demand System for GTAP", GTAP Technical Paper No. 20.
- Corong et al. (2017), "The Standard GTAP Model, Version 7", *Journal of Global Economic Analysis* 2(1), 1–119.

---

## Features

- Reads data directly from a **GTAPAgg3** zip archive (`.har` files) — both v6.2 and v7 formats.
- Three solution methods:
  - **Johansen** — single linearisation step (fast; accurate for small shocks).
  - **Euler** — multi-step, rebuilds the Jacobian at each sub-step (1st-order accuracy).
  - **Gragg** — modified midpoint method (2nd-order accuracy; ~2× the cost of Euler).
- Flexible **closure** via variable swaps in the experiment config file.
- Exports results to **CSV**.
- On-screen summary with welfare table (utility, income, CPI, equivalent variation in million USD) and validation checks (Walras' law, income–expenditure balance, factor market clearing).

---

## Repository contents

### Core (both model versions)

| File | Description |
|---|---|
| `RunGTAP.jl` | Main entry point — `include` this file in Julia |
| `gtap_names.jl` | Name-to-index resolution; config file parser |
| `har_reader.py` | Pure-Python HAR reader (called via PyCall) |
| `template.cfg` | Template experiment config file with documentation |
| `variables.txt` | Reference list of all endogenous and exogenous variables (v6.2 and v7) |

### GTAP v6.2

| File | Description |
|---|---|
| `gtap_v62.jl` | Sets, data structs, derived coefficients, equation residuals |
| `gtap_pack.jl` | Pack/unpack the endogenous variable vector; `F_core` residual function |
| `gtap_analytical.jl` | Analytical sparse Jacobian builder |
| `gtap_solver.jl` | Solution accessor (`get_variable`) |
| `gtap_euler.jl` | Level-update for multi-step Euler/Gragg integration |
| `load_gtapAgg3.jl` | HAR file loader for GTAPAgg3 v6.2 zip archives |

### GTAP v7

| File | Description |
|---|---|
| `gtap_v7.jl` | Sets, data structs, derived coefficients, equation residuals (v7) |
| `gtap_pack_v7.jl` | Pack/unpack endogenous vector; `F_core_v7` residual function |
| `gtap_analytical_v7.jl` | Analytical sparse Jacobian builder for v7 |
| `gtap_euler_v7.jl` | Level-update for multi-step Euler/Gragg integration (v7) |
| `load_gtapAgg3_v7.jl` | HAR file loader for GTAPAgg3 v7 zip archives |

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

**v6.2 archives:**
- `sets.har` — set definitions
- `basedata.har` — benchmark flow data
- `default.prm` — elasticity parameters

**v7 archives** contain additional HAR files for the activity-commodity split, factor supply by activity, and CET allocation parameters.

GTAP data are proprietary and distributed separately by the [GTAP Center](https://www.gtap.agecon.purdue.edu/).

---

## Quick start

```julia
include("RunGTAP.jl")
run_gtap_interactive()
```

The interactive runner will prompt you for:
1. The path to your GTAPAgg3 zip file.
2. The name of an experiment (reads `<name>.cfg`, writes `<name>.csv`).

The model version is selected in the `.cfg` file with `model = v62` (default) or `model = v7`.

---

## Experiment config file

Each experiment is described by a plain-text `.cfg` file. Copy and edit `template.cfg`:

```
# Model version
model  = v7           # v62 (default) | v7

# Method and accuracy
method = euler        # johansen | euler | gragg
steps  = 10           # number of sub-steps (euler and gragg only)

# Shocks: variable[index1, index2, ...] = percentage-point change
shock  tm[rice, Japan] = 50.0       # 50 pp tariff on rice imports into Japan

# Closure swaps (optional): endogenous <-> exogenous
# swap   rental[Japan] <-> qo_endw[Capital, Japan]   # long-run capital (v6.2)
# swap   pe[Capital, Japan] <-> qe[Capital, Japan]    # long-run capital (v7)

# User-defined sets (optional)
# set agri = [rice, wheat, grains]
```

Variable names, index conventions, and common closure swaps are documented in `variables.txt`.

---

## Programmatic use

### v6.2

```julia
include("RunGTAP.jl")

s, d, C = load_data("mydata_v62.zip")
shocks, swaps = resolve_config(["tm[pdr,Austria]" => 10.0], [], Dict(), s)
sol = run_gtap(gtap_experiment("tariff"; shocks=shocks, method=:euler, steps=10))
show_results(sol)
export_results(sol, "tariff.csv")
```

### v7

```julia
include("RunGTAP.jl")

s, d, C = load_data_v7("mydata_v7.zip")
shocks, swaps = resolve_config(["tm[1,1]" => 10.0], [], Dict(), s)
sol = run_gtap_v7(gtap_experiment("tariff_v7"; shocks=shocks, method=:euler, steps=10),
                  "mydata_v7.zip")
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

## Key differences between v6.2 and v7

| Aspect | v6.2 | v7 |
|---|---|---|
| Activities | Production sectors = commodities | Activities (ACTS) separate from commodities (COMM) |
| Endowment mobility | Mobile (ENDWM) + Sluggish (ENDWS) | Mobile (ENDWM) + Sluggish (ENDWS) + Fixed (ENDWF) |
| Factor supply | `qo_endw(E,R)` endogenous (long-run) | `qe(EMS,R)` exogenous (fixed stocks short-run) |
| Output taxes | `to(NS,R)` | `to(C,A,R)` activity-specific |
| Factor taxes | `tf(E,P,R)` | `tinc(E,A,R)` income + `tfe(E,A,R)` demand |
| Tech shifters | `ao(P,R)`, `afe(E,P,R)` | `ao(A,R)`, `afe(E,A,R)`, `aint(A,R)` added |
| Endogenous vars | ~5,800 (10×10×10 agg.) | ~23,900 (10×10×10 agg.) |

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
