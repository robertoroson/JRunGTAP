# jrungtap.jl
# ─────────────────────────────────────────────────────────────────────────────
cd(@__DIR__)   # ensure working directory = project folder regardless of call site
# Julia equivalent of the RunGTAP/GEMPACK workflow for the GTAP v6.2 CGE model.
#
# Quick start — interactive (reads zip, then loops over experiments):
#   include("jrungtap.jl")
#   run_gtap_interactive()          # prompts for zip file, then experiment name
#                                   # config = name.cfg, output = name.csv
#
# Programmatic use:
#   include("jrungtap.jl")
#   s, d, C = load_data("mydata.zip")
#   shocks, swaps = resolve_config(["tm[pdr,Austria]" => 10.0], [], Dict(), s)
#   sol = run_gtap(gtap_experiment("tariff"; shocks=shocks))
#   show_results(sol);  export_results(sol, "tariff.csv")
# ─────────────────────────────────────────────────────────────────────────────

# ── (a) Load model infrastructure ────────────────────────────────────────────

include("gtap_pack.jl")         # F_core, make_exog_zero, n_endog, CORE_EQ_NAMES
include("gtap_analytical.jl")   # build_A_analytical
include("gtap_solver.jl")       # solve_gtap_analytical, get_variable
include("load_gtapAgg3.jl")     # load_gsdf
include("gtap_names.jl")        # var_domains, resolve_config, expand_varspec
include("gtap_euler.jl")        # update_data_euler (multi-step Euler)

using SparseArrays, Printf, LinearAlgebra

# ── Types ─────────────────────────────────────────────────────────────────────

"""
    Swap(endo_out, exog_in; fix_at = 0.0)

A single closure swap: move `endo_out` from endogenous to exogenous (fixing its
% change at `fix_at`), and move `exog_in` from exogenous to endogenous.

Both arguments are strings of the form "varname[i,j,...]", matching the
convention used for shocks.

Examples
    Swap("rental[1]",     "qo_endw[1,1]")   # long-run: fix capital return, free supply
    Swap("walras_dem[1]", "walraslack[1]")   # switch Walras numeraire
"""
struct Swap
    endo_out :: String    # currently endogenous → moves to exogenous
    exog_in  :: String    # currently exogenous  → moves to endogenous
    fix_at   :: Float64   # value to fix endo_out at (default 0.0 = no % change)
end
Swap(endo_out::String, exog_in::String) = Swap(endo_out, exog_in, 0.0)

"""
    GTAPExperiment

Describes a GTAP simulation run — analogous to a GEMPACK `.cmf` file.

Fields
- `name`    : label for output files and display
- `shocks`  : Dict mapping "varname[i,j,...]" → shock value
- `swaps`   : Vector{Swap} defining closure modifications (empty = standard closure)
- `method`  : `:johansen` (default), `:euler`, or `:gragg`
- `steps`   : sub-steps for `:euler` / `:gragg`; ignored for `:johansen`
"""
struct GTAPExperiment
    name   :: String
    shocks :: Dict{String,Float64}
    swaps  :: Vector{Swap}
    method :: Symbol
    steps  :: Int
end

"""
    gtap_experiment(name; shocks, swaps, method, steps) → GTAPExperiment

Convenience constructor with sensible defaults.
"""
function gtap_experiment(name::String;
                          shocks :: Dict{String,Float64} = Dict{String,Float64}(),
                          swaps  :: Vector{Swap}         = Swap[],
                          method :: Symbol               = :johansen,
                          steps  :: Int                  = 10)
    method in (:johansen, :euler, :gragg) ||
        error("method must be :johansen, :euler, or :gragg, got :$method")
    GTAPExperiment(name, shocks, swaps, method, steps)
end

"""
    GTAPSolution

Returned by `run_gtap`. Holds the solution vector and all context for extraction.

- `swap_in`  : col j → "exog_in[i,j]"  — variable that was moved INTO x_endo
- `swap_out` : col j → ("endo_out[i,j]", fix_at) — variable moved OUT, and its fixed value
"""
struct GTAPSolution
    experiment :: GTAPExperiment
    x_endo     :: Vector{Float64}
    swap_in    :: Dict{Int,String}              # col j → spec of newly-endogenous var
    swap_out   :: Dict{Int,Tuple{String,Float64}} # col j → (spec of fixed var, fix_at)
    A          :: Any                           # sparse Jacobian (standard closure)
    s          :: GTAPSets
    d          :: GTAPData
    C          :: Any
end

# ── (b) Data loading ──────────────────────────────────────────────────────────

const _DATA_CACHE     = Ref{Union{Nothing,Tuple}}(nothing)
const _JACOBIAN_CACHE = Ref{Any}(nothing)
const _ZIP_CACHE      = Ref{Union{Nothing,String}}(nothing)   # last zip path used

"""
    load_data(zippath) → (s, d, C)
    load_data(; set_file, dat_file, par_file) → (s, d, C)

Load GTAP data.  The single-argument form reads from a GTAPAgg3 zip archive
(containing sets.har, basedata.har, default.prm).  The keyword form reads
three explicit HAR files and is kept for programmatic/backward-compatible use.

Results are cached — call `reload_data!()` to force a fresh load.
"""
function load_data(zippath::String)
    if _DATA_CACHE[] === nothing || _ZIP_CACHE[] != zippath
        println("Loading data from $(basename(zippath)) …")
        _DATA_CACHE[]    = load_from_zip(zippath)
        _ZIP_CACHE[]     = zippath
        _JACOBIAN_CACHE[] = nothing
        _print_data_summary(_DATA_CACHE[])
    end
    _DATA_CACHE[]
end

function load_data(;
        set_file = "gsdfset.har",
        dat_file = "gsdfdat.har",
        par_file = "gsdfpar.har")
    if _DATA_CACHE[] === nothing
        println("Loading data…")
        _DATA_CACHE[] = load_gsdf(set_file, dat_file, par_file)
        _print_data_summary(_DATA_CACHE[])
    end
    _DATA_CACHE[]
end

function _print_data_summary(sdc)
    s, d, C = sdc
    println("  nT=$(length(s.TRAD_COMM))  nR=$(length(s.REG))  " *
            "nP=$(length(s.PROD_COMM))  n_endo=$(n_endog(s))")
    b0 = F_core(zeros(n_endog(s)), make_exog_zero(s), d, s, C)
    println("  Benchmark ‖F(0)‖∞ = $(maximum(abs, b0))")
end

reload_data!() = (_DATA_CACHE[] = nothing; _JACOBIAN_CACHE[] = nothing;
                  _ZIP_CACHE[] = nothing; nothing)

function _get_jacobian(d, s, C; rebuild = false)
    if _JACOBIAN_CACHE[] === nothing || rebuild
        println("Building analytical Jacobian…")
        t0 = time()
        _JACOBIAN_CACHE[] = build_A_analytical(d, s, C)
        println("  done in $(round(time()-t0, digits=1))s   nnz=$(nnz(_JACOBIAN_CACHE[]))")
    else
        println("  (reusing cached Jacobian)")
    end
    _JACOBIAN_CACHE[]
end

# ── Closure swap mechanics ────────────────────────────────────────────────────

# Parse "varname[i,j,...]" → (:varname, [i,j,...])
function _parse_varspec(spec::String)
    m = match(r"^(\w+)\[([0-9,]+)\]$", spec)
    m === nothing && error("Cannot parse variable spec '$spec' — expected 'name[i,j,...]'")
    Symbol(m[1]), parse.(Int, split(m[2], ','))
end

# Column index in x_endo for an endogenous variable element
function _endo_col(name::Symbol, idxs::Vector{Int}, s::GTAPSets)
    _, offsets, _ = endo_offsets(s)
    specs = Dict(_endo_specs(s))
    haskey(offsets, name) || error("'$name' is not an endogenous variable in this closure")
    shape = specs[name]
    off   = offsets[name]           # 1-based start index in x_endo
    lin   = LinearIndices(shape)[idxs...]
    return off + lin - 1
end

# Compute B[:,k] = ∂F_core/∂exog_k  (one F_core evaluation with unit exog_k)
function _exog_col(name::Symbol, idxs::Vector{Int}, d, s, C)
    exog_base = make_exog_zero(s)
    haskey(pairs(exog_base), name) ||
        error("'$name' is not an exogenous variable in the standard closure")
    exog_dict = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                     for (k, v) in pairs(exog_base))
    exog_dict[name][idxs...] = 1.0
    scalars = (:pfactwld, :walraslack)
    exog = NamedTuple(k => (k in scalars ? exog_dict[k][1] : exog_dict[k])
                      for k in keys(exog_base))
    # B[:,k] = F_core(0, e_k)  since F_core(0,0)=0 and model is linear
    F_core(zeros(n_endog(s)), exog, d, s, C)
end

# Replace column j of a SparseMatrixCSC with a dense/sparse vector
function _replace_col(A::SparseMatrixCSC{Float64,Int}, j::Int,
                      col::AbstractVector{Float64})
    m, n = size(A)
    rows_new = Int[]; cols_new = Int[]; vals_new = Float64[]
    rv = rowvals(A); nzv = nonzeros(A)
    for jj in 1:n
        if jj == j
            for (i, v) in enumerate(col)
                abs(v) > 0 || continue
                push!(rows_new, i); push!(cols_new, jj); push!(vals_new, v)
            end
        else
            for ptr in nzrange(A, jj)
                push!(rows_new, rv[ptr]); push!(cols_new, jj); push!(vals_new, nzv[ptr])
            end
        end
    end
    sparse(rows_new, cols_new, vals_new, m, n)
end

"""
    apply_swaps(A, b, swaps, d, s, C) → (A_swapped, b_swapped, swap_index)

Modify the coefficient matrix and RHS to reflect closure swaps.

For each swap (endo_out → exo at v, exo_in → endo):
  - Column j of A (for endo_out) is replaced by B[:,k] = ∂F/∂exog_in
  - b is adjusted by  b -= A[:,j] * fix_at

Returns modified system, swap_in (col → exog_in spec), swap_out (col → (endo_out spec, fix_at)).
"""
function apply_swaps(A, b, swaps::Vector{Swap}, d, s, C)
    isempty(swaps) && return A, b, Dict{Int,String}(), Dict{Int,Tuple{String,Float64}}()

    A_sw     = A
    b_sw     = copy(b)
    swap_in  = Dict{Int,String}()
    swap_out = Dict{Int,Tuple{String,Float64}}()

    for sw in swaps
        nm_out, idxs_out = _parse_varspec(sw.endo_out)
        nm_in,  idxs_in  = _parse_varspec(sw.exog_in)

        j   = _endo_col(nm_out, idxs_out, s)
        B_k = _exog_col(nm_in,  idxs_in,  d, s, C)

        println("  Swap: $(sw.endo_out) → exo (fix=$(sw.fix_at)) | " *
                "$(sw.exog_in) → endo  [col $j]")

        if sw.fix_at != 0.0
            b_sw .-= A_sw[:, j] .* sw.fix_at
        end

        A_sw = _replace_col(A_sw, j, B_k)
        swap_in[j]  = sw.exog_in
        swap_out[j] = (sw.endo_out, sw.fix_at)
    end

    return A_sw, b_sw, swap_in, swap_out
end

# ── (c) Run experiment ────────────────────────────────────────────────────────

"""
    run_gtap(exp; rebuild_jacobian) → GTAPSolution

Solve the linearised GTAP model for the shocks and closure defined in `exp`.
Data and the standard Jacobian are loaded/built automatically and cached.

Swaps modify the system before solving; the standard Jacobian cache is unchanged
so subsequent experiments with different closures don't pay a rebuild cost.
"""
function run_gtap(exp::GTAPExperiment; rebuild_jacobian = false)
    s, d, C = load_data()
    A       = _get_jacobian(d, s, C; rebuild = rebuild_jacobian)

    println("\n── $(exp.name) " * "─"^max(0, 60 - length(exp.name)))
    meth_str = exp.method === :johansen ? "Johansen" :
               exp.method === :euler   ? "Euler ($(exp.steps) steps)" :
                                         "Gragg ($(exp.steps) steps)"
    println("  Method : $meth_str")
    isempty(exp.swaps)  || println("  Swaps  : $(length(exp.swaps))")
    isempty(exp.shocks) || for (k,v) in sort(collect(exp.shocks))
        println("  Shock  : $k = $v")
    end
    println()

    # Build exog bundle and apply shocks
    exog_dict = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                     for (k, v) in pairs(make_exog_zero(s)))
    for (key, val) in exp.shocks
        m = match(r"^(\w+)\[([0-9,]+)\]$", key)
        m === nothing && (@warn "Cannot parse shock '$key'"; continue)
        fn = Symbol(m[1]); idxs = parse.(Int, split(m[2], ','))
        haskey(exog_dict, fn) ? exog_dict[fn][idxs...] = val :
            @warn "Shock field '$fn' not in exogenous bundle"
    end
    scalars = (:pfactwld, :walraslack)
    exog = NamedTuple(k => (k in scalars ? exog_dict[k][1] : exog_dict[k])
                      for k in keys(make_exog_zero(s)))

    # RHS
    b = -F_core(zeros(n_endog(s)), exog, d, s, C)

    # Apply closure swaps
    A_eff, b_eff, swap_in, swap_out = apply_swaps(A, b, exp.swaps, d, s, C)

    # Solve
    x_endo = if exp.method === :johansen
        _johansen_solve(A_eff, b_eff)
    elseif exp.method === :euler
        _euler_solve(exp.shocks, exp.swaps, exp.steps, d, s, C)
    else
        _gragg_solve(exp.shocks, exp.swaps, exp.steps, d, s, C)
    end

    GTAPSolution(exp, x_endo, swap_in, swap_out, A, s, d, C)
end

# ── Solver back-ends ──────────────────────────────────────────────────────────

function _johansen_solve(A, b)
    lu(A) \ b
end

function _euler_solve(shocks, swaps, steps, d0::GTAPData, s::GTAPSets, C0)
    n       = n_endog(s)
    x_level = ones(n)       # product accumulator; final result = 100*(x_level - 1)
    d_cur   = d0
    C_cur   = C0
    scalars = (:pfactwld, :walraslack)

    for step in 1:steps
        print("\r  step $step/$steps…")

        # Sub-step exogenous bundle: apply 1/steps fraction of each shock
        exog_dict = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                         for (k, v) in pairs(make_exog_zero(s)))
        for (key, val) in shocks
            m = match(r"^(\w+)\[([0-9,]+)\]$", key)
            m === nothing && continue
            fn = Symbol(m[1]); idxs = parse.(Int, split(m[2], ','))
            haskey(exog_dict, fn) && (exog_dict[fn][idxs...] = val / steps)
        end
        exog = NamedTuple(k => (k in scalars ? exog_dict[k][1] : exog_dict[k])
                          for k in keys(make_exog_zero(s)))

        # Rebuild Jacobian and RHS at current benchmark
        A_cur = build_A_analytical(d_cur, s, C_cur)
        b_cur = -F_core(zeros(n), exog, d_cur, s, C_cur)

        # Apply closure swaps
        A_eff, b_eff, _, _ = apply_swaps(A_cur, b_cur, swaps, d_cur, s, C_cur)

        # Sub-step solution
        dx = lu(A_eff) \ b_eff

        # Accumulate via product rule (exact for multiplicative changes)
        x_level .*= (1 .+ dx ./ 100)

        # Update benchmark data for next sub-step
        d_cur = update_data_euler(d_cur, dx, s)
        C_cur = compute_derived(d_cur, s)
    end

    println("\r  done ($steps steps).   ")
    100 .* (x_level .- 1)
end

function _gragg_solve(shocks, swaps, steps, d0::GTAPData, s::GTAPSets, C0)
    # Modified midpoint (Gragg) method.
    #
    # Each sub-step:
    #   1. Solve A(d_cur)*k1 = b(shock/steps, d_cur)  — Euler slope at current benchmark
    #   2. Advance to midpoint:  d_mid = update(d_cur, k1/2)
    #   3. Solve A(d_mid)*k2  = b(shock/steps, d_mid)  — slope at midpoint
    #   4. Accumulate:   x_level *= (1 + k2/100)       — use midpoint slope (2nd-order)
    #   5. Advance benchmark:  d_cur = update(d_cur, k1) — Euler step, d_cur-consistent
    #
    # k2 is used for accumulation (higher-order accuracy); k1 is used for the
    # benchmark update so that value flows remain consistent with d_cur's price
    # structure.  Using k2 for the benchmark update causes catastrophic
    # inconsistency because k2 was linearised at d_mid, not d_cur.
    n       = n_endog(s)
    x_level = ones(n)
    d_cur   = d0
    C_cur   = C0
    scalars = (:pfactwld, :walraslack)

    function _make_exog_sub()
        ed = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                  for (k, v) in pairs(make_exog_zero(s)))
        for (key, val) in shocks
            m = match(r"^(\w+)\[([0-9,]+)\]$", key)
            m === nothing && continue
            fn = Symbol(m[1]); idxs = parse.(Int, split(m[2], ','))
            haskey(ed, fn) && (ed[fn][idxs...] = val / steps)
        end
        NamedTuple(k => (k in scalars ? ed[k][1] : ed[k]) for k in keys(make_exog_zero(s)))
    end

    for step in 1:steps
        print("\r  step $step/$steps…")
        exog_sub = _make_exog_sub()

        # ── k1: Euler slope at current benchmark ──────────────────────────────
        A_cur = build_A_analytical(d_cur, s, C_cur)
        b1    = -F_core(zeros(n), exog_sub, d_cur, s, C_cur)
        A1, b1e, _, _ = apply_swaps(A_cur, b1, swaps, d_cur, s, C_cur)
        k1 = lu(A1) \ b1e

        # ── Midpoint benchmark: advance d_cur by k1/2 ─────────────────────────
        d_mid = update_data_euler(d_cur, k1 ./ 2, s)
        C_mid = compute_derived(d_mid, s)

        # ── k2: midpoint slope — used for accumulation ────────────────────────
        A_mid = build_A_analytical(d_mid, s, C_mid)
        b2    = -F_core(zeros(n), exog_sub, d_mid, s, C_mid)
        A2, b2e, _, _ = apply_swaps(A_mid, b2, swaps, d_mid, s, C_mid)
        k2 = lu(A2) \ b2e

        # ── Accumulate with midpoint slope; advance benchmark with Euler slope ─
        x_level .*= (1 .+ k2 ./ 100)
        d_cur = update_data_euler(d_cur, k1, s)   # k1 is consistent with d_cur
        C_cur = compute_derived(d_cur, s)
    end

    println("\r  done ($steps steps, Gragg).   ")
    100 .* (x_level .- 1)
end

# ── (d) Extract and export results ────────────────────────────────────────────

"""
    get_result(sol, name) → scalar or Array

Extract a variable from the solution, correctly handling all cases:
- standard endogenous  → from x_endo (swapped-out positions patched to fix_at)
- swapped IN           → from the column in x_endo where it now lives
- exogenous (fixed)    → returns fix_at (or 0.0 if never mentioned in swaps)
"""
function get_result(sol::GTAPSolution, name::Symbol)
    specs = Dict(_endo_specs(sol.s))
    _, offsets, _ = endo_offsets(sol.s)

    # Case 1: standard endogenous variable (may have some swapped-out positions)
    if haskey(specs, name)
        arr = copy(get_variable(sol.x_endo, name, sol.s))
        # Patch positions that were swapped OUT — they hold the swapped-in value,
        # not the original variable.  Replace with fix_at.
        off = offsets[name]
        for (j, (spec_out, fix_val)) in sol.swap_out
            nm_out, _ = _parse_varspec(spec_out)
            nm_out == name || continue
            lin = j - off + 1          # 1-based position within this variable's block
            arr[lin] = fix_val
        end
        return arr
    end

    # Case 2: variable was swapped IN (now endogenous, lives at a specific column)
    hits = [(j, spec) for (j, spec) in sol.swap_in
            if _parse_varspec(spec)[1] == name]
    if !isempty(hits)
        length(hits) == 1 && return sol.x_endo[hits[1][1]]
        return Dict(spec => sol.x_endo[j] for (j, spec) in hits)
    end

    # Case 3: purely exogenous — return 0.0 (unshocked) or fix_at if it was swapped out
    for (_, (spec_out, fix_val)) in sol.swap_out
        nm_out, _ = _parse_varspec(spec_out)
        nm_out == name && return fix_val
    end
    return 0.0
end

"""
    export_results(sol, filename; variables, skip_zeros, digits)

Write simulation results to CSV with columns: experiment, variable, indices, pct_change.
Swapped-in variables are labelled with their exogenous name and column position.
"""
function export_results(sol::GTAPSolution, filename::String;
                        variables  :: Union{Nothing,Vector{Symbol}} = nothing,
                        skip_zeros :: Bool = true,
                        digits     :: Int  = 6)
    specs    = _endo_specs(sol.s)
    vars     = variables === nothing ? [nm for (nm, _) in specs] : variables
    dom_orig = var_domains_orig(sol.s)

    # Maximum number of index dimensions across all variables (for column count).
    max_dims = maximum(length(v) for v in values(dom_orig); init = 4)

    # Resolve a CartesianIndex to a Vector of name strings (one per dimension).
    function idx_names(nm::Symbol, idx::CartesianIndex)
        t = Tuple(idx)
        if haskey(dom_orig, nm)
            sets = dom_orig[nm]
            length(sets) == length(t) && return [sets[d][t[d]] for d in eachindex(sets)]
        end
        return [string(i) for i in t]   # fallback: numeric strings
    end

    # Pad a names vector to max_dims with empty strings.
    pad(v) = vcat(v, fill("", max_dims - length(v)))

    rows = 0
    open(filename, "w") do io
        dim_headers = join(["dim$i" for i in 1:max_dims], ",")
        println(io, "experiment,variable,", dim_headers, ",pct_change")
        expname = sol.experiment.name

        for nm in vars
            haskey(dom_orig, nm) || continue   # skip internal vars with no named dims
            try
                arr = get_variable(sol.x_endo, nm, sol.s)
                for idx in CartesianIndices(arr)
                    v = round(arr[idx], digits = digits)
                    skip_zeros && iszero(v) && continue
                    println(io, expname, ",", nm, ",",
                            join(pad(idx_names(nm, idx)), ","), ",", v)
                    rows += 1
                end
            catch
            end
        end

        # Swapped-in variables
        for (j, spec) in sort(collect(sol.swap_in))
            v = round(sol.x_endo[j], digits = digits)
            skip_zeros && iszero(v) && continue
            nm_sym, idxs = _parse_varspec(spec)
            names = if haskey(dom_orig, nm_sym)
                sets = dom_orig[nm_sym]
                length(sets) == length(idxs) ?
                    [sets[d][idxs[d]] for d in eachindex(sets)] :
                    [string(i) for i in idxs]
            else
                [string(i) for i in idxs]
            end
            println(io, expname, ",swapped_in,",
                    join(pad([string(nm_sym); names]), ","), ",", v)
            rows += 1
        end

        # EV (mn USD): post-processed from u and benchmark INCOME
        u_v = try get_variable(sol.x_endo, :u, sol.s) catch; nothing end
        if u_v !== nothing
            ev = sol.C.INCOME .* u_v ./ 100
            for r in 1:length(sol.s.REG)
                v = round(ev[r], digits = digits)
                skip_zeros && iszero(v) && continue
                println(io, expname, ",EV_mn_USD,", join(pad([sol.s.REG[r]]), ","), ",", v)
                rows += 1
            end
        end
    end
    println("Exported $rows non-zero entries → $filename")
end

"""
    show_results(sol)

Print a general-purpose summary of simulation results:
  - Regional welfare (u) and income (y): one row per region
  - Sectoral output (qva): min/max/‖·‖∞ across all sectors and regions
  - Trade volumes (qxs): min/max/‖·‖∞ across all bilateral flows
  - Swapped-in variables (if any closure swaps were applied)
"""
function show_results(sol::GTAPSolution)
    s = sol.s
    println("\n── Results: $(sol.experiment.name) " * "─"^40)

    # ── Regional welfare and income (one line per region) ──────────────────
    u_arr = try get_variable(sol.x_endo, :u, s) catch; nothing end
    y_arr = try get_variable(sol.x_endo, :y, s) catch; nothing end
    p_arr = try get_variable(sol.x_endo, :p, s) catch; nothing end

    if u_arr !== nothing
        # EV[r] = INCOME[r] * u[r] / 100  (million USD at benchmark prices)
        ev_arr = sol.C.INCOME .* u_arr ./ 100

        println("  Region              welfare(u)    income(y)    CPI(p)    EV (mn USD)")
        println("  " * "─"^72)
        for r in 1:length(s.REG)
            u_r  = u_arr[r]
            y_r  = y_arr !== nothing ? y_arr[r] : NaN
            p_r  = p_arr !== nothing ? p_arr[r] : NaN
            ev_r = ev_arr[r]
            @printf("  %-20s %10.4f   %10.4f   %10.4f   %12.2f\n",
                    s.REG[r], u_r, y_r, p_r, ev_r)
        end
        @printf("  %-20s %10s   %10s   %10s   %12.2f\n",
                "WORLD", "", "", "", sum(ev_arr))
    end

    # ── Aggregate summaries for multi-dimensional variables ─────────────────
    println()
    println("  variable   description                  min          max          ‖·‖∞")
    println("  " * "─"^72)
    agg_vars = [
        (:qva, "sectoral value added (%)"),
        (:qxs, "bilateral trade volumes (%)"),
        (:pms, "import prices (%)"),
        (:psave, "savings price (%)"),
    ]
    for (nm, desc) in agg_vars
        try
            v    = get_variable(sol.x_endo, nm, s)
            flat = vec(v)
            @printf("  %-10s %-28s %12.4f  %12.4f  %12.4f\n",
                    nm, desc, minimum(flat), maximum(flat), maximum(abs, flat))
        catch
        end
    end

    # ── Swapped-in variables ────────────────────────────────────────────────
    if !isempty(sol.swap_in)
        println("  " * "─"^72)
        println("  Swapped-in variables (now endogenous):")
        for (j, spec) in sort(collect(sol.swap_in))
            @printf("  %-36s = %12.6f\n", spec, sol.x_endo[j])
        end
    end

    # ── Validation checks ───────────────────────────────────────────────────
    println()
    println("  ── Validation checks " * "─"^49)
    _show_validation(sol)
end

function _show_validation(sol::GTAPSolution)
    s = sol.s; d = sol.d; x = sol.x_endo
    meth = sol.experiment.method

    # Acceptable residual depends on solution method:
    #   Johansen → machine epsilon; Euler/Gragg → linearization error, O(1/steps²)
    tol = meth === :johansen ? 1e-8 : 1e-2 / sol.experiment.steps^2

    _get(nm) = try get_variable(x, nm, s) catch; nothing end
    ok(v) = v < tol ? "✓" : (v < 100tol ? "~" : "⚠")

    # ── 1. Walras' law: global savings supply vs investment demand ───────────
    ws = _get(:walras_sup); wd = _get(:walras_dem)
    if ws !== nothing && wd !== nothing
        walras_gap = abs(ws[1] - wd[1])
        @printf("  Walras' law  (savings – investment gap)     : %11.2e  %s\n",
                walras_gap, ok(walras_gap))
    end

    # ── 2. Regional income–expenditure balance ───────────────────────────────
    # Identity: INCOME * y ≈ VPA * yp + VGA * yg + SAVE * (psave + qsave)
    yv = _get(:y); yp = _get(:yp); yg = _get(:yg)
    psv = _get(:psave); qsv = _get(:qsave)
    if all(!isnothing, (yv, yp, yg, psv, qsv))
        VPA    = vec(sum(d.VDPA .+ d.VIPA, dims=1))
        VGA    = vec(sum(d.VDGA .+ d.VIGA, dims=1))
        INCOME = VPA .+ VGA .+ d.SAVE
        resid  = (VPA .* yp .+ VGA .* yg .+ d.SAVE .* (psv .+ qsv)) ./ INCOME .- yv
        ie_max = maximum(abs, resid)
        @printf("  Income–expenditure balance  (max |residual|) : %11.2e  %s\n",
                ie_max, ok(ie_max))
    end

    # ── 3. Factor market clearing ────────────────────────────────────────────
    # Check: demand-weighted pfe[e,j,r] ≈ ps[e,r] (the economy-wide factor price)
    ps_v  = _get(:ps);  pfe_v = _get(:pfe)
    nE = length(s.ENDW_COMM); nR = length(s.REG)
    if ps_v !== nothing && pfe_v !== nothing && size(d.VFM) == (nE, length(s.PROD_COMM), nR)
        max_fc = 0.0
        for e in 1:nE, r in 1:nR
            total = sum(d.VFM[e, :, r])
            total < 1e-10 && continue
            θ = d.VFM[e, :, r] ./ total
            max_fc = max(max_fc, abs(ps_v[e, r] - dot(θ, pfe_v[e, :, r])))
        end
        @printf("  Factor market clearing       (max |residual|) : %11.2e  %s\n",
                max_fc, ok(max_fc))
    end

    meth !== :johansen &&
        println("  (✓ < $(round(tol,sigdigits=1))  ~  < $(round(100tol,sigdigits=1))  for $(meth)/$(sol.experiment.steps) steps)")
end

# ── Config file parser ───────────────────────────────────────────────────────

"""
    parse_config(filename) → (method, steps, raw_shocks, raw_swaps, user_sets)

Parse a `.cfg` experiment file.  Recognised directives:

    method = johansen           # or euler
    steps  = 10                 # sub-steps for euler

    # Set definitions (inline)
    set  agri = [pdr, wht, gro, v_f]
    set  eu   = [aut, deu, fra, ita, esp]

    # Shocks — indices may be names, set names, 'all'/'*', or integers
    shock  tm[pdr, aut]    = 10.0   # named indices
    shock  tm[agri, all]   = 5.0    # user set × all regions
    shock  tm[TRAD_COMM, aut] = 3.0 # model-set × named region
    shock  tm[1, 1]        = 10.0   # numeric (backward compatible)

    # Closure swaps
    swap  rental[aut] <-> qo_endw[Land, aut]
    swap  rental[aut] <-> qo_endw[Land, aut]  fix_at = 2.0

Lines beginning with `#` and blank lines are ignored; inline comments
after `#` are stripped.

Call `resolve_config(raw_shocks, raw_swaps, user_sets, s)` to convert the
returned raw specs into fully-numeric `Dict` and `Vector{Swap}`.
"""
function parse_config(filename::String)
    isfile(filename) || error("Config file not found: $filename")

    raw_shocks = Pair{String,Float64}[]          # ordered; duplicates OK
    raw_swaps  = Tuple{String,String,Float64}[]
    user_sets  = Dict{String,Vector{String}}()
    method     = :johansen
    steps      = 10

    # Regex for a numeric value (int or float, optional sign/exponent)
    numre = raw"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?"
    # Regex for a variable spec allowing names, quotes, spaces inside brackets
    specre = raw"\w+\[[^\]]+\]"

    for (lineno, raw) in enumerate(eachline(filename))
        line = strip(split(raw, '#')[1])
        isempty(line) && continue

        if (m = match(r"^\s*method\s*=\s*(\w+)", line)) !== nothing
            method = Symbol(lowercase(m[1]))
            method in (:johansen, :euler, :gragg) ||
                error("Line $lineno: unknown method '$(m[1])' — valid options are: johansen, euler, gragg")

        elseif (m = match(r"^\s*steps\s*=\s*(\d+)", line)) !== nothing
            steps = parse(Int, m[1])

        elseif (m = match(r"^\s*set\s+(\w+)\s*=\s*\[([^\]]+)\]", line)) !== nothing
            setname = lowercase(m[1])
            elems   = [lowercase(strip(e, ['\'', '"', ' ']))
                       for e in split(m[2], ',')]
            user_sets[setname] = filter(!isempty, elems)

        elseif (m = match(Regex("^\\s*shock\\s+($specre)\\s*=\\s*($numre)"), line)) !== nothing
            push!(raw_shocks, m[1] => parse(Float64, m[2]))

        elseif (m = match(Regex("^\\s*swap\\s+($specre)\\s*<->\\s*($specre)(?:\\s+fix_at\\s*=\\s*($numre))?"), line)) !== nothing
            fix_at = m[3] !== nothing ? parse(Float64, m[3]) : 0.0
            push!(raw_swaps, (m[1], m[2], fix_at))

        else
            @warn "Line $lineno: unrecognised directive — $line"
        end
    end

    return method, steps, raw_shocks, raw_swaps, user_sets
end

# ── Interactive runner ────────────────────────────────────────────────────────

"""
    run_gtap_interactive() → GTAPSolution

Prompt the user for an experiment name and a `.cfg` file, run the simulation,
display results, and optionally export to CSV.
"""
function run_gtap_interactive()
    println("\n" * "═"^60)
    println("  JRunGTAP — interactive experiment runner")
    println("═"^60)

    # ── Data: ask for zip path once; reuse cache on subsequent calls ────────
    if _DATA_CACHE[] === nothing
        local zippath
        while true
            print("\n  GTAPAgg3 zip file : ")
            zippath = String(strip(readline()))
            isfile(zippath) && break
            println("  ✗  File not found: \"$zippath\"  — please try again.")
        end
        load_data(zippath)
    else
        println("\n  (data already loaded — $(basename(_ZIP_CACHE[])))")
    end
    s, d, C = _DATA_CACHE[]

    # ── Experiment name → parse config → solve (loop on cfg errors) ─────────
    local sol
    while true
        local name
        while true
            print("\n  Experiment name : ")
            name = String(strip(readline()))
            isempty(name) && (name = "experiment")
            cfgfile_check = name * ".cfg"
            isfile(cfgfile_check) && break
            println("  ✗  Config file not found: \"$cfgfile_check\"  — please try again.")
        end
        cfgfile = name * ".cfg"
        csvfile = name * ".csv"

        try
            method, steps, raw_shocks, raw_swaps, user_sets = parse_config(cfgfile)
            shocks, swaps = resolve_config(raw_shocks, raw_swaps, user_sets, s)

            println("\n  ── Resolved configuration " * "─"^34)
            println("  config  = $cfgfile")
            println("  method  = $method")
            method in (:euler, :gragg) && println("  steps   = $steps")
            for (sn, elems) in sort(collect(user_sets))
                println("  set     $sn = [$(join(elems, ", "))]")
            end
            for (k, v) in sort(collect(shocks))
                println("  shock   $k = $v")
            end
            for sw in swaps
                fix_str = sw.fix_at != 0.0 ? "  fix_at=$(sw.fix_at)" : ""
                println("  swap    $(sw.endo_out) <-> $(sw.exog_in)$fix_str")
            end

            exp = gtap_experiment(name; shocks = shocks, swaps = swaps,
                                  method = method, steps = steps)
            sol = run_gtap(exp)
            break   # success — exit the retry loop
        catch err
            println("\n  ✗  Configuration error in \"$cfgfile\":")
            println("     ", replace(sprint(showerror, err), "\n" => "\n     "))
            println("     Fix the file and try again, or enter a different experiment name.")
        end
    end

    show_results(sol)

    # ── Export: default filename = experiment_name.csv ──────────────────────
    csvfile = sol.experiment.name * ".csv"
    print("\n  Export results to $csvfile? [Y/n] : ")
    ans = lowercase(String(strip(readline())))
    (ans == "" || ans == "y" || ans == "yes") && export_results(sol, csvfile)

    return sol
end

# Suppress REPL auto-display of the solution struct.
Base.show(io::IO, ::MIME"text/plain", ::GTAPSolution) = nothing

# ── Entry point when run as a script ─────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    run_gtap_interactive()
end
