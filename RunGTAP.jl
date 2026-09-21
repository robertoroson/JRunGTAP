# RunGTAP.jl
# ─────────────────────────────────────────────────────────────────────────────
cd(@__DIR__)   # ensure working directory = project folder regardless of call site
# Julia equivalent of the RunGTAP/GEMPACK workflow for the GTAP v6.2 CGE model.
#
# Quick start — interactive (reads zip, then loops over experiments):
#   include("RunGTAP.jl")
#   run_gtap_interactive()          # prompts for zip file, then experiment name
#                                   # config = name.cfg, output = name.csv
#
# Programmatic use:
#   include("RunGTAP.jl")
#   s, d, C = load_data("mydata.zip")          # or load_data(set_file=..., dat_file=..., par_file=...)
#   shocks, swaps = resolve_config(["tm[pdr,Austria]" => 10.0], [], Dict(), s)
#   sol = run_gtap(gtap_experiment("tariff"; shocks=shocks))
#   show_results(sol);  export_results(sol, "tariff.csv")
#
# With closure swaps (e.g. long-run: fix rental, let qo_endw adjust):
#   sol = run_gtap(gtap_experiment("tariff_longrun";
#                     shocks = [Shock("tm[1,1]", 10.0)],
#                     swaps  = [Swap("rental[1]", "qo_endw[1,1]")]))
# ─────────────────────────────────────────────────────────────────────────────

# ── (a) Load model infrastructure ────────────────────────────────────────────

include("gtap_pack.jl")         # F_core, make_exog_zero, n_endog, CORE_EQ_NAMES
include("gtap_analytical.jl")   # build_A_analytical
include("gtap_solver.jl")       # solve_gtap_analytical, get_variable
include("load_gtapAgg3.jl")     # load_gsdf
include("gtap_names.jl")        # var_domains, resolve_config, expand_varspec
include("gtap_euler.jl")        # update_data_euler (multi-step Euler)

# ── GTAPv7 infrastructure ─────────────────────────────────────────────────────
include("gtap_v7.jl")           # GTAPSetsV7, GTAPDataV7, gtap_v7_residuals
include("gtap_names_v7.jl")     # var_domains / _model_sets / resolve_config for GTAPSetsV7
include("gtap_pack_v7.jl")      # F_core_v7, make_exog_zero_v7, _endo_specs_v7
include("gtap_analytical_v7.jl")          # build_A_v7 (colored FD, pattern cache)
include("gtap_jacobian_analytical_v7.jl") # build_A_v7_analytical (direct, <1s)
include("gtap_euler_v7.jl")     # update_data_euler_v7
include("load_gtapAgg3_v7.jl")  # load_from_zip_v7

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

# ── (b2) GTAPv7 data loading ──────────────────────────────────────────────────

const _DATA_CACHE_V7     = Ref{Union{Nothing,Tuple}}(nothing)
const _JACOBIAN_CACHE_V7 = Ref{Any}(nothing)

# In the v7 model, technology aggregators (ao, ava, af, afe, aint) are endogenous
# variables defined by decomposition equations. The exogenous components that users
# typically shock are the all-dimension variants (aoall, avaall, afall, afeall, aintall).
# This dict lets users write the familiar GEMPACK names in .cfg files.
const _V7_TECH_ALIASES = Dict{Symbol,Symbol}(
    :ao   => :aoall,
    :ava  => :avaall,
    :af   => :afall,
    :afe  => :afeall,
    :aint => :aintall,
)
const _ZIP_CACHE_V7      = Ref{Union{Nothing,String}}(nothing)

"""
    load_data_v7(zippath) → (s, d, C)

Load GTAPv7 data from a zip archive (must contain v7 HAR files).
Results are cached; call reload_data_v7!() to force a fresh load.
"""
function load_data_v7(zippath::String)
    if _DATA_CACHE_V7[] === nothing || _ZIP_CACHE_V7[] != zippath
        println("Loading GTAPv7 data from $(basename(zippath)) …")
        _DATA_CACHE_V7[]    = load_from_zip_v7(zippath)
        _ZIP_CACHE_V7[]     = zippath
        _JACOBIAN_CACHE_V7[] = nothing
        set_jac_cache_path_v7(zippath)
        s, d, C = _DATA_CACHE_V7[]
        println("  nC=$(length(s.COMM))  nA=$(length(s.ACTS))  nR=$(length(s.REG))  " *
                "n_endo=$(n_endog_v7(s))")
        b0 = F_core_v7(zeros(n_endog_v7(s)), make_exog_zero_v7(s), d, s, C)
        println("  Benchmark ‖F(0)‖∞ = $(maximum(abs, b0))")
    end
    _DATA_CACHE_V7[]
end

reload_data_v7!() = (_DATA_CACHE_V7[] = nothing; _JACOBIAN_CACHE_V7[] = nothing;
                     _ZIP_CACHE_V7[] = nothing;
                     _JAC_PATTERN_V7[] = nothing; _JAC_GROUPS_V7[] = nothing; nothing)

function _get_jacobian_v7(d, s, C; rebuild = false)
    if _JACOBIAN_CACHE_V7[] === nothing || rebuild
        println("Building GTAPv7 Jacobian…")
        t0 = time()
        _JACOBIAN_CACHE_V7[] = build_A_v7_analytical(d, s, C)
        println("  done in $(round(time()-t0, digits=1))s   nnz=$(nnz(_JACOBIAN_CACHE_V7[]))")
    else
        println("  (reusing cached v7 Jacobian)")
    end
    _JACOBIAN_CACHE_V7[]
end

"""
    GTAPSolutionV7

Returned by `run_gtap_v7`. Analogous to GTAPSolution but for the v7 model.
"""
struct GTAPSolutionV7
    experiment :: GTAPExperiment
    x_endo     :: Vector{Float64}
    swap_in    :: Dict{Int,String}
    swap_out   :: Dict{Int,Tuple{String,Float64}}
    A          :: Any
    s          :: GTAPSetsV7
    d          :: GTAPDataV7
    C          :: Any
end

Base.show(io::IO, ::MIME"text/plain", ::GTAPSolutionV7) = nothing

# Parse v7 variable spec: "varname[i,j,...]" → (:varname, [i,j,...])
_parse_varspec_v7(spec::String) = _parse_varspec(spec)

function _endo_col_v7(name::Symbol, idxs::Vector{Int}, s::GTAPSetsV7)
    off   = endo_offsets_v7(s)
    specs = Dict(_endo_specs_v7(s))
    haskey(off, name) || error("'$name' is not endogenous in the v7 closure")
    shape = specs[name]
    lin   = LinearIndices(shape)[idxs...]
    return off[name] + lin - 1
end

function _exog_col_v7(name::Symbol, idxs::Vector{Int}, d, s::GTAPSetsV7, C)
    exog_base = make_exog_zero_v7(s)
    haskey(pairs(exog_base), name) ||
        error("'$name' is not exogenous in the v7 standard closure")
    exog_dict = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                     for (k, v) in pairs(exog_base))
    exog_dict[name][idxs...] = 1.0
    scalars = (:pfactwld, :walraslack)
    exog = NamedTuple(k => (k in scalars ? exog_dict[k][1] : exog_dict[k])
                      for k in keys(exog_base))
    F_core_v7(zeros(n_endog_v7(s)), exog, d, s, C)
end

function _replace_col_v7(A, j, col)
    _replace_col(A, j, col)   # same implementation as v6.2 version
end

function apply_swaps_v7(A, b, swaps::Vector{Swap}, d, s::GTAPSetsV7, C)
    isempty(swaps) && return A, b, Dict{Int,String}(), Dict{Int,Tuple{String,Float64}}()
    A_sw = A;  b_sw = copy(b)
    swap_in  = Dict{Int,String}()
    swap_out = Dict{Int,Tuple{String,Float64}}()
    for sw in swaps
        nm_out, idxs_out = _parse_varspec(sw.endo_out)
        nm_in,  idxs_in  = _parse_varspec(sw.exog_in)
        j   = _endo_col_v7(nm_out, idxs_out, s)
        B_k = _exog_col_v7(nm_in,  idxs_in,  d, s, C)
        println("  Swap v7: $(sw.endo_out) → exo (fix=$(sw.fix_at)) | $(sw.exog_in) → endo  [col $j]")
        sw.fix_at != 0.0 && (b_sw .-= A_sw[:, j] .* sw.fix_at)
        A_sw = _replace_col(A_sw, j, B_k)
        swap_in[j]  = sw.exog_in
        swap_out[j] = (sw.endo_out, sw.fix_at)
    end
    return A_sw, b_sw, swap_in, swap_out
end

"""
    run_gtap_v7(exp, zippath; rebuild_jacobian) → GTAPSolutionV7

Solve the linearised GTAPv7 model for the experiment defined in `exp`.
`zippath` must point to a v7-format GTAPAgg3 zip archive.
"""
function run_gtap_v7(exp::GTAPExperiment, zippath::String; rebuild_jacobian = false)
    s, d, C = load_data_v7(zippath)
    A       = _get_jacobian_v7(d, s, C; rebuild = rebuild_jacobian)

    println("\n── v7: $(exp.name) " * "─"^max(0, 57 - length(exp.name)))
    meth_str = exp.method === :johansen ? "Johansen" :
               exp.method === :euler   ? "Euler ($(exp.steps) steps)" :
               (exp.steps >= 6 && exp.steps % 3 == 0) ?
                   let n = exp.steps ÷ 3; "Gragg Richardson [$n,$(2n),$(3n)]" end :
                   "Gragg ($(exp.steps) steps)"
    println("  Method : $meth_str")
    isempty(exp.swaps)  || println("  Swaps  : $(length(exp.swaps))")
    isempty(exp.shocks) || for (k,v) in sort(collect(exp.shocks))
        println("  Shock  : $k = $v")
    end
    println()

    exog_dict = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                     for (k, v) in pairs(make_exog_zero_v7(s)))
    for (key, val) in exp.shocks
        m = match(r"^(\w+)\[([0-9,]+)\]$", key)
        m === nothing && (@warn "Cannot parse shock '$key'"; continue)
        fn = get(_V7_TECH_ALIASES, Symbol(m[1]), Symbol(m[1]))
        idxs = parse.(Int, split(m[2], ','))
        haskey(exog_dict, fn) ? exog_dict[fn][idxs...] = val :
            @warn "Shock field '$fn' not in v7 exogenous bundle"
    end
    scalars = (:pfactwld, :walraslack)
    exog = NamedTuple(k => (k in scalars ? exog_dict[k][1] : exog_dict[k])
                      for k in keys(make_exog_zero_v7(s)))

    n = n_endog_v7(s)
    b = -F_core_v7(zeros(n), exog, d, s, C)
    # Trim Walras redundancy: drop last row so system is square (n×n)
    A_sq = size(A,1) > n ? A[1:n,:] : A
    b_sq = length(b)  > n ? b[1:n]  : b

    A_eff, b_eff, swap_in, swap_out = apply_swaps_v7(A_sq, b_sq, exp.swaps, d, s, C)

    x_endo = if exp.method === :johansen
        _johansen_solve(A_eff, b_eff)
    elseif exp.method === :euler
        _euler_solve_v7(exp.shocks, exp.swaps, exp.steps, d, s, C)
    else
        _gragg_solve_v7(exp.shocks, exp.swaps, exp.steps, d, s, C)
    end

    GTAPSolutionV7(exp, x_endo, swap_in, swap_out, A, s, d, C)
end

function _euler_solve_v7(shocks, swaps, steps, d0::GTAPDataV7, s::GTAPSetsV7, C0)
    n       = n_endog_v7(s)
    x_level = ones(n)
    d_cur   = d0;  C_cur = C0
    scalars = (:pfactwld, :walraslack)

    for step in 1:steps
        print("\r  step $step/$steps…")
        ed = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                  for (k, v) in pairs(make_exog_zero_v7(s)))
        for (key, val) in shocks
            m = match(r"^(\w+)\[([0-9,]+)\]$", key); m === nothing && continue
            fn = get(_V7_TECH_ALIASES, Symbol(m[1]), Symbol(m[1]))
            idxs = parse.(Int, split(m[2], ','))
            haskey(ed, fn) && (ed[fn][idxs...] = val / steps)
        end
        exog = NamedTuple(k => (k in scalars ? ed[k][1] : ed[k]) for k in keys(make_exog_zero_v7(s)))

        A_cur = build_A_v7_analytical(d_cur, s, C_cur)
        b_cur = -F_core_v7(zeros(n), exog, d_cur, s, C_cur)
        # Drop the last row (E_walras) — it's Walras' law redundancy in the square system
        if size(A_cur, 1) > n; A_cur = A_cur[1:n, :]; b_cur = b_cur[1:n]; end
        A_eff, b_eff, _, _ = apply_swaps_v7(A_cur, b_cur, swaps, d_cur, s, C_cur)
        dx = lu(A_eff) \ b_eff

        x_level .*= (1 .+ dx ./ 100)
        d_cur = update_data_euler_v7(d_cur, unpack_endo_v7(dx, make_exog_zero_v7(s), s), s)
        C_cur = compute_derived_v7(d_cur, s)
    end
    println("\r  done ($steps steps).   ")
    100 .* (x_level .- 1)
end

# Single Gragg pass with `np` sub-steps. Returns the x_level vector (not % change).
function _gragg_pass_v7(shocks, swaps, np::Int, d0::GTAPDataV7, s::GTAPSetsV7, C0;
                        quiet::Bool=false)
    n       = n_endog_v7(s)
    x_level = ones(n)
    d_cur   = d0;  C_cur = C0
    scalars = (:pfactwld, :walraslack)

    _make_exog_sub() = begin
        ed = Dict(k => copy(float.(v isa Number ? [float(v)] : float.(v)))
                  for (k, v) in pairs(make_exog_zero_v7(s)))
        for (key, val) in shocks
            m = match(r"^(\w+)\[([0-9,]+)\]$", key); m === nothing && continue
            fn = get(_V7_TECH_ALIASES, Symbol(m[1]), Symbol(m[1]))
            idxs = parse.(Int, split(m[2], ','))
            haskey(ed, fn) && (ed[fn][idxs...] = val / np)
        end
        NamedTuple(k => (k in scalars ? ed[k][1] : ed[k]) for k in keys(make_exog_zero_v7(s)))
    end

    for step in 1:np
        quiet || print("\r  step $step/$np…")
        exog_sub = _make_exog_sub()

        A_cur = build_A_v7_analytical(d_cur, s, C_cur)
        b1    = -F_core_v7(zeros(n), exog_sub, d_cur, s, C_cur)
        if size(A_cur,1) > n; A_cur = A_cur[1:n,:]; b1 = b1[1:n]; end
        A1, b1e, _, _ = apply_swaps_v7(A_cur, b1, swaps, d_cur, s, C_cur)
        k1 = lu(A1) \ b1e

        dx_half = unpack_endo_v7(k1 ./ 2, make_exog_zero_v7(s), s)
        d_mid   = update_data_euler_v7(d_cur, dx_half, s)
        C_mid   = compute_derived_v7(d_mid, s)

        A_mid = build_A_v7_analytical(d_mid, s, C_mid)
        b2    = -F_core_v7(zeros(n), exog_sub, d_mid, s, C_mid)
        if size(A_mid,1) > n; A_mid = A_mid[1:n,:]; b2 = b2[1:n]; end
        A2, b2e, _, _ = apply_swaps_v7(A_mid, b2, swaps, d_mid, s, C_mid)
        k2 = lu(A2) \ b2e

        x_level .*= (1 .+ k2 ./ 100)
        dx_full  = unpack_endo_v7(k2, make_exog_zero_v7(s), s)
        d_cur    = update_data_euler_v7(d_cur, dx_full, s)
        C_cur    = compute_derived_v7(d_cur, s)
    end
    x_level
end

# Gragg solver with Richardson extrapolation over three passes [n1, n2, n3].
# When steps is divisible by 3 and ≥ 6, uses n1=steps/3, n2=2*steps/3, n3=steps
# (matching RunGTAP's "Steps = 2 4 6" convention).  Falls back to a single pass otherwise.
function _gragg_solve_v7(shocks, swaps, steps::Int, d0::GTAPDataV7, s::GTAPSetsV7, C0)
    if steps >= 6 && steps % 3 == 0
        n1, n2, n3 = steps ÷ 3, (steps ÷ 3) * 2, steps   # e.g. 2, 4, 6

        print("  Gragg pass n=$n1 …"); flush(stdout)
        xl1 = _gragg_pass_v7(shocks, swaps, n1, d0, s, C0; quiet=true)
        print("\r  Gragg pass n=$n2 …"); flush(stdout)
        xl2 = _gragg_pass_v7(shocks, swaps, n2, d0, s, C0; quiet=true)
        print("\r  Gragg pass n=$n3 …"); flush(stdout)
        xl3 = _gragg_pass_v7(shocks, swaps, n3, d0, s, C0; quiet=true)

        # Polynomial extrapolation to h²→0 (Lagrange at u=h²=1/n²).
        # Error of modified midpoint is O(h²), so T(h²) = T* + a·h² + b·h⁴ + …
        # Extrapolate using Lagrange basis at u=0 in the variable u=1/n².
        u1, u2, u3 = 1.0/n1^2, 1.0/n2^2, 1.0/n3^2
        c1 = (u2 * u3) / ((u1 - u2) * (u1 - u3))   # ≈  1/24  for [2,4,6]
        c2 = (u1 * u3) / ((u2 - u1) * (u2 - u3))   # ≈ -16/15 for [2,4,6]
        c3 = (u1 * u2) / ((u3 - u1) * (u3 - u2))   # ≈  81/40 for [2,4,6]

        xl = c1 .* xl1 .+ c2 .* xl2 .+ c3 .* xl3
        println("\r  done (Richardson [$n1,$n2,$n3]).   ")
        100 .* (xl .- 1)
    else
        xl = _gragg_pass_v7(shocks, swaps, steps, d0, s, C0)
        println("\r  done ($steps steps, Gragg).   ")
        100 .* (xl .- 1)
    end
end

"""
    get_result_v7(sol, name) → Array

Extract a named variable from a GTAPSolutionV7.
"""
function get_result_v7(sol::GTAPSolutionV7, name::Symbol)
    off   = endo_offsets_v7(sol.s)
    specs = Dict(_endo_specs_v7(sol.s))
    if haskey(off, name)
        arr = copy(get_variable_v7(sol.x_endo, name, sol.s))
        for (j, (spec_out, fix_val)) in sol.swap_out
            nm_out, _ = _parse_varspec(spec_out)
            nm_out == name || continue
            arr[j - off[name] + 1] = fix_val
        end
        return arr
    end
    for (_, (spec_out, fix_val)) in sol.swap_out
        nm_out, _ = _parse_varspec(spec_out)
        nm_out == name && return fix_val
    end
    derived = gtap_derived_v7(sol)
    haskey(derived, name) && return derived[name]
    return 0.0
end

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
function _replace_col(A::SparseMatrixCSC{Float64,<:Integer}, j::Integer,
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
               (exp.steps >= 6 && exp.steps % 3 == 0) ?
                   let n = exp.steps ÷ 3; "Gragg Richardson [$n,$(2n),$(3n)]" end :
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

# Single Gragg pass with `np` sub-steps. Returns the x_level vector (not % change).
function _gragg_pass(shocks, swaps, np::Int, d0::GTAPData, s::GTAPSets, C0;
                     quiet::Bool = false)
    # Modified midpoint (Gragg) method.
    # k1 (Euler slope at d_cur) is used for the benchmark update to keep value
    # flows consistent with d_cur's price structure.
    # k2 (midpoint slope) is used for accumulation (2nd-order accuracy).
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
            haskey(ed, fn) && (ed[fn][idxs...] = val / np)
        end
        NamedTuple(k => (k in scalars ? ed[k][1] : ed[k]) for k in keys(make_exog_zero(s)))
    end

    for step in 1:np
        quiet || print("\r  step $step/$np…")
        exog_sub = _make_exog_sub()

        A_cur = build_A_analytical(d_cur, s, C_cur)
        b1    = -F_core(zeros(n), exog_sub, d_cur, s, C_cur)
        A1, b1e, _, _ = apply_swaps(A_cur, b1, swaps, d_cur, s, C_cur)
        k1 = lu(A1) \ b1e

        d_mid = update_data_euler(d_cur, k1 ./ 2, s)
        C_mid = compute_derived(d_mid, s)

        A_mid = build_A_analytical(d_mid, s, C_mid)
        b2    = -F_core(zeros(n), exog_sub, d_mid, s, C_mid)
        A2, b2e, _, _ = apply_swaps(A_mid, b2, swaps, d_mid, s, C_mid)
        k2 = lu(A2) \ b2e

        x_level .*= (1 .+ k2 ./ 100)
        d_cur = update_data_euler(d_cur, k1, s)
        C_cur = compute_derived(d_cur, s)
    end

    x_level
end

# Gragg solver with Richardson extrapolation when steps ≥ 6 and steps % 3 == 0.
function _gragg_solve(shocks, swaps, steps::Int, d0::GTAPData, s::GTAPSets, C0)
    if steps >= 6 && steps % 3 == 0
        n1, n2, n3 = steps ÷ 3, (steps ÷ 3) * 2, steps
        print("  Gragg pass n=$n1 …"); flush(stdout)
        xl1 = _gragg_pass(shocks, swaps, n1, d0, s, C0; quiet=true)
        print("\r  Gragg pass n=$n2 …"); flush(stdout)
        xl2 = _gragg_pass(shocks, swaps, n2, d0, s, C0; quiet=true)
        print("\r  Gragg pass n=$n3 …"); flush(stdout)
        xl3 = _gragg_pass(shocks, swaps, n3, d0, s, C0; quiet=true)
        u1, u2, u3 = 1.0/n1^2, 1.0/n2^2, 1.0/n3^2
        c1 = (u2 * u3) / ((u1 - u2) * (u1 - u3))
        c2 = (u1 * u3) / ((u2 - u1) * (u2 - u3))
        c3 = (u1 * u2) / ((u3 - u1) * (u3 - u2))
        xl = c1 .* xl1 .+ c2 .* xl2 .+ c3 .* xl3
        println("\r  done (Richardson [$n1,$n2,$n3]).   ")
        100 .* (xl .- 1)
    else
        xl = _gragg_pass(shocks, swaps, steps, d0, s, C0)
        println("\r  done ($steps steps, Gragg).   ")
        100 .* (xl .- 1)
    end
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
    derived = gtap_derived_v62(sol)
    haskey(derived, name) && return derived[name]
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

        # Derived / reporting aggregates
        derived = gtap_derived_v62(sol)
        for (nm, arr) in sort(collect(derived); by = x -> string(x[1]))
            haskey(dom_orig, nm) || continue
            for idx in CartesianIndices(arr)
                v = round(arr[idx], digits = digits)
                skip_zeros && iszero(v) && continue
                println(io, expname, ",", nm, ",",
                        join(pad(idx_names(nm, idx)), ","), ",", v)
                rows += 1
            end
        end
    end
    println("Exported $rows non-zero entries → $filename")
end

"""
    gtap_derived_v62(sol) → Dict{Symbol, Array}

Compute derived reporting variables from a GTAPSolution (v6.2) — trade indices,
GDP, terms of trade, factor price ratios, EV, etc.
Returns a Dict of Symbol → Array (all percentage-change form).
"""
function gtap_derived_v62(sol::GTAPSolution)
    s = sol.s;  d = sol.d;  C = sol.C
    nT = length(s.TRAD_COMM);  nP = length(s.PROD_COMM)
    nR = length(s.REG);        nE = length(s.ENDW_COMM)
    nM = length(s.MARG_COMM)

    _get(nm) = try get_result(sol, nm) catch; nothing end

    _wsumR(W, X) = sum(W .* X) / max(sum(W), 1e-30)
    _wsum1(W, X) = [_wsumR(W[:,r], X[:,r]) for r in 1:nR]

    out = Dict{Symbol, Array}()

    pfob  = _get(:pfob)   # (nT, nR, nR)
    pcif  = _get(:pcif)   # (nT, nR, nR)
    qxs   = _get(:qxs)    # (nT, nR, nR)
    qva_v = _get(:qva)    # (nP, nR)
    p     = _get(:p)      # (nR,)
    yp    = _get(:yp)     # (nR,)
    yg    = _get(:yg)     # (nR,)
    pfe_v = _get(:pfe)    # (nE, nP, nR)
    y     = _get(:y)      # (nR,)
    u_v   = _get(:u)      # (nR,)
    pcgds = _get(:pcgds)  # (nR,)
    qcgds = _get(:qcgds)  # (nR,)

    # VFOB_r[t,r] = total FOB exports of t from r (sum over destinations)
    VFOB_r = dropdims(sum(d.VXWD, dims=3), dims=3)  # (nT, nR)
    # VCIF_r[t,r] = total CIF imports of t into r (sum over sources)
    VCIF_r = dropdims(sum(d.VIWS, dims=2), dims=2)  # (nT, nR)

    # pxw/qxw are now endogenous; use solution values when available.
    pxw_sol = _get(:pxw)
    qxw_sol = _get(:qxw)
    if pfob !== nothing && qxs !== nothing
        # ── Export price/volume indices ────────────────────────────────────
        pxw = pxw_sol !== nothing ? pxw_sol : begin
            tmp = zeros(nT, nR)
            for t in 1:nT, r in 1:nR
                w  = d.VXWD[t,r,:]
                sw = max(sum(w), 1e-30)
                tmp[t,r] = sum(w .* pfob[t,r,:]) / sw
            end
            tmp
        end
        qxw = qxw_sol !== nothing ? qxw_sol : begin
            tmp = zeros(nT, nR)
            for t in 1:nT, r in 1:nR
                w  = d.VXWD[t,r,:]
                sw = max(sum(w), 1e-30)
                tmp[t,r] = sum(w .* qxs[t,r,:]) / sw
            end
            tmp
        end
        out[:pxw] = pxw;  out[:qxw] = qxw
        out[:vxwfob] = pxw .+ qxw

        pxwreg = [_wsumR(VFOB_r[:,r], pxw[:,r]) for r in 1:nR]
        qxwreg = [_wsumR(VFOB_r[:,r], qxw[:,r]) for r in 1:nR]
        out[:pxwreg] = pxwreg;  out[:qxwreg] = qxwreg
        out[:vxwreg] = pxwreg .+ qxwreg

        VFOB_t = vec(sum(VFOB_r, dims=2))
        out[:pxwcom] = [_wsumR(VFOB_r[t,:], pxw[t,:]) for t in 1:nT]
        out[:qxwcom] = [_wsumR(VFOB_r[t,:], qxw[t,:]) for t in 1:nT]
        out[:vxwcom] = out[:pxwcom] .+ out[:qxwcom]

        # ── World output / price index ─────────────────────────────────────
        VDSA = try
            dropdims(sum(d.VXMD, dims=3), dims=3)  # (nT, nR) dom sales
        catch
            nothing
        end
        qds_v = _get(:qds)
        pds_v = try _get(:pr) catch; nothing end   # pr ≈ domestic price in v6.2
        if qds_v !== nothing && VDSA !== nothing
            VSALES = VFOB_r .+ VDSA
            qow_t = zeros(nT)
            pw_t  = zeros(nT)
            for t in 1:nT
                w_exp = VFOB_r[t,:]
                w_dom = VDSA[t,:]
                W = w_exp .+ w_dom
                sw = max(sum(W), 1e-30)
                qow_t[t] = (sum(w_exp .* qxwreg) + sum(w_dom .* qds_v[t,:])) / sw
                if pds_v !== nothing
                    pw_t[t] = (sum(w_exp .* pxwreg) + sum(w_dom .* pds_v[t,:])) / sw
                end
            end
            out[:qow]    = qow_t
            out[:pw]     = pw_t
            out[:valuew] = pw_t .+ qow_t
        end
    end

    if pcif !== nothing && qxs !== nothing
        # ── Import price/volume indices ────────────────────────────────────
        pmw = zeros(nT, nR);  qmw = zeros(nT, nR)
        for t in 1:nT, r in 1:nR
            w = d.VIWS[t,:,r]
            sw = max(sum(w), 1e-30)
            pmw[t,r] = sum(w .* pcif[t,:,r]) / sw
            qmw[t,r] = sum(w .* qxs[t,:,r])  / sw
        end
        out[:pmw] = pmw;  out[:qmw] = qmw
        out[:vmwcif] = pmw .+ qmw

        pmwreg = [_wsumR(VCIF_r[:,r], pmw[:,r]) for r in 1:nR]
        qmwreg = [_wsumR(VCIF_r[:,r], qmw[:,r]) for r in 1:nR]
        out[:pmwreg] = pmwreg;  out[:qmwreg] = qmwreg
        out[:vmwreg] = pmwreg .+ qmwreg

        out[:pmwcom] = [_wsumR(VCIF_r[t,:], pmw[t,:]) for t in 1:nT]
        out[:qmwcom] = [_wsumR(VCIF_r[t,:], qmw[t,:]) for t in 1:nT]
        out[:vmwcom] = out[:pmwcom] .+ out[:qmwcom]
    end

    if haskey(out, :vxwfob) && haskey(out, :vmwcif)
        del_tbal  = zeros(nR);  del_tbalc = zeros(nT, nR)
        for r in 1:nR, t in 1:nT
            dX = VFOB_r[t,r] * out[:vxwfob][t,r] / 100.0
            dM = VCIF_r[t,r] * out[:vmwcif][t,r] / 100.0
            del_tbalc[t,r] = dX - dM
            del_tbal[r]   += del_tbalc[t,r]
        end
        out[:del_tbal]  = del_tbal
        out[:del_tbalc] = del_tbalc
        out[:del_tbalry] = del_tbal ./ max(sum(C.INCOME), 1e-10) .* 100.0
    end

    if haskey(out, :pxwreg) && haskey(out, :pmwreg)
        out[:psw] = copy(out[:pxwreg])
        out[:pdw] = copy(out[:pmwreg])
        out[:tot] = out[:pxwreg] .- out[:pmwreg]
    end

    if yp !== nothing && yg !== nothing && pcgds !== nothing && qcgds !== nothing &&
       haskey(out, :pxwreg) && haskey(out, :pmwreg)
        vgdp = zeros(nR)
        for r in 1:nR
            VFOB_tot = sum(VFOB_r[:,r])
            VCIF_tot = sum(VCIF_r[:,r])
            NX_base  = VFOB_tot - VCIF_tot
            dC  = C.PRIVEXP[r] * yp[r]    / 100.0
            dG  = C.GOVEXP[r]  * yg[r]    / 100.0
            dI  = C.REGINV[r]  * (pcgds[r] + qcgds[r]) / 100.0
            dX  = VFOB_tot * (out[:pxwreg][r] + out[:qxwreg][r]) / 100.0
            dM  = VCIF_tot * (out[:pmwreg][r] + out[:qmwreg][r]) / 100.0
            dGDP = dC + dG + dI + dX - dM
            gdp_base = C.PRIVEXP[r] + C.GOVEXP[r] + C.REGINV[r] + NX_base
            vgdp[r] = gdp_base > 1e-10 ? dGDP / gdp_base * 100.0 : 0.0
        end
        out[:vgdp] = vgdp
        p !== nothing && (out[:pgdp] = copy(p))
        haskey(out, :pgdp) && (out[:qgdp] = vgdp .- out[:pgdp])
    end

    if pfe_v !== nothing && p !== nothing
        pfactreal = zeros(nE, nP, nR)
        for e in 1:nE, pr in 1:nP, r in 1:nR
            pfactreal[e,pr,r] = pfe_v[e,pr,r] - p[r]
        end
        out[:pfactreal] = pfactreal
    end

    qva_v !== nothing && (out[:compvalad] = copy(qva_v))

    if u_v !== nothing
        out[:EV]       = C.INCOME .* u_v ./ 100.0
        out[:ueprivev] = copy(C.UELASPRIV)
        y !== nothing && (out[:yev] = copy(y))
    end

    out
end

"""
    gtap_derived_v7(sol) → Dict{Symbol, Array}

Compute derived reporting variables from a GTAPSolutionV7 — quantities that
GTAP computes post-solution (GDP, terms of trade, trade value indices, etc.).
Returns a Dict of Symbol → Array (all percentage-change form, same convention as
the rest of the solution).
"""
function gtap_derived_v7(sol::GTAPSolutionV7)
    s = sol.s;  d = sol.d;  C = sol.C
    nC = length(s.COMM);  nA = length(s.ACTS);  nR = length(s.REG)
    nE = length(s.ENDW);  nM = length(s.MARG)
    nEMS = length(s.ENDWMS)

    _get(nm) = try get_result_v7(sol, nm) catch; nothing end

    # Helper: value-share weighted sum of percentage changes → one percentage change
    # result = sum(w_i * x_i) / sum(w_i)   where w_i are benchmark values
    _wsumR(W, X) = sum(W .* X) / max(sum(W), 1e-30)   # scalar
    _wsum1(W, X) = [_wsumR(W[:,r], X[:,r]) for r in 1:nR]   # (nR,) from (nC,nR)

    out = Dict{Symbol, Array}()

    # ── Retrieve key endogenous results ──────────────────────────────────────
    qo    = _get(:qo)     # (nA, nR)  activity output
    pds   = _get(:pds)    # (nC, nR)  domestic commodity price
    pms   = _get(:pms)    # (nC, nR)  composite import price
    qms   = _get(:qms)    # (nC, nR)  aggregate imports
    pfob  = _get(:pfob)   # (nC, nR, nR)  FOB export price
    pcif  = _get(:pcif)   # (nC, nR, nR)  CIF import price
    qxs   = _get(:qxs)    # (nC, nR, nR)  bilateral export volume
    psave = _get(:psave)  # (nR,)  savings price
    qsave = _get(:qsave)  # (nR,)  savings demand
    p     = _get(:p)      # (nR,)  CPI
    pe    = _get(:pe)     # (nEMS, nR)  endowment price
    pfe   = _get(:pfe)    # (nE, nA, nR)  factor price paid by firms
    peb   = _get(:peb)    # (nE, nA, nR)  basic factor price
    y     = _get(:y)      # (nR,)  regional income

    # ── Export value and volume aggregates ────────────────────────────────────
    # FOB export values by (c,r): VFOB_r(c,r) = sum_d VFOB(c,r,d)
    VFOB_r = dropdims(sum(d.VFOB, dims=3), dims=3)   # (nC, nR)
    # CIF import values by (c,r): VCIF_r(c,r) = sum_s VCIF(c,s,r)
    VCIF_r = dropdims(sum(d.VCIF, dims=2), dims=2)   # (nC, nR)

    # pxw(c,r) and qxw(c,r): now endogenous variables; use solution values directly.
    # Fall back to computing from pfob/qxs if solution doesn't carry them (old runs).
    pxw_sol = _get(:pxw)   # (nC,nR) or nothing
    qxw_sol = _get(:qxw)   # (nC,nR) or nothing
    if pfob !== nothing
        pxw = pxw_sol !== nothing ? pxw_sol : begin
            tmp = zeros(nC, nR)
            for c in 1:nC, r in 1:nR
                tmp[c,r] = _wsumR(d.VFOB[c,r,:], pfob[c,r,:])
            end
            tmp
        end
        qxw = qxw_sol !== nothing ? qxw_sol : begin
            tmp = zeros(nC, nR)
            for c in 1:nC, r in 1:nR
                VFOB_r[c,r] < 1e-10 && continue
                val_pct = sum(d.VFOB[c,r,dd]*(pfob[c,r,dd]+qxs[c,r,dd]) for dd in 1:nR)
                tmp[c,r] = val_pct/VFOB_r[c,r] - pxw[c,r]
            end
            tmp
        end
        out[:pxw] = pxw
        out[:qxw] = qxw

        # vxwfob(c,r): % change in FOB export value = pxw + qxw
        out[:vxwfob] = pxw .+ qxw

        # pxwreg(r): regional export price index
        pxwreg = zeros(nR)
        for r in 1:nR
            pxwreg[r] = _wsumR(VFOB_r[:,r], pxw[:,r])
        end
        out[:pxwreg] = pxwreg

        # qxwreg(r): regional export volume
        qxwreg = zeros(nR)
        for r in 1:nR
            VFOB_tot = sum(VFOB_r[:,r])
            VFOB_tot < 1e-10 && continue
            val_pct = sum(VFOB_r[c,r]*(pxw[c,r]+qxw[c,r]) for c in 1:nC)
            qxwreg[r] = val_pct/VFOB_tot - pxwreg[r]
        end
        out[:qxwreg] = qxwreg

        # vxwreg(r): % change in regional FOB export value
        out[:vxwreg] = pxwreg .+ qxwreg

        # pxwcom(c): world export price index by commodity
        pxwcom = zeros(nC)
        for c in 1:nC
            pxwcom[c] = _wsumR(VFOB_r[c,:], pxw[c,:])
        end
        out[:pxwcom] = pxwcom

        # qxwcom(c): world export volume by commodity
        qxwcom = zeros(nC)
        for c in 1:nC
            VFOB_c = sum(VFOB_r[c,:])
            VFOB_c < 1e-10 && continue
            val_pct = sum(VFOB_r[c,r]*(pxw[c,r]+qxw[c,r]) for r in 1:nR)
            qxwcom[c] = val_pct/VFOB_c - pxwcom[c]
        end
        out[:qxwcom] = qxwcom

        # vxwcom(c): % change in world FOB export value by commodity
        out[:vxwcom] = pxwcom .+ qxwcom
    end

    # pmw(c,r): CIF import price index — value-share weighted avg over sources
    if pcif !== nothing
        pmw = zeros(nC, nR)
        for c in 1:nC, r in 1:nR
            pmw[c,r] = _wsumR(d.VCIF[c,:,r], pcif[c,:,r])
        end
        out[:pmw] = pmw

        # qmw(c,r): CIF import volume = sum_s VCIF*(pcif+qxs) / VCIF_r - pmw
        qmw = zeros(nC, nR)
        for c in 1:nC, r in 1:nR
            VCIF_r[c,r] < 1e-10 && continue
            val_pct = sum(d.VCIF[c,ss,r]*(pcif[c,ss,r]+qxs[c,ss,r]) for ss in 1:nR)
            qmw[c,r] = val_pct/VCIF_r[c,r] - pmw[c,r]
        end
        out[:qmw] = qmw

        # vmwcif(c,r): % change in CIF import value
        out[:vmwcif] = pmw .+ qmw

        # pmwreg(r): regional import price index (CIF weights)
        pmwreg = zeros(nR)
        for r in 1:nR
            pmwreg[r] = _wsumR(VCIF_r[:,r], pmw[:,r])
        end
        out[:pmwreg] = pmwreg

        # qmwreg(r): regional import volume
        qmwreg = zeros(nR)
        for r in 1:nR
            VCIF_tot = sum(VCIF_r[:,r])
            VCIF_tot < 1e-10 && continue
            val_pct = sum(VCIF_r[c,r]*(pmw[c,r]+qmw[c,r]) for c in 1:nC)
            qmwreg[r] = val_pct/VCIF_tot - pmwreg[r]
        end
        out[:qmwreg] = qmwreg

        # vmwreg(r): % change in regional CIF import value
        out[:vmwreg] = pmwreg .+ qmwreg

        # pmwcom(c): world import price index by commodity
        pmwcom = zeros(nC)
        for c in 1:nC
            pmwcom[c] = _wsumR(VCIF_r[c,:], pmw[c,:])
        end
        out[:pmwcom] = pmwcom

        # qmwcom(c): world import volume by commodity
        qmwcom = zeros(nC)
        for c in 1:nC
            VCIF_c = sum(VCIF_r[c,:])
            VCIF_c < 1e-10 && continue
            val_pct = sum(VCIF_r[c,r]*(pmw[c,r]+qmw[c,r]) for r in 1:nR)
            qmwcom[c] = val_pct/VCIF_c - pmwcom[c]
        end
        out[:qmwcom] = qmwcom

        # vmwcom(c): % change in world CIF import value by commodity
        out[:vmwcom] = pmwcom .+ qmwcom
    end

    # ── World commodity supply price and value ────────────────────────────────
    # VCB(c,r) = MAKEBCOM: total commodity supply at basic prices
    VCB = C.MAKEBCOM   # (nC, nR)  sum_a MAKEB(c,a,r)
    VCB_w = vec(sum(VCB, dims=2))   # (nC,)

    if pds !== nothing && qo !== nothing
        # pw(c): world commodity price — supply-value-weighted average of pds
        pw = zeros(nC)
        for c in 1:nC
            pw[c] = _wsumR(VCB[c,:], pds[c,:])
        end
        out[:pw] = pw

        # qow(c): world supply quantity — value-weighted average of qca change
        # approximated from qo using make shares
        qca = _get(:qca)
        if qca !== nothing
            qow = zeros(nC)
            for c in 1:nC
                VCB_w[c] < 1e-10 && continue
                val_pct = sum(VCB[c,r]*(pds[c,r]+sum(C.MAKEBCOMSHR[c,a,r]*qca[c,a,r] for a in 1:nA)) for r in 1:nR)
                qow[c] = val_pct/VCB_w[c] - pw[c]
            end
            out[:qow] = qow

            # valuew(c): % change in world value of good c = pw + qow
            out[:valuew] = pw .+ qow
        end
    end

    # ── Trade balance ─────────────────────────────────────────────────────────
    # del_tbal(r): dollar change in trade balance (exports FOB - imports CIF)
    # In the linearised model: Δ(X-M) = sum_c VFOB_r(c,r)*(%X_c,r)/100 - sum_c VCIF_r(c,r)*(%M_c,r)/100
    if haskey(out, :vxwfob) && haskey(out, :vmwcif)
        vxwfob = out[:vxwfob]
        vmwcif = out[:vmwcif]
        del_tbal = zeros(nR)
        del_tbalc = zeros(nC, nR)
        for r in 1:nR, c in 1:nC
            dX = VFOB_r[c,r] * vxwfob[c,r] / 100.0
            dM = VCIF_r[c,r] * vmwcif[c,r] / 100.0
            del_tbalc[c,r] = dX - dM
            del_tbal[r]   += del_tbalc[c,r]
        end
        out[:del_tbal]  = del_tbal
        out[:del_tbalc] = del_tbalc

        # del_tbalry(r): change in trade balance as % of world income
        world_income = sum(C.INCOME)
        out[:del_tbalry] = del_tbal ./ max(world_income, 1e-10) .* 100.0
    end

    # ── Terms of trade ────────────────────────────────────────────────────────
    # psw(r): export price index = pxwreg(r)
    # pdw(r): import price index = pmwreg(r)
    # tot(r) = psw(r) - pdw(r)
    if haskey(out, :pxwreg) && haskey(out, :pmwreg)
        out[:psw] = copy(out[:pxwreg])
        out[:pdw] = copy(out[:pmwreg])
        out[:tot] = out[:pxwreg] .- out[:pmwreg]
    end

    # ── GDP (expenditure approach) ────────────────────────────────────────────
    # vgdp(r): % change in nominal GDP = (C + G + I + X - M) weighted
    # C = private expenditure (PRIVEXP), G = gov (GOVEXP), I = investment (REGINV),
    # NX = VFOB_tot - VCIF_tot
    yp_v  = _get(:yp)
    yg_v  = _get(:yg)
    pinv_v = _get(:pinv)
    qinv_v = _get(:qinv)
    if yp_v !== nothing && yg_v !== nothing && pinv_v !== nothing && qinv_v !== nothing &&
       haskey(out, :vxwreg) && haskey(out, :vmwreg)
        vgdp = zeros(nR)
        for r in 1:nR
            VFOB_tot = sum(VFOB_r[:,r])
            VCIF_tot = sum(VCIF_r[:,r])
            NX_base  = VFOB_tot - VCIF_tot
            dC  = C.PRIVEXP[r] * yp_v[r]  / 100.0
            dG  = C.GOVEXP[r]  * yg_v[r]  / 100.0
            dI  = C.REGINV[r]  * (pinv_v[r] + qinv_v[r]) / 100.0
            dX  = VFOB_tot     * (out[:pxwreg][r] + out[:qxwreg][r]) / 100.0
            dM  = VCIF_tot     * (out[:pmwreg][r] + out[:qmwreg][r]) / 100.0
            dGDP = dC + dG + dI + dX - dM
            gdp_base = C.PRIVEXP[r] + C.GOVEXP[r] + C.REGINV[r] + NX_base
            vgdp[r] = gdp_base > 1e-10 ? dGDP / gdp_base * 100.0 : 0.0
        end
        out[:vgdp] = vgdp

        # pgdp(r): GDP deflator — approximated as CPI (p)
        if p !== nothing
            out[:pgdp] = copy(p)
        end

        # qgdp(r): real GDP = vgdp - pgdp
        if haskey(out, :pgdp)
            out[:qgdp] = vgdp .- out[:pgdp]
        end
    end

    # ── Factor price ratios ───────────────────────────────────────────────────
    # pfactreal(e,a,r): return to factor e in act. a relative to CPI
    if peb !== nothing && p !== nothing
        pfactreal = zeros(nE, nA, nR)
        for e in 1:nE, a in 1:nA, r in 1:nR
            pfactreal[e,a,r] = peb[e,a,r] - p[r]
        end
        out[:pfactreal] = pfactreal
    end

    # pebfactreal(e,r): mobile/sluggish endowment return relative to CPI
    if pe !== nothing && p !== nothing
        pebfactreal = zeros(nEMS, nR)
        for e in 1:nEMS, r in 1:nR
            pebfactreal[e,r] = pe[e,r] - p[r]
        end
        out[:pebfactreal] = pebfactreal
    end

    # ── Value added composition ───────────────────────────────────────────────
    # compvalad(a,r): composition of value added = qva(a,r) (% ch. in real VA)
    qva_v = _get(:qva)
    if qva_v !== nothing
        out[:compvalad] = copy(qva_v)
    end

    # ── EV variants ──────────────────────────────────────────────────────────
    # EV(r): equivalent variation in mn USD (already in export_results, included here too)
    u_v = _get(:u)
    if u_v !== nothing
        out[:EV] = C.INCOME .* u_v ./ 100.0

        # ueprivev(r): utility elasticity of priv. cons. expenditure for EV
        out[:ueprivev] = copy(C.UELASPRIV)

        # yev(r) = y (income at base prices used for EV)
        if y !== nothing
            out[:yev] = copy(y)
        end
    end

    out
end

"""
    export_results(sol::GTAPSolutionV7, filename; variables, skip_zeros, digits)

Write v7 simulation results to CSV with columns: experiment, variable, indices, pct_change.
"""
function export_results(sol::GTAPSolutionV7, filename::String;
                        variables  :: Union{Nothing,Vector{Symbol}} = nothing,
                        skip_zeros :: Bool = true,
                        digits     :: Int  = 6)
    specs    = _endo_specs_v7(sol.s)
    vars     = variables === nothing ? [nm for (nm, _) in specs] : variables
    dom_orig = var_domains_orig(sol.s)

    max_dims = maximum(length(v) for v in values(dom_orig); init = 4)

    function idx_names(nm::Symbol, idx::CartesianIndex)
        t = Tuple(idx)
        if haskey(dom_orig, nm)
            sets = dom_orig[nm]
            length(sets) == length(t) && return [sets[d][t[d]] for d in eachindex(sets)]
        end
        return [string(i) for i in t]
    end

    pad(v) = vcat(v, fill("", max_dims - length(v)))

    rows = 0
    open(filename, "w") do io
        dim_headers = join(["dim$i" for i in 1:max_dims], ",")
        println(io, "experiment,variable,", dim_headers, ",pct_change")
        expname = sol.experiment.name

        for nm in vars
            haskey(dom_orig, nm) || continue
            try
                arr = get_result_v7(sol, nm)
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
            nm_sym, idxs = _parse_varspec_v7(spec)
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

        # EV (mn USD) — kept as legacy column for backward compatibility
        u_v = try get_result_v7(sol, :u) catch; nothing end
        if u_v !== nothing
            ev = sol.C.INCOME .* u_v ./ 100
            for r in 1:length(sol.s.REG)
                v = round(ev[r], digits = digits)
                skip_zeros && iszero(v) && continue
                println(io, expname, ",EV_mn_USD,", join(pad([sol.s.REG[r]]), ","), ",", v)
                rows += 1
            end
        end

        # Derived / reporting aggregates (GDP, trade indices, ToT, etc.)
        derived = gtap_derived_v7(sol)
        for (nm, arr) in sort(collect(derived); by = x -> string(x[1]))
            haskey(dom_orig, nm) || continue
            for idx in CartesianIndices(arr)
                v = round(arr[idx], digits = digits)
                skip_zeros && iszero(v) && continue
                println(io, expname, ",", nm, ",",
                        join(pad(idx_names(nm, idx)), ","), ",", v)
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
    # GEMPACK .cmf/.exp files are always v7; .cfg files default to v62 for backwards compat
    model      = endswith(lowercase(filename), ".cmf") || endswith(lowercase(filename), ".exp") ? :v7 : :v62

    # Regex for a numeric value (int or float, optional sign/exponent)
    numre = raw"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?"
    # Regex for a variable spec: either bracket form  foo[a,b]  or GEMPACK paren form  foo("a","b")
    specre_sq = raw"\w+\[[^\]]+\]"
    specre_gp = raw"\w+\([^)]+\)"
    specre    = "(?:$specre_sq|$specre_gp)"

    # Convert GEMPACK paren spec  foo("a","b","c")  →  foo[a,b,c]  for uniform downstream handling
    function _norm_spec(s::AbstractString)
        m2 = match(r"^(\w+)\(([^)]+)\)$", strip(s))
        m2 === nothing && return s   # already bracket form
        args = join([strip(t, [' ', '"', '\'']) for t in split(m2[2], ',')], ',')
        "$(m2[1])[$args]"
    end

    for (lineno, raw) in enumerate(eachline(filename))
        # Strip GEMPACK-style ! comments AND ; terminators
        line = strip(split(split(raw, '!')[1], ';')[1])
        # Also strip # comments (Julia/cfg style)
        line = strip(split(line, '#')[1])
        isempty(line) && continue

        if (m = match(r"^\s*model\s*=\s*(\w+)"i, line)) !== nothing
            mv = lowercase(m[1])
            mv in ("v62", "v7") ||
                error("Line $lineno: unknown model '$(m[1])' — valid options are: v62, v7")
            model = Symbol(mv)

        elseif (m = match(r"^\s*method\s*=\s*(\w+)"i, line)) !== nothing
            method = Symbol(lowercase(m[1]))
            method in (:johansen, :euler, :gragg) ||
                error("Line $lineno: unknown method '$(m[1])' — valid options are: johansen, euler, gragg")

        elseif (m = match(r"^\s*steps\s*=\s*([\d\s,]+)"i, line)) !== nothing
            # Accept "2", "2 4 6", or "2,4,6" — use the LAST value (highest accuracy, like GEMPACK)
            nums = [parse(Int, s) for s in split(m[1], r"[\s,]+") if !isempty(s)]
            steps = isempty(nums) ? 10 : last(nums)

        elseif (m = match(r"^\s*set\s+(\w+)\s*=\s*\[([^\]]+)\]", line)) !== nothing
            setname = lowercase(m[1])
            elems   = [lowercase(strip(e, ['\'', '"', ' ']))
                       for e in split(m[2], ',')]
            user_sets[setname] = filter(!isempty, elems)

        elseif (m = match(Regex("^\\s*shock\\s+($specre)\\s*=\\s*($numre)", "i"), line)) !== nothing
            push!(raw_shocks, _norm_spec(m[1]) => parse(Float64, m[2]))

        elseif (m = match(Regex("^\\s*swap\\s+($specre)\\s*<->\\s*($specre)(?:\\s+fix_at\\s*=\\s*($numre))?", "i"), line)) !== nothing
            fix_at = m[3] !== nothing ? parse(Float64, m[3]) : 0.0
            push!(raw_swaps, (_norm_spec(m[1]), _norm_spec(m[2]), fix_at))

        elseif match(r"^\s*(?:cpu|nds|extrapolation|aux\s+files?|file\s|updated\s+file|solution\s+file|verbal\s+description|subintervals?|exogenous|rest\s+endogenous|!|@)"i, line) !== nothing ||
               !occursin('=', line)
            # Silently ignore: GEMPACK-specific directives, and lines without '='
            # (the latter covers exogenous variable list continuation lines)
            nothing

        else
            @warn "Line $lineno: unrecognised directive — $line"
        end
    end

    return method, steps, raw_shocks, raw_swaps, user_sets, model
end

function show_results(sol::GTAPSolutionV7)
    s = sol.s
    println("\n── Results: $(sol.experiment.name) (v7) " * "─"^36)

    _get(nm) = try get_result_v7(sol, nm) catch; nothing end

    u_arr = _get(:u)
    y_arr = _get(:y)
    p_arr = _get(:p)

    if u_arr !== nothing
        ev_arr = sol.C.INCOME .* u_arr ./ 100
        println("  Region              welfare(u)    income(y)    CPI(p)    EV (mn USD)")
        println("  " * "─"^72)
        for r in 1:length(s.REG)
            @printf("  %-20s %10.4f   %10.4f   %10.4f   %12.2f\n",
                    s.REG[r],
                    u_arr[r],
                    y_arr !== nothing ? y_arr[r] : NaN,
                    p_arr  !== nothing ? p_arr[r]  : NaN,
                    ev_arr[r])
        end
        @printf("  %-20s %10s   %10s   %10s   %12.2f\n",
                "WORLD", "", "", "", sum(ev_arr))
    end

    println()
    println("  variable   description                  min          max          ‖·‖∞")
    println("  " * "─"^72)
    for (nm, desc) in [(:qo,  "output (%)"),
                        (:qxs, "bilateral trade volumes (%)"),
                        (:pms, "import prices (%)"),
                        (:psave, "savings price (%)")]
        v = _get(nm)
        v === nothing && continue
        flat = vec(v)
        @printf("  %-10s %-28s %12.4f  %12.4f  %12.4f\n",
                nm, desc, minimum(flat), maximum(flat), maximum(abs, flat))
    end

    if !isempty(sol.swap_in)
        println("  " * "─"^72)
        println("  Swapped-in variables (now endogenous):")
        for (j, spec) in sort(collect(sol.swap_in))
            @printf("  %-36s = %12.6f\n", spec, sol.x_endo[j])
        end
    end
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

    # ── Zip path: ask once; cache for subsequent calls ───────────────────────
    local zippath
    if _ZIP_CACHE[] !== nothing
        zippath = _ZIP_CACHE[]
        println("\n  (zip cached — $(basename(zippath)))")
    elseif _ZIP_CACHE_V7[] !== nothing
        zippath = _ZIP_CACHE_V7[]
        println("\n  (zip cached — $(basename(zippath)))")
    else
        while true
            print("\n  GTAPAgg3 zip file : ")
            zippath = String(strip(readline()))
            isfile(zippath) && break
            println("  ✗  File not found: \"$zippath\"  — please try again.")
        end
    end

    # ── Experiment name → parse config → load data → solve ──────────────────
    local sol
    while true
        local name
        local cfgfile
        while true
            print("\n  Experiment name : ")
            name = String(strip(readline()))
            isempty(name) && (name = "experiment")
            # Accept .cfg (native), .cmf, or .exp (GEMPACK formats)
            found = findfirst(ext -> isfile(name * ext), [".cfg", ".cmf", ".exp"])
            if found !== nothing
                cfgfile = name * [".cfg", ".cmf", ".exp"][found]
                break
            end
            println("  ✗  Config file not found: \"$(name).cfg\" / .cmf / .exp  — please try again.")
        end
        csvfile = name * ".csv"

        try
            method, steps, raw_shocks, raw_swaps, user_sets, model = parse_config(cfgfile)

            # Load the appropriate data now that we know the model version.
            s_cfg = if model === :v7
                load_data_v7(zippath)[1]
            else
                load_data(zippath)
                _DATA_CACHE[][1]
            end
            shocks, swaps = resolve_config(raw_shocks, raw_swaps, user_sets, s_cfg)

            println("\n  ── Resolved configuration " * "─"^34)
            println("  config  = $cfgfile")
            println("  model   = $model")
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
            sol = if model === :v7
                run_gtap_v7(exp, zippath)
            else
                run_gtap(exp)
            end
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
