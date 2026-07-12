# =============================================================================
# load_gtapAgg3.jl
# =============================================================================
# Loader for GTAPAgg3 output HAR files (gsdfset.har, gsdfdat.har, gsdfpar.har)
# Maps them into the GTAPData / GTAPSets structs required by gtap_v62.jl.
#
# Prerequisites:
#   • gtap_v62.jl  in the same directory (provides GTAPSets, GTAPData, etc.)
#   • har_reader.py in the same directory (pure-Python HAR reader)
#   • Julia packages: PyCall
#
# Usage:
#   include("gtap_v62.jl")
#   include("load_gtapAgg3.jl")
#   s, d, C = load_gsdf("gsdfset.har", "gsdfdat.har", "gsdfpar.har")
#   model   = build_gtap_mcp(d, s, C)
#   optimize!(model)
#
# ─────────────────────────────────────────────────────────────────────────────
# HEADER MAPPING  (GTAPAgg3  →  standard GTAP.TAB / GTAPData field)
# ─────────────────────────────────────────────────────────────────────────────
#   GTAPAgg3   GTAPData field   Shape            Description
#   ─────────  ───────────────  ───────────────  ───────────────────────────────
#   VDFB       VDFM             (nT, nT, nR)     Dom. intermediates, mkt prices
#   VDFP       VDFA             (nT, nT, nR)     Dom. intermediates, agent prices
#   VMFB       VIFM             (nT, nT, nR)     Imp. intermediates, mkt prices
#   VMFP       VIFA             (nT, nT, nR)     Imp. intermediates, agent prices
#   VDPB       VDPM             (nT, nR)         Dom. private cons., mkt prices
#   VDPP       VDPA             (nT, nR)         Dom. private cons., agent prices
#   VMPB       VIPM             (nT, nR)         Imp. private cons., mkt prices
#   VMPP       VIPA             (nT, nR)         Imp. private cons., agent prices
#   VDGB       VDGM             (nT, nR)         Dom. govt cons., mkt prices
#   VDGP       VDGA             (nT, nR)         Dom. govt cons., agent prices
#   VMGB       VIGM             (nT, nR)         Imp. govt cons., mkt prices
#   VMGP       VIGA             (nT, nR)         Imp. govt cons., agent prices
#   EVFB       VFM              (nE, nT, nR)     Endow. inputs, mkt prices
#   EVFP       EVFA             (nE, nT, nR)     Endow. inputs, agent prices
#   VDIB       VDFM[:,cgds,:]   (nT, nR)         Dom. investment demand, basic prices
#   VDIP       VDFA[:,cgds,:]   (nT, nR)         Dom. investment demand, agent prices
#   VMIB       VIFM[:,cgds,:]   (nT, nR)         Imp. investment demand, basic prices
#   VMIP       VIFA[:,cgds,:]   (nT, nR)         Imp. investment demand, agent prices
#   EVOS       EVOA             (nE, nR)         Endow. supply (sum over acts)
#   VXSB       VXMD             (nT, nR, nR)     Exports at dom. mkt prices
#   VFOB       VXWD             (nT, nR, nR)     Exports FOB
#   VCIF       VIWS             (nT, nR, nR)     Imports CIF
#   VMSB       VIMS             (nT, nR, nR)     Imports at dom. mkt prices
#   VST        VST              (nM, nR)          Margin supply to transport
#   VTWR       VTMFSD           (nM, nT, nR, nR)  Margin usage by route
#   SAVE       SAVE             (nR,)             Net saving
#   VDEP       VDEP             (nR,)             Capital depreciation
#   VKB        VKB              (nR,)             Beginning-of-period capital
#   DPSM       DPARSUM          (nR,)             CDE distribution param. sum
#   ESBV       ESUBVA           (nT,)  ← avg(nT,nR)  Factor substitution elas.
#   ESBD       ESUBD            (nT,)  ← avg(nT,nR)  Dom/imp substitution elas.
#   ESBM       ESUBM            (nT,)  ← avg(nT,nR)  Armington import elas.
#   ETRE       ETRAE            (nE,)  ← avg(nE,nR)  Sluggish endow. transform.
#   INCP       INCPAR           (nT, nR)          CDE expansion parameter
#   SUBP       SUBPAR           (nT, nR)          CDE substitution parameter
#   RFLX       RORFLEX          (nR,)             ROR flexibility
#   RDLT       RORDELTA         scalar Int        Investment allocation switch
#   EFLG       SLUG             (nE,)             Endowment mobility (1=sluggish)
# =============================================================================

using PyCall
using Statistics: mean

# ── Initialise Python HAR reader ────────────────────────────────────────────
const _PY_INITIALIZED = Ref(false)

function _init_python(har_reader_dir::String = @__DIR__)
    if !_PY_INITIALIZED[]
        pushfirst!(PyVector(pyimport("sys")."path"), har_reader_dir)
        _PY_INITIALIZED[] = true
    end
end

"""
    load_har(path; har_reader_dir=@__DIR__) → Dict{String, Array}

Read a GEMPACK HAR file via `har_reader.py`.
"""
function load_har(path::String; har_reader_dir::String = @__DIR__)
    _init_python(har_reader_dir)
    py"""
    import numpy as _np
    from har_reader import read_har as _rh
    _raw, _sets = _rh($path)
    _num  = {k: v for k, v in _raw.items() if not _np.issubdtype(v.dtype, _np.str_)}
    _strs = {k: v.tolist() for k, v in _raw.items() if _np.issubdtype(v.dtype, _np.str_)}
    """
    arrs = Dict{String, Any}(k => Array(v) for (k, v) in py"_num")
    strs_from_raw = Dict{String, Vector{String}}(k => Vector{String}(v) for (k, v) in py"_strs")
    sets_dict     = Dict{String, Vector{String}}(k => Vector{String}(v) for (k, v) in py"_sets")
    merge!(arrs, strs_from_raw)
    return arrs, sets_dict
end

# ── Set loading ──────────────────────────────────────────────────────────────

"""
    load_gtapAgg3_sets(set_file; endwc="Capital", har_reader_dir=@__DIR__)
        → GTAPSets
"""
function load_gtapAgg3_sets(set_file::String;
                             endwc::String        = "Capital",
                             har_reader_dir::String = @__DIR__)

    s, _ = load_har(set_file; har_reader_dir)

    REG       = String.(vec(s["REG"]))
    COMM      = String.(vec(s["COMM"]))
    MARG      = String.(vec(s["MARG"]))
    ENDW      = String.(vec(s["ENDW"]))
    ENDWS_raw = String.(vec(s["ENDS"]))   # header "ENDS" = Set ENDWS (sluggish)

    SLUG = [e ∈ ENDWS_raw ? 1 : 0 for e in ENDW]

    # GTAPAgg3 folds cgds into ACTS; we create a synthetic singleton
    return build_sets(REG, COMM, MARG, ["cgds"], ENDW, SLUG, [endwc])
end

# ── Data + parameter loading ──────────────────────────────────────────────────

"""
    load_gtapAgg3(dat_file, par_file, sets;
                  esubva_source = :regional,
                  esubd_source  = :regional,
                  esubm_source  = :regional,
                  har_reader_dir = @__DIR__)
        → GTAPData

Map GTAPAgg3 HAR arrays into a GTAPData struct.

`*_source` keywords choose whether to use region-specific elasticities
(`:regional`) or the global average (`:global`) for ESUBVA, ESUBD, ESUBM.
`:regional` averages the (COMM,REG) matrix over regions to get a (COMM,)
vector, which is what standard GTAP v6.2 uses.
"""
function load_gtapAgg3(dat_file::String, par_file::String, sets::GTAPSets;
                        esubva_source::Symbol  = :regional,
                        esubd_source::Symbol   = :regional,
                        esubm_source::Symbol   = :regional,
                        har_reader_dir::String = @__DIR__)

    d, _ = load_har(dat_file; har_reader_dir)
    p, _ = load_har(par_file; har_reader_dir)

    nT = length(sets.TRAD_COMM)    # 65 commodities
    nE = length(sets.ENDW_COMM)    # 5 endowments
    nR = length(sets.REG)           # 30 regions
    nM = length(sets.MARG_COMM)    # 3 margin commodities

    # ── Helper: assert shape and return Float64 array ─────────────────────
    function chk(key, src, expected_shape)
        arr = src[key]
        got = size(arr)
        if got != expected_shape
            error("$key: got shape $got, expected $expected_shape")
        end
        return Float64.(arr)
    end
    chkD(k, s) = chk(k, d, s)
    chkP(k, s) = chk(k, p, s)

    # ── Flow matrices ─────────────────────────────────────────────────────

    _VDFM = chkD("VDFB", (nT, nT, nR))   # domestic intermediates, market prices
    _VDFA = chkD("VDFP", (nT, nT, nR))   # domestic intermediates, agent prices
    _VIFM = chkD("VMFB", (nT, nT, nR))   # imported intermediates, market prices
    _VIFA = chkD("VMFP", (nT, nT, nR))   # imported intermediates, agent prices

    VDPM  = chkD("VDPB", (nT, nR))
    VDPA  = chkD("VDPP", (nT, nR))
    VIPM  = chkD("VMPB", (nT, nR))
    VIPA  = chkD("VMPP", (nT, nR))

    VDGM  = chkD("VDGB", (nT, nR))
    VDGA  = chkD("VDGP", (nT, nR))
    VIGM  = chkD("VMGB", (nT, nR))
    VIGA  = chkD("VMGP", (nT, nR))

    # Investment demand by commodity — fills the synthetic cgds column
    VDIB  = chkD("VDIB", (nT, nR))   # domestic investment, basic (market) prices
    VDIP  = chkD("VDIP", (nT, nR))   # domestic investment, agent (purchaser) prices
    VMIB  = chkD("VMIB", (nT, nR))   # imported investment, basic (market) prices
    VMIP  = chkD("VMIP", (nT, nR))   # imported investment, agent (purchaser) prices

    # GSDF stores factor payments over nT activities; GTAPData expects nP=nT+1 (cgds column)
    # Factor inputs to cgds = 0 (investment good assembled from commodities, not factors)
    _pad3z(A) = cat(A, zeros(size(A,1), 1, size(A,3)), dims=2)
    VFM   = _pad3z(chkD("EVFB", (nE, nT, nR)))   # (nE, nP, nR)
    EVFA  = _pad3z(chkD("EVFP", (nE, nT, nR)))   # (nE, nP, nR)

    # Intermediate demand: pad (nT,nT,nR) → (nT,nP,nR), cgds column = investment demand
    function _pad3inv(A, inv_nT_nR)
        B = similar(A, size(A,1), size(A,2)+1, size(A,3))
        B[:, 1:end-1, :] = A
        for i in 1:size(A,1), r in 1:size(A,3)
            B[i, end, r] = inv_nT_nR[i, r]
        end
        return B
    end
    VDFM  = _pad3inv(_VDFM, VDIB)
    VDFA  = _pad3inv(_VDFA, VDIP)
    VIFM  = _pad3inv(_VIFM, VMIB)
    VIFA  = _pad3inv(_VIFA, VMIP)

    # EVOA: standard GTAP is (ENDW, REG); GTAPAgg3 has (ENDW, ACTS, REG) → sum over acts
    EVOS_full = chkD("EVOS", (nE, nT, nR))
    EVOA = dropdims(sum(EVOS_full, dims=2), dims=2)   # (nE, nR)

    # Trade flows (COMM, REG_src, REG_dest)
    VXMD  = chkD("VXSB", (nT, nR, nR))   # exports at dom. mkt prices
    VXWD  = chkD("VFOB", (nT, nR, nR))   # exports FOB
    VIWS  = chkD("VCIF", (nT, nR, nR))   # imports CIF
    VIMS  = chkD("VMSB", (nT, nR, nR))   # imports at dom. mkt prices

    VST    = Float64.(d["VST"])            # (nM, nR)
    VTMFSD = Float64.(d["VTWR"])          # (nM, nT, nR, nR)

    SAVE    = Float64.(vec(d["SAVE"]))
    VDEP    = Float64.(vec(d["VDEP"]))
    VKB     = Float64.(vec(d["VKB"]))
    DPARSUM = Float64.(vec(d["DPSM"]))

    # ── Elasticities ────────────────────────────────────────────────────────
    # GTAPAgg3 stores elasticities as (COMM, REG) matrices (region-specific).
    # Standard GTAP v6.2 uses (COMM,) vectors (averaged over regions).

    function pick_elas(global_key, regional_key, ndim_exp)
        if esubva_source == :regional && haskey(p, regional_key)
            raw = Float64.(p[regional_key])
        else
            raw = Float64.(p[global_key])
        end
        ndims(raw) == 1 && return raw
        return vec(mean(raw, dims=2))   # average over regions
    end

    ESUBVA = let
        key = (esubva_source == :regional && haskey(p, "SBVR")) ? "SBVR" : "ESBV"
        v = vec(mean(Float64.(p[key]), dims=2))   # (nT,nR) → (nT,)
        vcat(v, 0.0)   # append zero for cgds → length nP
    end

    ESUBD = let
        key = (esubd_source == :regional && haskey(p, "SBDR")) ? "SBDR" : "ESBD"
        arr = Float64.(p[key])
        ndims(arr) == 1 ? arr : vec(mean(arr, dims=2))
    end

    ESUBM = let
        key = (esubm_source == :regional && haskey(p, "SBMR")) ? "SBMR" : "ESBM"
        arr = Float64.(p[key])
        ndims(arr) == 1 ? arr : vec(mean(arr, dims=2))
    end

    # ESUBT: top-level production elasticity (= 0 in standard GTAP v6.2 Leontief)
    # GTAPAgg3 stores ESBT (top-level) — use if present, else zeros
    ESUBT_trad = if haskey(p, "ESBT")
        arr = Float64.(p["ESBT"])
        ndims(arr) == 1 ? arr : vec(mean(arr, dims=2))
    else
        zeros(nT)
    end
    ESUBT = vcat(ESUBT_trad, 0.0)   # append zero for synthetic cgds sector → length nP

    # ETRAE: (ENDW, REG) → (ENDW,) by averaging over regions
    ETRAE = let
        arr = Float64.(p["ETRE"])
        ndims(arr) == 1 ? arr : vec(mean(arr, dims=2))
    end

    INCPAR  = chkP("INCP", (nT, nR))
    SUBPAR  = chkP("SUBP", (nT, nR))
    RORFLEX = Float64.(vec(p["RFLX"]))

    # RORDELTA: binary investment-allocation switch
    RORDELTA = haskey(p, "RDLT") ? Int(p["RDLT"][1]) : 1

    # SLUG: derived from set membership (ENDWS_COMM populated from gsdfset.har)
    SLUG = [e ∈ sets.ENDWS_COMM ? 1 : 0 for e in sets.ENDW_COMM]

    return GTAPData(
        SAVE,
        VDGA, VDGM, VIGA, VIGM,
        VDPA, VDPM, VIPA, VIPM,
        EVOA,
        EVFA,
        VDFA, VDFM, VIFA, VIFM,
        VFM,
        VXMD, VXWD, VIWS, VIMS,
        VST,
        VTMFSD,
        VKB,
        VDEP,
        DPARSUM,
        ESUBD, ESUBT, ESUBVA, ESUBM,
        ETRAE,
        INCPAR, SUBPAR,
        RORFLEX,
        RORDELTA,
        SLUG,
    )
end

# ── One-call convenience loader ──────────────────────────────────────────────

"""
    load_gsdf(set_file, dat_file, par_file; endwc="Capital", kwargs...)
        → (GTAPSets, GTAPData, NamedTuple)

Load all three GTAPAgg3 HAR files, build sets, data, and derived coefficients.

Example:
    s, d, C = load_gsdf("gsdfset.har", "gsdfdat.har", "gsdfpar.har")
    check_balance(d, s, C)               # print accounting checks
"""
function load_gsdf(set_file::String, dat_file::String, par_file::String;
                   endwc::String = "Capital",
                   har_reader_dir::String = @__DIR__,
                   kwargs...)

    println("Loading sets …  $(basename(set_file))")
    sets = load_gtapAgg3_sets(set_file; endwc, har_reader_dir)
    println("  $(length(sets.REG)) regions  ×  $(length(sets.TRAD_COMM)) commodities  "  *
            "×  $(length(sets.ENDW_COMM)) endowments  ×  $(length(sets.MARG_COMM)) margins")

    println("Loading data …  $(basename(dat_file))")
    data = load_gtapAgg3(dat_file, par_file, sets; har_reader_dir, kwargs...)

    println("Computing derived coefficients …")
    C = compute_derived(data, sets)

    return sets, data, C
end

# ── Zip-based loader ────────────────────────────────────────────────────────────

"""
    load_from_zip(zippath; endwc="Capital", har_reader_dir=@__DIR__) → (GTAPSets, GTAPData, NamedTuple)

Load GTAP data from a GTAPAgg3 zip archive.  The archive must contain (anywhere
in its directory tree, case-insensitive):

  sets.har      — set definitions  (REG, COMM, MARG, ENDW, ENDS)
  basedata.har  — flow data
  default.prm   — elasticities and other parameters

The zip is extracted to a temporary directory that is deleted after loading.
"""
function load_from_zip(zippath::String;
                       endwc          ::String = "Capital",
                       har_reader_dir ::String = @__DIR__)

    isfile(zippath) || error("Zip file not found: $zippath")

    tmpdir = mktempdir()
    try
        _init_python(har_reader_dir)
        py"""
        import zipfile as _zf
        with _zf.ZipFile($zippath, 'r') as _z:
            _z.extractall($tmpdir)
        """

        sets_f = _find_in_tree(tmpdir, ["sets.har"])
        dat_f  = _find_in_tree(tmpdir, ["basedata.har"])
        par_f  = _find_in_tree(tmpdir, ["default.prm"])

        println("  sets    : $(relpath(sets_f, tmpdir))")
        println("  data    : $(relpath(dat_f,  tmpdir))")
        println("  params  : $(relpath(par_f,  tmpdir))")

        return load_gsdf(sets_f, dat_f, par_f; endwc, har_reader_dir)
    finally
        rm(tmpdir; recursive = true, force = true)
    end
end

# Recursive case-insensitive file search; returns first match.
function _find_in_tree(root::String, candidates::Vector{String})
    lower = lowercase.(candidates)
    for (dir, _, files) in walkdir(root)
        for f in files
            if lowercase(f) in lower
                return joinpath(dir, f)
            end
        end
    end
    error("Cannot find any of [$(join(candidates, ", "))] inside $root")
end

# ── Accounting sanity check ───────────────────────────────────────────────────

"""
    check_balance(d, s, C)

Print key accounting identities to verify the loaded data is consistent.
"""
function check_balance(d::GTAPData, s::GTAPSets, C)
    println("\n── Accounting checks ────────────────────────────────────────")

    fob  = sum(d.VXWD)
    cif  = sum(d.VIWS)
    tpt  = sum(d.VST)
    println("  Global exports (FOB)   : $(round(fob/1e6, digits=1)) bn USD")
    println("  Global imports (CIF)   : $(round(cif/1e6, digits=1)) bn USD")
    println("  Global transport VST   : $(round(tpt/1e6, digits=3)) bn USD")
    println("  CIF - FOB - VST        : $(round((cif-fob-tpt)/1e6, digits=3)) bn USD  (≈ 0 if balanced)")

    glob_save = sum(d.SAVE)
    glob_inv  = C.GLOBINV
    println("  Global net saving      : $(round(glob_save/1e6, digits=1)) bn USD")
    println("  Global net investment  : $(round(glob_inv/1e6,  digits=1)) bn USD")
    println("  Difference             : $(round((glob_save-glob_inv)/1e6, digits=3)) bn USD  (≈ 0 if balanced)")

    income_err = maximum(abs.(C.INCOME .- (C.PRIVEXP .+ C.GOVEXP .+ d.SAVE)))
    println("  Max income residual    : $(round(income_err, digits=4))  (= 0 by construction)")
    println("────────────────────────────────────────────────────────────")
end
