# =============================================================================
# load_gtapAgg3_v7.jl
# Loader for GTAP v7 HAR files.
#
# GTAPv7 data archives contain (at minimum):
#   sets.har     – set definitions (REG, COMM, ACTS, ENDW, ECAP, MARG)
#   basedata.har – flow arrays (MAKB, MAKS, EVOS, EVFB, EVFP, VDFB, ...)
#   default.prm  – elasticity parameters (ESBT, ESBC, ESBV, ETRQ, ...)
#
# Returns (s::GTAPSetsV7, d::GTAPDataV7, C::NamedTuple) ready for solving.
#
# Note: GTAPv7 data archives are NOT the same as GTAPAgg3 v6.2 archives.
#       You must supply a zip produced by GTAPAgg3 version ≥ 3.x with
#       the "version 7 model" option, or a hand-assembled v7 data set.
# =============================================================================

using PyCall
using Statistics: mean

# Reuse the Python HAR loader from load_gtapAgg3.jl (already initialised).
# load_har() is already defined in load_gtapAgg3.jl which is included first.

"""
    load_from_zip_v7(zippath) → (s::GTAPSetsV7, d::GTAPDataV7, C)

Load GTAPv7 data from a zip archive and return the (sets, data, derived) tuple
ready for the v7 solver.

The zip must contain (case-insensitive):
  sets.har / gtapsets.har  – set labels
  basedata.har             – value flows
  default.prm              – parameters

HAR header → GTAPDataV7 field mapping:
  MAKB → MAKEB      (nC,nA,nR)   make matrix at basic prices
  MAKS → MAKES      (nC,nA,nR)   make matrix at supplier prices
  EVOS → EVOS       (nE,nA,nR)   endow. supply
  EVFB → EVFB       (nE,nA,nR)   endow. firm demand, basic prices
  EVFP → EVFP       (nE,nA,nR)   endow. firm demand, purch. prices
  VDFB → VDFB       (nC,nA,nR)   dom. interm. demand, basic
  VMFB → VMFB       (nC,nA,nR)   imp. interm. demand, basic
  VMFP → VMFP       (nC,nA,nR)   imp. interm. demand, purch.
  VDPB → VDPB       (nC,nR)
  VMPB → VMPB       (nC,nR)
  VDPP → VDPP       (nC,nR)
  VMPP → VMPP       (nC,nR)
  VDGB → VDGB       (nC,nR)
  VMGB → VMGB       (nC,nR)
  VDGP → VDGP       (nC,nR)
  VMGP → VMGP       (nC,nR)
  VDIB → VDIB       (nC,nR)
  VMIB → VMIB       (nC,nR)
  VDIP → VDIP       (nC,nR)
  VMIP → VMIP       (nC,nR)
  VXSB → VXSB       (nC,nR,nR)
  VFOB → VFOB       (nC,nR,nR)
  VCIF → VCIF       (nC,nR,nR)
  VMSB → VMSB       (nC,nR,nR)
  VST  → VST        (nM,nR)
  VTWR → VTMFSD     (nM,nC,nR,nR)
  SAVE → SAVE       (nR,)
  VDEP → VDEP       (nR,)
  VKB  → VKB        (nR,)
  DPSM → DPARSUM    (nR,)
  ESBT → ESUBT      (nA,nR)
  ESBC → ESUBC      (nA,nR)
  ESBV → ESUBVA     (nA,nR)
  ETRQ → ETRAQ      (nA,nR)
  ESBQ → ESUBQ      (nC,nR)
  ESBD → ESUBD      (nC,)   (or nC,nR → averaged)
  ESBM → ESUBM      (nC,)
  ESBG → ESUBG      (nR,)
  ESBS → ESUBS      (nM,)
  ETRE → ETRAE      (nE,nR)
  INCP → INCPAR     (nC,nR)
  SUBP → SUBPAR     (nC,nR)
  RFLX → RORFLEX   (nR,)
  RDLT → RORDELTA   Int
  EFLG → ENDOWFLAG  (nE,3)
"""
function load_from_zip_v7(zippath::String)
    using_zip = endswith(lowercase(zippath), ".zip")

    # Extract to temp dir if zip
    workdir = if using_zip
        td = mktempdir()
        _init_python(@__DIR__)
        py"""
        import zipfile, os
        with zipfile.ZipFile($zippath) as z:
            z.extractall($td)
        """
        td
    else
        zippath   # treat as a directory
    end

    # Find HAR files (case-insensitive)
    files = Dict{String,String}()
    for f in readdir(workdir; join=true)
        key = lowercase(basename(f))
        files[key] = f
    end
    _find(names...) = begin
        for n in names
            haskey(files, n) && return files[n]
        end
        nothing
    end

    set_file = _find("sets.har", "gtapsets.har")
    dat_file = _find("basedata.har")
    par_file = _find("default.prm", "default.har")

    set_file !== nothing || error("Cannot find sets.har/gtapsets.har in $zippath")
    dat_file !== nothing || error("Cannot find basedata.har in $zippath")
    par_file !== nothing || error("Cannot find default.prm/default.har in $zippath")

    # ── Read raw HAR data ──────────────────────────────────────────────────────
    sets_arrs, _ = load_har(set_file)
    dat_arrs,  _ = load_har(dat_file)
    par_arrs,  _ = load_har(par_file)
    # Also load baserate.har if present (contains ad-valorem rate matrices rTFD, rTFM, etc.)
    rate_file = _find("baserate.har")
    rate_arrs = rate_file !== nothing ? load_har(rate_file)[1] : Dict{String,Any}()
    all_arrs  = merge(sets_arrs, dat_arrs, par_arrs, rate_arrs)

    g(h) = begin
        v = get(all_arrs, h, nothing)
        v !== nothing && return v
        v = get(all_arrs, lowercase(h), nothing)
        v !== nothing && return v
        v = get(all_arrs, uppercase(h), nothing)
        v !== nothing && return v
        error("HAR header '$h' not found in data files")
    end
    gn(h, default) = begin
        v = get(all_arrs, h, get(all_arrs, lowercase(h), get(all_arrs, uppercase(h), nothing)))
        v !== nothing ? v : default
    end

    # ── Sets ───────────────────────────────────────────────────────────────────
    # GTAPAgg3 v7 uses descriptive header names (not H1/H2/H6).
    REG  = Vector{String}(g("REG"))
    COMM = Vector{String}(g("COMM"))
    ACTS = Vector{String}(g("ACTS"))
    ENDW = Vector{String}(g("ENDW"))

    # Capital endowment: try ENDC first, fall back to ECAP
    ECAP_raw = gn("ENDC", gn("ECAP", nothing))
    ECAP = ECAP_raw !== nothing ? Vector{String}(ECAP_raw) :
           filter(e -> occursin(r"(?i)capital", e), ENDW)   # last-resort: grep

    # Margin commodities: read from MARG set in sets.har; fall back to intersection
    MARG_raw = gn("MARG", nothing)
    MARG = if MARG_raw !== nothing
        filter(m -> m ∈ COMM, Vector{String}(MARG_raw))
    else
        filter(m -> m ∈ COMM, ["Air", "Water", "OTP"])
    end

    nC = length(COMM);  nA = length(ACTS);  nR = length(REG)
    nE = length(ENDW);  nM = length(MARG)

    # ── ENDOWFLAG matrix (nE × 3: Mobile / Sluggish / Fixed) ──────────────────
    # GTAPAgg3 v7 stores endowment subsets as ENDM, ENDS, ENDF in sets.har.
    ENDM_set = gn("ENDM", gn("ENDWM", String[]))   # mobile
    ENDS_set = gn("ENDS", gn("ENDWS", String[]))   # sluggish
    ENDF_set = gn("ENDF", gn("ENDWF", String[]))   # fixed / sector-specific

    mobile_set   = Vector{String}(ENDM_set)
    sluggish_set = Vector{String}(ENDS_set)
    fixed_set    = Vector{String}(ENDF_set)

    # If none of ENDM/ENDS/ENDF found, try reading EFLG matrix directly
    if isempty(mobile_set) && isempty(sluggish_set) && isempty(fixed_set)
        EFLG_raw = gn("EFLG", nothing)
        ENDOWFLAG_int = if EFLG_raw !== nothing && ndims(EFLG_raw) == 2
            Int.(round.(EFLG_raw))
        else
            # Ultimate fallback: treat all as mobile except last (sluggish capital)
            @warn "Cannot determine endowment types — defaulting to all mobile"
            hcat(ones(Int, nE), zeros(Int, nE), zeros(Int, nE))
        end
    else
        ENDOWFLAG_int = hcat(
            [e ∈ mobile_set   ? 1 : 0 for e in ENDW],
            [e ∈ sluggish_set ? 1 : 0 for e in ENDW],
            [e ∈ fixed_set    ? 1 : 0 for e in ENDW],
        )
    end

    s = build_sets_v7(REG, COMM, ACTS, MARG, ENDW, ENDOWFLAG_int, ECAP)

    # ── Helper: ensure array has expected shape ────────────────────────────────
    function _reshape_to(arr, shape, header)
        length(arr) == prod(shape) || error(
            "Header '$header': expected $(prod(shape)) elements $shape, got $(length(arr))")
        reshape(arr, shape)
    end

    # ── Value flows ────────────────────────────────────────────────────────────
    # MAKS (make matrix at supplier prices): fall back to MAKB when absent.
    MAKEB  = _reshape_to(Float64.(g("MAKB")), (nC, nA, nR), "MAKB")
    MAKES  = if gn("MAKS", nothing) !== nothing
        _reshape_to(Float64.(g("MAKS")), (nC, nA, nR), "MAKS")
    else
        @info "MAKS not found — using MAKB as supplier-price make matrix"
        copy(MAKEB)
    end

    EVOS   = _reshape_to(Float64.(g("EVOS")), (nE, nA, nR), "EVOS")
    EVFB   = _reshape_to(Float64.(g("EVFB")), (nE, nA, nR), "EVFB")
    EVFP   = _reshape_to(Float64.(g("EVFP")), (nE, nA, nR), "EVFP")
    VMFB   = _reshape_to(Float64.(g("VMFB")), (nC, nA, nR), "VMFB")
    VMFP   = _reshape_to(Float64.(g("VMFP")), (nC, nA, nR), "VMFP")

    # VDFB (domestic intermediate demand at basic prices):
    # Newer GTAPAgg3 archives store only VDFP (at purchaser prices).
    # Recover VDFB using the ad-valorem intermediate tax rate rTFD from baserate.har:
    #   VDFP = VDFB * (1 + rTFD/100)  →  VDFB = VDFP / (1 + rTFD/100)
    VDFP_arr = _reshape_to(Float64.(g("VDFP")), (nC, nA, nR), "VDFP")
    VDFB   = if gn("VDFB", nothing) !== nothing
        _reshape_to(Float64.(g("VDFB")), (nC, nA, nR), "VDFB")
    else
        rTFD_raw = gn("rTFD", gn("RTFD", nothing))
        if rTFD_raw !== nothing
            @info "VDFB not found — deriving from VDFP and rTFD (baserate.har)"
            rTFD_mat = _reshape_to(Float64.(rTFD_raw), (nC, nA, nR), "rTFD")
            VDFP_arr ./ max.(1.0 .+ rTFD_mat ./ 100.0, 1e-6)
        else
            @warn "VDFB and rTFD not found — using VDFP as basic-price approximation"
            copy(VDFP_arr)
        end
    end

    VDPB   = _reshape_to(Float64.(g("VDPB")), (nC, nR), "VDPB")
    VMPB   = _reshape_to(Float64.(g("VMPB")), (nC, nR), "VMPB")
    VDPP   = _reshape_to(Float64.(g("VDPP")), (nC, nR), "VDPP")
    VMPP   = _reshape_to(Float64.(g("VMPP")), (nC, nR), "VMPP")
    VDGB   = _reshape_to(Float64.(g("VDGB")), (nC, nR), "VDGB")
    VMGB   = _reshape_to(Float64.(g("VMGB")), (nC, nR), "VMGB")
    VDGP   = _reshape_to(Float64.(g("VDGP")), (nC, nR), "VDGP")
    VMGP   = _reshape_to(Float64.(g("VMGP")), (nC, nR), "VMGP")
    VDIB   = _reshape_to(Float64.(g("VDIB")), (nC, nR), "VDIB")
    VMIB   = _reshape_to(Float64.(g("VMIB")), (nC, nR), "VMIB")
    VDIP   = _reshape_to(Float64.(g("VDIP")), (nC, nR), "VDIP")
    VMIP   = _reshape_to(Float64.(g("VMIP")), (nC, nR), "VMIP")
    VXSB   = _reshape_to(Float64.(g("VXSB")), (nC, nR, nR), "VXSB")
    VFOB   = _reshape_to(Float64.(g("VFOB")), (nC, nR, nR), "VFOB")
    VCIF   = _reshape_to(Float64.(g("VCIF")), (nC, nR, nR), "VCIF")
    VMSB   = _reshape_to(Float64.(g("VMSB")), (nC, nR, nR), "VMSB")
    VST    = _reshape_to(Float64.(g("VST")),  (nM, nR), "VST")
    VTMFSD = _reshape_to(Float64.(g("VTWR")), (nM, nC, nR, nR), "VTWR")

    # SAVE, VKB, VDEP, DPARSUM: present in GTAP12-SA style archives; absent in newer
    # GTAPAgg3 archives.  Derive from available data when not stored explicitly.

    SUBPAR_arr = _reshape_to(Float64.(g("SUBP")), (nC, nR), "SUBP")

    DPARSUM = if gn("DPSM", nothing) !== nothing
        vec(Float64.(g("DPSM")))
    else
        # DPARSUM[r] = sum_c SUBPAR[c,r]  (sum of CDE substitution parameters)
        @info "DPSM not found — computing as sum_c SUBPAR[c,r]"
        vec(sum(SUBPAR_arr, dims=1))
    end

    SAVE = if gn("SAVE", nothing) !== nothing
        vec(Float64.(g("SAVE")))
    else
        # Derive SAVE from the regional budget identity:
        #   INCOME[r] = EVFP.sum(ea,r)         (factor income incl. factor taxes)
        #             + PTAX.sum(ca,r)           (production/output taxes)
        #             + (VMSB-VCIF).sum(s,r)     (import tariff revenue into r)
        #             + XTRV.sum(c,r,d)          (export tax revenue from r)
        #             + (VDIP+VMIP-VDIB-VMIB).sum(c,r)  (investment demand taxes)
        #             + (VDFP-VDFB).sum(ca,r)    (domestic intermediate taxes)
        #             + (VMFP-VMFB).sum(ca,r)    (imported intermediate taxes)
        #   SAVE[r]   = INCOME[r]
        #             - (VDPB+VMPB).sum(c,r)     (private consumption at basic)
        #             - (VDGB+VMGB).sum(c,r)     (govt consumption at basic)
        @info "SAVE not found — computing from budget identity"
        PTAX_raw = gn("PTAX", nothing)
        PTAX_arr = PTAX_raw !== nothing ?
            _reshape_to(Float64.(PTAX_raw), (nC, nA, nR), "PTAX") : zeros(nC, nA, nR)
        XTRV_raw = gn("XTRV", nothing)
        XTRV_arr = XTRV_raw !== nothing ?
            _reshape_to(Float64.(XTRV_raw), (nC, nR, nR), "XTRV") : zeros(nC, nR, nR)
        SAVE_vec = zeros(nR)
        for r in 1:nR
            factor_inc   = sum(EVFP[:,:,r])
            output_tax   = sum(PTAX_arr[:,:,r])
            tariff_rev   = sum(VMSB[:,:,r]) - sum(VCIF[:,:,r])
            export_tax   = sum(XTRV_arr[:,r,:])
            inv_tax      = sum(VDIP[:,r] .- VDIB[:,r]) + sum(VMIP[:,r] .- VMIB[:,r])
            dom_int_tax  = sum(VDFP_arr[:,:,r] .- VDFB[:,:,r])
            imp_int_tax  = sum(VMFP[:,:,r]  .- VMFB[:,:,r])
            priv_basic   = sum(VDPB[:,r]) + sum(VMPB[:,r])
            govt_basic   = sum(VDGB[:,r]) + sum(VMGB[:,r])
            SAVE_vec[r]  = factor_inc + output_tax + tariff_rev + export_tax +
                           inv_tax + dom_int_tax + imp_int_tax - priv_basic - govt_basic
        end
        SAVE_vec
    end

    # VKB (beginning-of-period capital stock) and VDEP (depreciation):
    # When absent, estimate VKB from capital endowment income at a benchmark rate of return
    # and set depreciation at 5% of the capital stock.
    cap_idx = findfirst(e -> occursin(r"(?i)capital", e), ENDW)
    cap_idx === nothing && (cap_idx = length(ENDW))   # last endowment as fallback
    VKB = if gn("VKB", nothing) !== nothing
        vec(Float64.(g("VKB")))
    else
        @info "VKB not found — estimating from capital endowment income (10% return)"
        # Capital income = EVOS[cap,:,r].sum(); VKB = capital income / rate_of_return
        [sum(EVOS[cap_idx,:,r]) / 0.10 for r in 1:nR]
    end
    VDEP = if gn("VDEP", nothing) !== nothing
        vec(Float64.(g("VDEP")))
    else
        @info "VDEP not found — estimating as 5% of VKB"
        VKB .* 0.05
    end

    # ── Elasticity parameters ─────────────────────────────────────────────────
    # Helper: try multiple header names, fall back to a default array
    _try(default, names...) = begin
        for h in names
            v = gn(h, nothing)
            v !== nothing && return Float64.(v)
        end
        default
    end

    # ESUBT (VA vs. intermediate composite): not always in file → default 0 (Leontief)
    ESUBT_raw = _try(zeros(nA, nR), "ESBT", "ESUBT")
    ESUBT   = _reshape_to(ESUBT_raw, (nA, nR), "ESBT")

    # ESUBC (among intermediates): same — default 0 (Leontief fixed proportions)
    ESUBC_raw = _try(zeros(nA, nR), "ESBC", "ESUBC")
    ESUBC   = _reshape_to(ESUBC_raw, (nA, nR), "ESBC")

    # ESUBVA (among primary factors in VA bundle)
    ESUBVA  = _reshape_to(Float64.(g("ESBV")), (nA, nR), "ESBV")

    # ETRAQ (CET output transformation elasticity, negative by convention)
    ETRAQ   = _reshape_to(Float64.(g("ETRQ")), (nA, nR), "ETRQ")

    # ESUBQ (CES sourcing of commodity across activities): not always stored
    # Default: 0 → law of one price (each pca[c,a,r] = pds[c,r]).
    # For a diagonal make matrix (one activity per commodity) this value is irrelevant.
    ESUBQ_raw = _try(zeros(nC, nR), "ESBQ", "ESUBQ")
    ESUBQ   = _reshape_to(ESUBQ_raw, (nC, nR), "ESBQ")

    # ESUBD (dom/imp Armington): RunGTAP reads from "ESBD"; "ESDC" is CEPII alt
    ESUBD_raw = _try(nothing, "ESBD", "ESDC", "ESUBD")
    ESUBD_raw !== nothing || error("Cannot find dom/imp Armington elasticity (ESDC/ESBD)")
    # May be (nC,nR) → keep as matrix; GTAPDataV7 expects (nC,) vector so average over regions
    ESUBD   = if length(ESUBD_raw) == nC
        vec(Float64.(ESUBD_raw))
    elseif length(ESUBD_raw) == nC * nR
        vec(mean(reshape(Float64.(ESUBD_raw), nC, nR), dims=2))
    else
        error("ESDC/ESBD: unexpected size $(size(ESUBD_raw))")
    end

    # ESUBM (import source Armington): stored as ESBM in GTAPAgg3 v7
    ESUBM_raw = _try(nothing, "ESBM", "ESMC", "ESUBM")
    ESUBM_raw !== nothing || error("Cannot find import-source Armington elasticity (ESBM)")
    ESUBM   = if length(ESUBM_raw) == nC
        vec(Float64.(ESUBM_raw))
    elseif length(ESUBM_raw) == nC * nR
        vec(mean(reshape(Float64.(ESUBM_raw), nC, nR), dims=2))
    else
        error("ESBM: unexpected size $(size(ESUBM_raw))")
    end

    ESUBG_r = vec(Float64.(_try(fill(2.0, nR), "ESBG")))   # government CES
    # ESUBS (transport mode substitution): (nM,) vector
    ESUBS_raw = _try(fill(1.0, nM), "ESBS", "ESUBS")
    ESUBS   = length(ESUBS_raw) == nM ? vec(Float64.(ESUBS_raw)) :
              fill(Float64.(ESUBS_raw)[1], nM)   # scalar → broadcast to (nM,)

    ETRAE   = _reshape_to(Float64.(g("ETRE")), (nE, nR), "ETRE")
    INCPAR  = _reshape_to(Float64.(g("INCP")), (nC, nR), "INCP")
    SUBPAR  = _reshape_to(Float64.(g("SUBP")), (nC, nR), "SUBP")
    # RFLX (ROR flexibility): scalar or (nR,); default 1.0 when absent.
    RORFLEX_raw = gn("RFLX", nothing)
    RORFLEX = if RORFLEX_raw !== nothing
        v = vec(Float64.(RORFLEX_raw))
        length(v) == nR ? v : fill(v[1], nR)
    else
        @info "RFLX not found — using standard GTAP default ROR flexibility of 10.0"
        fill(10.0, nR)
    end
    RORDELTA= Int(round(first(Float64.(gn("RDLT", [1.0])))))

    d = GTAPDataV7(
        SAVE, VDPB, VMPB, VDPP, VMPP,
        VDGB, VMGB, VDGP, VMGP,
        VDIB, VMIB, VDIP, VMIP,
        MAKES, MAKEB, EVOS, EVFB, EVFP,
        VDFB, VDFP_arr, VMFB, VMFP,
        VXSB, VFOB, VCIF, VMSB, VST, VTMFSD,
        VKB, VDEP, DPARSUM,
        ESUBT, ESUBC, ESUBVA, ETRAQ, ESUBQ,
        ESUBD, ESUBM, ESUBG_r, ESUBS, ETRAE,
        INCPAR, SUBPAR, RORFLEX, RORDELTA, ENDOWFLAG_int,
    )

    C = compute_derived_v7(d, s)
    return s, d, C
end
