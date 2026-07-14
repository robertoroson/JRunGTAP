# =============================================================================
# gtap_v7.jl
# Julia translation of the GTAP model, Version 7 (Corong et al., 2017)
#
# Reference: Corong, E. et al. (2017), "The Standard GTAP Model, Version 7",
#            Journal of Global Economic Analysis 2(1): 1-119.
#
# Key structural changes from v6.2:
#   - ACTS (activities) distinct from COMM (commodities)
#   - Multi-product activities via CET/CES 'make' matrix (MAKES, MAKEB)
#   - Three endowment types: Mobile, Sluggish, Fixed (via ENDOWFLAG)
#   - Activity-specific taxes: to(c,a,r), tinc(e,a,r), tfe(e,a,r)
#   - New nesting: composite intermediate (ESUBC), VA vs. intermediate (ESUBT)
#   - Government demand with CES (ESUBG(r)) instead of Leontief
#   - Investment demand: Leontief (qia = qinv)
#   - Domestic price pds(c,r), composite import price pms(c,r)
#   - Numeraire: pfactor(r) → pfactwld (same as v6.2 appendix)
#
# This file provides:
#   Part 1 – GTAPSetsV7, GTAPDataV7, build_sets_v7, compute_derived_v7
#   Part 2 – gtap_v7_residuals
# =============================================================================

using LinearAlgebra, SparseArrays

# =============================================================================
# PART 1 – SETS, DATA STRUCTS, AND DERIVED COEFFICIENTS
# =============================================================================

struct GTAPSetsV7
    REG    ::Vector{String}   # regions
    COMM   ::Vector{String}   # traded commodities
    ACTS   ::Vector{String}   # production activities
    MARG   ::Vector{String}   # margin commodities ⊆ COMM
    NMRG   ::Vector{String}   # non-margin = COMM - MARG
    ENDW   ::Vector{String}   # all endowment factors
    ENDWM  ::Vector{String}   # mobile endowments
    ENDWS  ::Vector{String}   # sluggish endowments
    ENDWF  ::Vector{String}   # sector-specific (fixed) endowments
    ENDWMS ::Vector{String}   # mobile + sluggish = ENDWM ∪ ENDWS
    ENDWC  ::Vector{String}   # capital endowment ⊆ ENDWM
    DEMD   ::Vector{String}   # ENDW ∪ COMM (demanded factors)
end

"""
Build v7 sets from raw vectors and ENDOWFLAG matrix.

`ENDOWFLAG` is a (nE × 3) matrix where columns are Mobile, Sluggish, Fixed.
Rows with ENDOWFLAG[e,1]=1 → mobile; [e,2]=1 → sluggish; [e,3]=1 → fixed.
"""
function build_sets_v7(REG, COMM, ACTS, MARG, ENDW, ENDWFLAG::Matrix{Int},
                       ENDWC::Vector{String})
    NMRG   = [c for c in COMM if c ∉ MARG]
    ENDWM  = ENDW[ENDWFLAG[:,1] .> 0]
    ENDWS  = ENDW[ENDWFLAG[:,2] .> 0]
    ENDWF  = ENDW[ENDWFLAG[:,3] .> 0]
    ENDWMS = vcat(ENDWM, ENDWS)
    DEMD   = vcat(ENDW, COMM)
    return GTAPSetsV7(REG, COMM, ACTS, MARG, NMRG, ENDW, ENDWM, ENDWS, ENDWF,
                      ENDWMS, ENDWC, DEMD)
end

"""Raw base-data and parameter arrays for GTAP v7 (read from HAR files)."""
struct GTAPDataV7
    # ── Value flows ──────────────────────────────────────────────────────────
    SAVE   ::Vector{Float64}         # (nR)         net saving
    VDPB   ::Matrix{Float64}         # (nC,nR)      priv. dom. demand, basic prices
    VMPB   ::Matrix{Float64}         # (nC,nR)      priv. imp. demand, basic prices
    VDPP   ::Matrix{Float64}         # (nC,nR)      priv. dom. demand, purch. prices
    VMPP   ::Matrix{Float64}         # (nC,nR)      priv. imp. demand, purch. prices
    VDGB   ::Matrix{Float64}         # (nC,nR)      gov. dom. demand, basic prices
    VMGB   ::Matrix{Float64}         # (nC,nR)      gov. imp. demand, basic prices
    VDGP   ::Matrix{Float64}         # (nC,nR)      gov. dom. demand, purch. prices
    VMGP   ::Matrix{Float64}         # (nC,nR)      gov. imp. demand, purch. prices
    VDIB   ::Matrix{Float64}         # (nC,nR)      inv. dom. demand, basic prices
    VMIB   ::Matrix{Float64}         # (nC,nR)      inv. imp. demand, basic prices
    VDIP   ::Matrix{Float64}         # (nC,nR)      inv. dom. demand, purch. prices
    VMIP   ::Matrix{Float64}         # (nC,nR)      inv. imp. demand, purch. prices
    MAKES  ::Array{Float64,3}        # (nC,nA,nR)   make matrix, supplier prices
    MAKEB  ::Array{Float64,3}        # (nC,nA,nR)   make matrix, basic prices
    EVOS   ::Array{Float64,3}        # (nE,nA,nR)   endow. supply at supply prices
    EVFB   ::Array{Float64,3}        # (nE,nA,nR)   endow. firm demand, basic prices
    EVFP   ::Array{Float64,3}        # (nE,nA,nR)   endow. firm demand, purch. prices
    VDFB   ::Array{Float64,3}        # (nC,nA,nR)   dom. interm. demand, basic prices
    VMFB   ::Array{Float64,3}        # (nC,nA,nR)   imp. interm. demand, basic prices
    VMFP   ::Array{Float64,3}        # (nC,nA,nR)   imp. interm. demand, purch. prices
    VXSB   ::Array{Float64,3}        # (nC,nR,nR)   exports at basic prices
    VFOB   ::Array{Float64,3}        # (nC,nR,nR)   exports at FOB prices
    VCIF   ::Array{Float64,3}        # (nC,nR,nR)   imports at CIF prices
    VMSB   ::Array{Float64,3}        # (nC,nR,nR)   imports at domestic basic prices
    VST    ::Matrix{Float64}         # (nM,nR)      intl. margin supply
    VTMFSD ::Array{Float64,4}        # (nM,nC,nR,nR) intl. margin usage
    VKB    ::Vector{Float64}         # (nR)         beginning capital stock
    VDEP   ::Vector{Float64}         # (nR)         depreciation
    DPARSUM::Vector{Float64}         # (nR)         sum of distribution parameters
    # ── Elasticity parameters ────────────────────────────────────────────────
    ESUBT  ::Matrix{Float64}         # (nA,nR)  VA vs. intermediate
    ESUBC  ::Matrix{Float64}         # (nA,nR)  among intermediates
    ESUBVA ::Matrix{Float64}         # (nA,nR)  among primary factors
    ETRAQ  ::Matrix{Float64}         # (nA,nR)  CET output transformation (<0)
    ESUBQ  ::Matrix{Float64}         # (nC,nR)  CES commodity sourcing
    ESUBD  ::Vector{Float64}         # (nC)     dom/imp substitution
    ESUBM  ::Vector{Float64}         # (nC)     Armington import sources
    ESUBG  ::Vector{Float64}         # (nR)     gov't commodity substitution
    ESUBS  ::Vector{Float64}         # (nM)     transport mode substitution
    ETRAE  ::Matrix{Float64}         # (nE,nR)  sluggish endowment CET (<0)
    INCPAR ::Matrix{Float64}         # (nC,nR)  CDE income parameter
    SUBPAR ::Matrix{Float64}         # (nC,nR)  CDE substitution parameter
    RORFLEX::Vector{Float64}         # (nR)
    RORDELTA::Int
    ENDOWFLAG::Matrix{Int}           # (nE,3)  Mobile/Sluggish/Fixed flags
end

# =============================================================================
# DERIVED COEFFICIENTS (computed once from GTAPDataV7 before solving)
# =============================================================================

function compute_derived_v7(d::GTAPDataV7, s::GTAPSetsV7)
    nC  = length(s.COMM);   nA  = length(s.ACTS);   nR  = length(s.REG)
    nM  = length(s.MARG);   nE  = length(s.ENDW)
    nEM = length(s.ENDWM);  nES = length(s.ENDWS);  nEF = length(s.ENDWF)
    nEMS= length(s.ENDWMS); nEC = length(s.ENDWC)

    # index helpers
    iC(x) = findfirst(==(x), s.COMM)::Int
    iA(x) = findfirst(==(x), s.ACTS)::Int
    iE(x) = findfirst(==(x), s.ENDW)::Int
    iM(x) = findfirst(==(x), s.MARG)::Int
    iR(x) = findfirst(==(x), s.REG)::Int
    iD(x) = findfirst(==(x), s.DEMD)::Int
    iEM(x)= findfirst(==(x), s.ENDWM)::Int
    iES(x)= findfirst(==(x), s.ENDWS)::Int
    iEMS(x)=findfirst(==(x), s.ENDWMS)::Int
    iEC(x)= findfirst(==(x), s.ENDWC)::Int

    isMARG = Bool[c ∈ s.MARG for c in s.COMM]

    # ── Make matrix aggregates ───────────────────────────────────────────────
    MAKESCOM = dropdims(sum(d.MAKES, dims=1), dims=1)  # (nA, nR)

    MAKESACTSHR = zeros(nC, nA, nR)   # MAKES(c,a,r) / MAKESCOM(a,r)
    for a in 1:nA, r in 1:nR
        denom = max(MAKESCOM[a,r], 1e-10)
        MAKESACTSHR[:,a,r] = d.MAKES[:,a,r] ./ denom
    end

    MAKEBACT = dropdims(sum(d.MAKEB, dims=1), dims=1)  # (nA,nR)
    MAKEBCOM = dropdims(sum(d.MAKEB, dims=2), dims=2)  # (nC,nR)

    MAKEBACTSHR = zeros(nC, nA, nR)
    for a in 1:nA, r in 1:nR
        denom = max(MAKEBACT[a,r], 1e-10)
        MAKEBACTSHR[:,a,r] = d.MAKEB[:,a,r] ./ denom
    end

    MAKEBCOMSHR = zeros(nC, nA, nR)
    for c in 1:nC, r in 1:nR
        denom = max(MAKEBCOM[c,r], 1e-10)
        MAKEBCOMSHR[c,:,r] = d.MAKEB[c,:,r] ./ denom
    end

    # ── Value of commodity supply at basic prices (VCB = MAKEBCOM) ──────────
    VCB = MAKEBCOM   # (nC, nR)

    # ── Endowment supply aggregates ─────────────────────────────────────────
    VES = dropdims(sum(d.EVOS, dims=2), dims=2)  # (nE, nR)  sum_a EVOS(e,a,r)

    GROSSCAP = zeros(nR)
    for e in s.ENDWC; GROSSCAP .+= VES[iE(e),:]; end

    # ── Total domestic demand for commodity c at basic prices ───────────────
    # VDB(c,r) = sum_a VDFB(c,a,r) + VDPB(c,r) + VDGB(c,r) + VDIB(c,r)
    VDB = zeros(nC, nR)
    for c in 1:nC, r in 1:nR
        VDB[c,r] = sum(d.VDFB[c,a,r] for a in 1:nA) + d.VDPB[c,r] + d.VDGB[c,r] + d.VDIB[c,r]
    end

    # ── Total aggregate imports at basic prices ──────────────────────────────
    # VMB(c,r) = sum_a VMFB(c,a,r) + VMIB(c,r) + VMPB(c,r) + VMGB(c,r)
    VMB = zeros(nC, nR)
    for c in 1:nC, r in 1:nR
        VMB[c,r] = sum(d.VMFB[c,a,r] for a in 1:nA) + d.VMIB[c,r] + d.VMPB[c,r] + d.VMGB[c,r]
    end

    # ── VFB(c,a,r) = VDFB + VMFB (total basic-price intermediate demand) ────
    VFB_com = d.VDFB .+ d.VMFB   # (nC, nA, nR)

    # ── STC: cost share of input d in total costs of act. a at basic prices ─
    # For endowments: EVFB(e,a,r); for commodities: VDFB(c,a,r)+VMFB(c,a,r)
    # VOS is at supplier prices; STC normalized to VOS
    STC_endw = zeros(nE, nA, nR)
    STC_comm = zeros(nC, nA, nR)
    for a in 1:nA, r in 1:nR
        denom = max(MAKESCOM[a,r], 1e-10)
        STC_endw[:,a,r] = d.EVFB[:,a,r] ./ denom
        STC_comm[:,a,r] = VFB_com[:,a,r] ./ denom
    end

    # ── VVA: value added = sum_e EVFB(e,a,r) ───────────────────────────────
    VVA = dropdims(sum(d.EVFB, dims=1), dims=1)  # (nA, nR)

    # ── SVA: share of endowment e in VA of act. a in r ──────────────────────
    DEFAULTVASHR = zeros(nE)
    denom_sva = max(sum(VVA), 1e-10)
    for ei in 1:nE; DEFAULTVASHR[ei] = sum(d.EVFB[ei,:,:]) / denom_sva; end

    SVA = zeros(nE, nA, nR)
    for ei in 1:nE, a in 1:nA, r in 1:nR
        SVA[ei,a,r] = VVA[a,r] > 0 ? d.EVFB[ei,a,r] / VVA[a,r] : DEFAULTVASHR[ei]
    end

    # ── INTSHR: share of c in total interm. inputs of act. a ───────────────
    TINTCOM = dropdims(sum(VFB_com, dims=1), dims=1)   # (nA, nR)
    INTSHR = zeros(nC, nA, nR)
    for a in 1:nA, r in 1:nR
        denom = max(TINTCOM[a,r], 1e-10)
        INTSHR[:,a,r] = VFB_com[:,a,r] ./ denom
    end

    # ── FMSHR: share of imports in firm demand for c by act. a ─────────────
    FMSHR = zeros(nC, nA, nR)
    for c in 1:nC, a in 1:nA, r in 1:nR
        denom = max(VFB_com[c,a,r], 1e-10)
        FMSHR[c,a,r] = d.VMFB[c,a,r] / denom
    end

    # ── ENDWMSHR: share of mobile endowment e used by act. a ────────────────
    ENDWMSHR = zeros(nEM, nA, nR)
    for (mi,m) in enumerate(s.ENDWM)
        ei = iE(m)
        for a in 1:nA, r in 1:nR
            ENDWMSHR[mi,a,r] = VES[ei,r] > 0 ? d.EVOS[ei,a,r] / VES[ei,r] : 0.0
        end
    end

    # ── REVSHR: share of endowment e in total supply of e (for sluggish) ───
    REVSHR = zeros(nE, nA, nR)
    for ei in 1:nE, a in 1:nA, r in 1:nR
        REVSHR[ei,a,r] = VES[ei,r] > 0 ? d.EVOS[ei,a,r] / VES[ei,r] : 0.0
    end

    # ── Domestic demand shares for commodity c ───────────────────────────────
    FDCSHR = zeros(nC, nA, nR)   # share of c used by act. a in dom. demand
    for c in 1:nC, a in 1:nA, r in 1:nR
        FDCSHR[c,a,r] = VDB[c,r] > 0 ? d.VDFB[c,a,r] / VDB[c,r] : 0.0
    end
    PDCSHR = zeros(nC, nR)
    GDCSHR = zeros(nC, nR)
    IDCSHR = zeros(nC, nR)
    for c in 1:nC, r in 1:nR
        denom = max(VDB[c,r], 1e-10)
        PDCSHR[c,r] = d.VDPB[c,r] / denom
        GDCSHR[c,r] = d.VDGB[c,r] / denom
        IDCSHR[c,r] = d.VDIB[c,r] / denom
    end

    # ── Import demand shares for commodity c ────────────────────────────────
    FMCSHR = zeros(nC, nA, nR)
    PMCSHR = zeros(nC, nR)
    GMCSHR = zeros(nC, nR)
    IMCSHR = zeros(nC, nR)
    for c in 1:nC, r in 1:nR
        denom = max(VMB[c,r], 1e-10)
        PMCSHR[c,r] = d.VMPB[c,r] / denom
        GMCSHR[c,r] = d.VMGB[c,r] / denom
        IMCSHR[c,r] = d.VMIB[c,r] / denom
        for a in 1:nA
            FMCSHR[c,a,r] = d.VMFB[c,a,r] / denom
        end
    end

    # ── Trade and market-clearing shares ─────────────────────────────────────
    MSHRS = zeros(nC, nR, nR)  # share of imports from s in imports of c by d
    for c in 1:nC, dd in 1:nR
        tot = max(sum(d.VMSB[c,:,dd]), 1e-10)
        MSHRS[c,:,dd] = d.VMSB[c,:,dd] ./ tot
    end

    DSSHR = zeros(nC, nR)      # share of domestic sales in total commodity supply
    XSSHR = zeros(nC, nR, nR)  # share of exports in commodity supply
    STSHR = zeros(nM, nR)      # share of transport supply in commodity supply
    for c in 1:nC, r in 1:nR
        denom = max(VCB[c,r], 1e-10)
        DSSHR[c,r] = VDB[c,r] / denom
        XSSHR[c,r,:] = d.VXSB[c,r,:] ./ denom
    end
    for (mi,m) in enumerate(s.MARG), r in 1:nR
        ci = iC(m)
        STSHR[mi,r] = VCB[ci,r] > 0 ? d.VST[mi,r] / VCB[ci,r] : 0.0
    end

    # ── Private consumption ──────────────────────────────────────────────────
    VPP  = d.VDPP .+ d.VMPP     # (nC, nR)  priv. hhld total at purch. prices
    VGP  = d.VDGP .+ d.VMGP     # (nC, nR)  gov. total at purch. prices
    VIP  = d.VDIP .+ d.VMIP     # (nC, nR)  inv. total at purch. prices

    PRIVEXP = vec(sum(VPP, dims=1))
    GOVEXP  = vec(sum(VGP, dims=1))
    REGINV  = vec(sum(VIP, dims=1))
    INCOME  = PRIVEXP .+ GOVEXP .+ d.SAVE

    PMSHR  = d.VMPP ./ max.(VPP, 1e-10)   # (nC, nR)
    GMSHR  = d.VMGP ./ max.(VGP, 1e-10)   # (nC, nR)
    IMSHR  = d.VMIP ./ max.(VIP, 1e-10)   # (nC, nR)

    CONSHR = VPP ./ max.(sum(VPP, dims=1), 1e-10)    # (nC, nR)

    # ── CDE private demand elasticities ─────────────────────────────────────
    ALPHA       = 1.0 .- d.SUBPAR                       # (nC, nR)
    sumCALPHA   = vec(sum(CONSHR .* ALPHA, dims=1))     # (nR)
    UELASPRIV   = vec(sum(CONSHR .* d.INCPAR, dims=1))  # (nR)
    XWCONSHR    = CONSHR .* d.INCPAR ./ max.(UELASPRIV', 1e-10)

    APE = zeros(nC, nC, nR)
    for i in 1:nC, k in 1:nC, r in 1:nR
        APE[i,k,r] = ALPHA[i,r] + ALPHA[k,r] - sumCALPHA[r]
    end
    for i in 1:nC, r in 1:nR
        APE[i,i,r] = 2*ALPHA[i,r] - sumCALPHA[r] - ALPHA[i,r]/max(CONSHR[i,r],1e-10)
    end
    sumCIALPHA = vec(sum(CONSHR .* d.INCPAR .* ALPHA, dims=1))
    EY = zeros(nC, nR)
    for i in 1:nC, r in 1:nR
        EY[i,r] = (1/max(UELASPRIV[r],1e-10)) *
                  (d.INCPAR[i,r]*(1-ALPHA[i,r]) + sumCIALPHA[r]) +
                  ALPHA[i,r] - sumCALPHA[r]
    end
    EP = zeros(nC, nC, nR)
    for i in 1:nC, k in 1:nC, r in 1:nR
        EP[i,k,r] = (APE[i,k,r] - EY[i,r]) * CONSHR[k,r]
    end

    # ── Government spending shares ───────────────────────────────────────────
    GOVSHR = VGP ./ max.(sum(VGP, dims=1), 1e-10)    # (nC, nR)

    # ── Investment shares ────────────────────────────────────────────────────
    INVSHR = VIP ./ max.(sum(VIP, dims=1), 1e-10)    # (nC, nR)

    # ── Regional household ───────────────────────────────────────────────────
    XSHRPRIV = PRIVEXP ./ INCOME
    XSHRGOV  = GOVEXP  ./ INCOME
    XSHRSAVE = d.SAVE  ./ INCOME
    UTILELAS = (UELASPRIV .* XSHRPRIV .+ XSHRGOV .+ XSHRSAVE) ./ max.(d.DPARSUM, 1e-10)
    DPARPRIV = UELASPRIV .* XSHRPRIV ./ max.(UTILELAS, 1e-10)
    DPARGOV  = XSHRGOV   ./ max.(UTILELAS, 1e-10)
    DPARSAVE = XSHRSAVE  ./ max.(UTILELAS, 1e-10)

    FY = zeros(nR)
    for ei in 1:nE, a in 1:nA, r in 1:nR; FY[r] += d.EVFB[ei,a,r]; end
    FY .-= d.VDEP

    # ── Investment and capital ───────────────────────────────────────────────
    NETINV    = REGINV .- d.VDEP
    GLOBINV   = sum(NETINV)
    INVKERATIO= REGINV ./ max.(d.VKB .+ NETINV, 1e-10)
    GRNETRATIO= GROSSCAP ./ max.(GROSSCAP .- d.VDEP, 1e-10)

    # ── International transport ──────────────────────────────────────────────
    VTFSD   = zeros(nC, nR, nR)
    for m in 1:nM, c in 1:nC, r in 1:nR, ss in 1:nR
        VTFSD[c,r,ss] += d.VTMFSD[m,c,r,ss]
    end
    VTMUSE    = [sum(d.VTMFSD[m,:,:,:]) for m in 1:nM]
    VTMPROV   = [sum(d.VST[m,:]) for m in 1:nM]
    VTRPROV   = [sum(d.VST[:,r]) for r in 1:nR]
    VT        = max(sum(d.VST), 1e-10)
    VTUSE     = max(sum(d.VTMFSD), 1e-10)

    VTMUSESHR = zeros(nM, nC, nR, nR)
    for m in 1:nM, c in 1:nC, r in 1:nR, ss in 1:nR
        VTMUSESHR[m,c,r,ss] = VTMUSE[m]>0 ? d.VTMFSD[m,c,r,ss]/VTMUSE[m] :
                                              VTFSD[c,r,ss]/VT
    end
    VTSUPPSHR = zeros(nM, nR)
    for m in 1:nM, r in 1:nR
        VTSUPPSHR[m,r] = VTMPROV[m]>0 ? d.VST[m,r]/VTMPROV[m] : VTRPROV[r]/VT
    end
    VTFSD_MSH = zeros(nM, nC, nR, nR)
    for m in 1:nM, c in 1:nC, r in 1:nR, ss in 1:nR
        VTFSD_MSH[m,c,r,ss] = VTFSD[c,r,ss]>0 ? d.VTMFSD[m,c,r,ss]/VTFSD[c,r,ss] :
                                                   VTMUSE[m]/VTUSE
    end

    VCIFCOST = d.VFOB .+ VTFSD
    FOBSHR   = d.VFOB ./ max.(VCIFCOST, 1e-10)
    TRNSHR   = VTFSD  ./ max.(VCIFCOST, 1e-10)

    return (;
        iC, iA, iE, iM, iR, iD, iEM, iES, iEMS, iEC,
        isMARG,
        MAKESCOM, MAKESACTSHR, MAKEBACT, MAKEBCOM, MAKEBACTSHR, MAKEBCOMSHR,
        VCB, VDB, VMB, VFB_com, VES, GROSSCAP,
        STC_endw, STC_comm, VVA, SVA, INTSHR, FMSHR,
        ENDWMSHR, REVSHR,
        FDCSHR, PDCSHR, GDCSHR, IDCSHR,
        FMCSHR, PMCSHR, GMCSHR, IMCSHR,
        MSHRS, DSSHR, XSSHR, STSHR,
        VPP, VGP, VIP, PRIVEXP, GOVEXP, REGINV, INCOME,
        PMSHR, GMSHR, IMSHR, CONSHR, GOVSHR, INVSHR,
        ALPHA, sumCALPHA, UELASPRIV, XWCONSHR, APE, EY, EP,
        XSHRPRIV, XSHRGOV, XSHRSAVE, UTILELAS, DPARPRIV, DPARGOV, DPARSAVE, FY,
        NETINV, GLOBINV, INVKERATIO, GRNETRATIO,
        VTFSD, VTMUSE, VTMPROV, VTRPROV, VT, VTUSE,
        VTMUSESHR, VTSUPPSHR, VTFSD_MSH, VCIFCOST, FOBSHR, TRNSHR,
    )
end

# =============================================================================
# PART 2 – EQUATION RESIDUALS (all 10 core modules of GTAPv7)
# =============================================================================

"""
    gtap_v7_residuals(v, d, s, C) → Dict{Symbol,Any}

Evaluate all GTAPv7 core equation residuals.
Returns Dict mapping equation name → residual array (zero at solution).
"""
function gtap_v7_residuals(v, d::GTAPDataV7, s::GTAPSetsV7, C)
    nC  = length(s.COMM);   nA  = length(s.ACTS);   nR  = length(s.REG)
    nM  = length(s.MARG);   nE  = length(s.ENDW)
    nEM = length(s.ENDWM);  nES = length(s.ENDWS);  nEF = length(s.ENDWF)
    nEC = length(s.ENDWC);  nEMS= length(s.ENDWMS)

    iC=C.iC; iA=C.iA; iE=C.iE; iM=C.iM; iR=C.iR
    iEM=C.iEM; iES=C.iES; iEMS=C.iEMS; iEC=C.iEC

    R = Dict{Symbol,Any}()

    # ════════════════════════════════════════════════════════════════════
    # MODULE 1 – FIRMS (Production and intermediate demand)
    # ════════════════════════════════════════════════════════════════════

    # E_ao: ao(a,r) = aosec(a) + aoreg(r) + aoall(a,r)
    R[:E_ao] = [v.ao[a,r] - v.aosec[a] - v.aoreg[r] - v.aoall[a,r]
                for a in 1:nA, r in 1:nR]

    # E_ava: ava(a,r) = avasec(a) + avareg(r) + avaall(a,r)
    R[:E_ava] = [v.ava[a,r] - v.avasec[a] - v.avareg[r] - v.avaall[a,r]
                 for a in 1:nA, r in 1:nR]

    # E_afa: afa(c,a,r) = afcom(c) + afsec(a) + afreg(r) + afall(c,a,r)
    R[:E_afa] = [v.afa[c,a,r] - v.afcom[c] - v.afsec[a] - v.afreg[r] - v.afall[c,a,r]
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_afe: afe(e,a,r) = afecom(e) + afesec(a) + afereg(r) + afeall(e,a,r)
    R[:E_afe] = [v.afe[e,a,r] - v.afecom[e] - v.afesec[a] - v.afereg[r] - v.afeall[e,a,r]
                 for e in 1:nE, a in 1:nA, r in 1:nR]

    # E_aint: aint(a,r) = aintsec(a) + aintreg(r) + aintall(a,r)
    R[:E_aint] = [v.aint[a,r] - v.aintsec[a] - v.aintreg[r] - v.aintall[a,r]
                  for a in 1:nA, r in 1:nR]

    # E_qint: composite intermediate demand (top-nest, ESUBT substitution)
    R[:E_qint] = [v.qint[a,r] + v.aint[a,r] - v.qo[a,r] + v.ao[a,r] +
                  d.ESUBT[a,r]*(v.pint[a,r] - v.aint[a,r] - v.po[a,r] - v.ao[a,r])
                  for a in 1:nA, r in 1:nR]

    # E_qva: value-added demand (top-nest)
    R[:E_qva] = [v.qva[a,r] + v.ava[a,r] - v.qo[a,r] + v.ao[a,r] +
                 d.ESUBT[a,r]*(v.pva[a,r] - v.ava[a,r] - v.po[a,r] - v.ao[a,r])
                 for a in 1:nA, r in 1:nR]

    # E_qfa: composite intermediate demand for c by act. a (second level, ESUBC)
    R[:E_qfa] = [v.qfa[c,a,r] + v.afa[c,a,r] - v.qint[a,r] +
                 d.ESUBC[a,r]*(v.pfa[c,a,r] - v.afa[c,a,r] - v.pint[a,r])
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_pint: price of composite intermediate inputs
    R[:E_pint] = [v.pint[a,r] - sum(C.INTSHR[c,a,r]*(v.pfa[c,a,r]-v.afa[c,a,r])
                  for c in 1:nC) for a in 1:nA, r in 1:nR]

    # E_qfe: endowment demand (CES in value added, ESUBVA)
    R[:E_qfe] = [v.qfe[e,a,r] + v.afe[e,a,r] - v.qva[a,r] +
                 d.ESUBVA[a,r]*(v.pfe[e,a,r] - v.afe[e,a,r] - v.pva[a,r])
                 for e in 1:nE, a in 1:nA, r in 1:nR]

    # E_pva: effective price of VA composite
    R[:E_pva] = [v.pva[a,r] - sum(C.SVA[e,a,r]*(v.pfe[e,a,r]-v.afe[e,a,r])
                 for e in 1:nE) for a in 1:nA, r in 1:nR]

    # E_qfd: domestic intermediate demand (dom/imp substitution, ESUBD)
    R[:E_qfd] = [v.qfd[c,a,r] - v.qfa[c,a,r] + d.ESUBD[c]*(v.pfd[c,a,r]-v.pfa[c,a,r])
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_qfm: imported intermediate demand
    R[:E_qfm] = [v.qfm[c,a,r] - v.qfa[c,a,r] + d.ESUBD[c]*(v.pfm[c,a,r]-v.pfa[c,a,r])
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_pfa: composite input price
    R[:E_pfa] = [v.pfa[c,a,r] - (1-C.FMSHR[c,a,r])*v.pfd[c,a,r] - C.FMSHR[c,a,r]*v.pfm[c,a,r]
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_pfd: domestic firm price = domestic supply price + firm input tax
    R[:E_pfd] = [v.pfd[c,a,r] - v.pds[c,r] - v.tfd[c,a,r]
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_pfm: imported firm price = composite import price + firm import tax
    R[:E_pfm] = [v.pfm[c,a,r] - v.pms[c,r] - v.tfm[c,a,r]
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_qo (zero pure profits):
    # po(a,r) + ao(a,r) = sum_e STC_endw(e,a,r)*[pfe(e,a,r)-afe(e,a,r)-ava(a,r)]
    #                    + sum_c STC_comm(c,a,r)*[pfa(c,a,r)-afa(c,a,r)-aint(a,r)]
    #                    + profitslack(a,r)
    R[:E_qo_zero] = [v.po[a,r] + v.ao[a,r] -
                     sum(C.STC_endw[e,a,r]*(v.pfe[e,a,r]-v.afe[e,a,r]-v.ava[a,r]) for e in 1:nE) -
                     sum(C.STC_comm[c,a,r]*(v.pfa[c,a,r]-v.afa[c,a,r]-v.aint[a,r]) for c in 1:nC) -
                     v.profitslack[a,r]
                     for a in 1:nA, r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 2 – COMMODITY SUPPLY (CET supply, CES sourcing via make matrix)
    # ════════════════════════════════════════════════════════════════════

    # E_qca: supply of commodity c by activity a (CET, ETRAQ < 0)
    # qca(c,a,r) = qo(a,r) - ETRAQ(a,r)*[ps(c,a,r) - po(a,r)]  if MAKES(c,a,r)>0
    R[:E_qca] = [d.MAKES[c,a,r] > 0 ?
                 v.qca[c,a,r] - v.qo[a,r] + d.ETRAQ[a,r]*(v.ps[c,a,r]-v.po[a,r]) :
                 v.qca[c,a,r]   # trivially zero when MAKES(c,a,r)=0
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_po: average unit cost (tax-exclusive) of activity a
    R[:E_po] = [v.po[a,r] - sum(C.MAKESACTSHR[c,a,r]*v.ps[c,a,r] for c in 1:nC)
                for a in 1:nA, r in 1:nR]

    # E_ps: basic (post-tax) price = supplier price + output tax
    R[:E_ps] = [v.pca[c,a,r] - v.ps[c,a,r] - v.to[c,a,r]
                for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_pb: basic price index for activity a (weighted avg of pca)
    R[:E_pb] = [v.pb[a,r] - sum(C.MAKEBACTSHR[c,a,r]*v.pca[c,a,r] for c in 1:nC)
                for a in 1:nA, r in 1:nR]

    # E_pca: CES sourcing condition (ESUBQ is 1/CES elasticity)
    # pca(c,a,r) = pds(c,r) - ESUBQ(c,r)*[qca(c,a,r) - qc(c,r)]  if MAKEB(c,a,r)>0
    R[:E_pca] = [d.MAKEB[c,a,r] > 0 ?
                 v.pca[c,a,r] - v.pds[c,r] + d.ESUBQ[c,r]*(v.qca[c,a,r]-v.qc[c,r]) :
                 v.pca[c,a,r] - v.pds[c,r]   # law of one price when ESUBQ=0
                 for c in 1:nC, a in 1:nA, r in 1:nR]

    # E_qc: market clearing for total commodity supply
    R[:E_qc] = [v.qc[c,r] - sum(C.MAKEBCOMSHR[c,a,r]*v.qca[c,a,r] for a in 1:nA)
                for c in 1:nC, r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 3 – INCOME DISTRIBUTION
    # ════════════════════════════════════════════════════════════════════

    # E_fincome: factor income at basic prices net of depreciation
    # FY(r)*fincome(r) = sum_e sum_a EVFB(e,a,r)*[peb(e,a,r)+qes(e,a,r)] - VDEP*[pinv+kb]
    R[:E_fincome] = [C.FY[r]*v.fincome[r] -
                     (sum(d.EVFB[e,a,r]*(v.peb[e,a,r]+v.qes[e,a,r]) for e in 1:nE, a in 1:nA) -
                      d.VDEP[r]*(v.pinv[r]+v.kb[r]))
                     for r in 1:nR]

    # E_y: regional income = factor income + indirect taxes (simplified)
    R[:E_y] = [C.INCOME[r]*v.y[r] - C.FY[r]*v.fincome[r] -
               100.0*C.INCOME[r]*v.del_indtaxr[r] - (C.INCOME[r]-C.FY[r])*v.y[r] -
               C.INCOME[r]*v.incomeslack[r]
               for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 4 – INCOME ALLOCATION
    # ════════════════════════════════════════════════════════════════════

    # E_qsave: saving
    R[:E_qsave] = [v.psave[r] + v.qsave[r] - v.y[r] - v.uelas[r] - v.dpsave[r]
                   for r in 1:nR]

    # E_yg: government expenditure
    R[:E_yg] = [v.yg[r] - v.y[r] - v.uelas[r] - v.dpgov[r] for r in 1:nR]

    # E_yp: private expenditure
    R[:E_yp] = [v.yp[r] - v.y[r] + (v.uepriv[r]-v.uelas[r]) - v.dppriv[r] for r in 1:nR]

    # E_uelas: elasticity of cost of utility wrt utility
    R[:E_uelas] = [v.uelas[r] - C.XSHRPRIV[r]*v.uepriv[r] + v.dpav[r] for r in 1:nR]

    # E_dpav: average distribution parameter shift
    R[:E_dpav] = [v.dpav[r] - C.XSHRPRIV[r]*v.dppriv[r] - C.XSHRGOV[r]*v.dpgov[r] -
                  C.XSHRSAVE[r]*v.dpsave[r] for r in 1:nR]

    # E_p: price index for household income disposition
    R[:E_p] = [v.p[r] - C.XSHRPRIV[r]*v.ppriv[r] - C.XSHRGOV[r]*v.pgov[r] -
               C.XSHRSAVE[r]*v.psave[r] for r in 1:nR]

    # E_u: regional household utility
    R[:E_u] = [v.u[r] - v.au[r] - (1/max(C.UTILELAS[r],1e-10))*(v.y[r]-v.pop[r]-v.p[r])
               for r in 1:nR]

    # E_dpsum: sum of distribution parameters
    R[:E_dpsum] = [d.DPARSUM[r]*v.dpsum[r] - (C.DPARPRIV[r]*v.dppriv[r] +
                   C.DPARGOV[r]*v.dpgov[r] + C.DPARSAVE[r]*v.dpsave[r])
                   for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 5 – DOMESTIC FINAL DEMAND
    # ════════════════════════════════════════════════════════════════════

    # 5-1. Private household demand (CDE demand system)
    # E_qpa: CDE private demand for composite commodity c
    R[:E_qpa] = [v.qpa[c,r] - v.pop[r] -
                 sum(C.EP[c,k,r]*v.ppa[k,r] for k in 1:nC) -
                 C.EY[c,r]*(v.yp[r]-v.pop[r])
                 for c in 1:nC, r in 1:nR]

    # E_uepriv: elasticity of expenditure wrt private utility
    R[:E_uepriv] = [v.uepriv[r] -
                    sum(C.XWCONSHR[c,r]*(v.ppa[c,r]+v.qpa[c,r]-v.yp[r]) for c in 1:nC)
                    for r in 1:nR]

    # E_ppriv: price index for aggregate private consumption
    R[:E_ppriv] = [v.ppriv[r] - sum(C.CONSHR[c,r]*v.ppa[c,r] for c in 1:nC) for r in 1:nR]

    # E_up: utility from private consumption
    R[:E_up] = [C.UELASPRIV[r]*v.up[r] - v.yp[r] + v.ppriv[r] + v.pop[r] for r in 1:nR]

    # E_qpd: domestic private demand (dom/imp substitution)
    R[:E_qpd] = [v.qpd[c,r] - v.qpa[c,r] + d.ESUBD[c]*(v.ppd[c,r]-v.ppa[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_qpm: imported private demand
    R[:E_qpm] = [v.qpm[c,r] - v.qpa[c,r] + d.ESUBD[c]*(v.ppm[c,r]-v.ppa[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_ppa: private composite price
    R[:E_ppa] = [v.ppa[c,r] - (1-C.PMSHR[c,r])*v.ppd[c,r] - C.PMSHR[c,r]*v.ppm[c,r]
                 for c in 1:nC, r in 1:nR]

    # E_ppd, E_ppm: private household prices (= supply/import prices + consumption taxes)
    R[:E_ppd] = [v.ppd[c,r] - v.pds[c,r] - v.tpd[c,r] for c in 1:nC, r in 1:nR]
    R[:E_ppm] = [v.ppm[c,r] - v.pms[c,r] - v.tpm[c,r] for c in 1:nC, r in 1:nR]

    # E_tpd, E_tpm: private consumption tax shifters
    R[:E_tpd] = [v.tpd[c,r] - v.tpdall[c,r] - v.tpreg[r] for c in 1:nC, r in 1:nR]
    R[:E_tpm] = [v.tpm[c,r] - v.tpmall[c,r] - v.tpreg[r] for c in 1:nC, r in 1:nR]

    # 5-2. Government demand (CES, ESUBG)
    # E_qga: government demand for composite commodity c
    R[:E_qga] = [v.qga[c,r] - v.yg[r] + v.pgov[r] + d.ESUBG[r]*(v.pga[c,r]-v.pgov[r])
                 for c in 1:nC, r in 1:nR]

    # E_pgov: government price index
    R[:E_pgov] = [v.pgov[r] - sum(C.GOVSHR[c,r]*v.pga[c,r] for c in 1:nC) for r in 1:nR]

    # E_ug: government utility
    R[:E_ug] = [v.ug[r] - v.yg[r] + v.pgov[r] + v.pop[r] for r in 1:nR]

    # E_qgd, E_qgm: domestic/imported government demand
    R[:E_qgd] = [v.qgd[c,r] - v.qga[c,r] + d.ESUBD[c]*(v.pgd[c,r]-v.pga[c,r])
                 for c in 1:nC, r in 1:nR]
    R[:E_qgm] = [v.qgm[c,r] - v.qga[c,r] + d.ESUBD[c]*(v.pgm[c,r]-v.pga[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_pga: government composite price
    R[:E_pga] = [v.pga[c,r] - (1-C.GMSHR[c,r])*v.pgd[c,r] - C.GMSHR[c,r]*v.pgm[c,r]
                 for c in 1:nC, r in 1:nR]

    # E_pgd, E_pgm: government prices
    R[:E_pgd] = [v.pgd[c,r] - v.pds[c,r] - v.tgd[c,r] for c in 1:nC, r in 1:nR]
    R[:E_pgm] = [v.pgm[c,r] - v.pms[c,r] - v.tgm[c,r] for c in 1:nC, r in 1:nR]

    # 5-3. Investment demand (Leontief: qia = qinv)
    # E_qia: investment demand for commodity c (Leontief)
    R[:E_qia] = [v.qia[c,r] - v.qinv[r] for c in 1:nC, r in 1:nR]

    # E_pinv: price of investment
    R[:E_pinv] = [v.pinv[r] - sum(C.INVSHR[c,r]*v.pia[c,r] for c in 1:nC) for r in 1:nR]

    # E_qid, E_qim: domestic/imported investment demand
    R[:E_qid] = [v.qid[c,r] - v.qia[c,r] + d.ESUBD[c]*(v.pid[c,r]-v.pia[c,r])
                 for c in 1:nC, r in 1:nR]
    R[:E_qim] = [v.qim[c,r] - v.qia[c,r] + d.ESUBD[c]*(v.pim[c,r]-v.pia[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_pia: investment composite price
    R[:E_pia] = [v.pia[c,r] - (1-C.IMSHR[c,r])*v.pid[c,r] - C.IMSHR[c,r]*v.pim[c,r]
                 for c in 1:nC, r in 1:nR]

    # E_pid, E_pim: investment prices
    R[:E_pid] = [v.pid[c,r] - v.pds[c,r] - v.tid[c,r] for c in 1:nC, r in 1:nR]
    R[:E_pim] = [v.pim[c,r] - v.pms[c,r] - v.tim[c,r] for c in 1:nC, r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 6 – TRADE, GOODS MARKET EQUILIBRIUM AND PRICES
    # ════════════════════════════════════════════════════════════════════

    # 6-1. Import sourcing (Armington)
    # E_qms: market clearing for imported goods in each region
    R[:E_qms] = [v.qms[c,r] -
                 (sum(C.FMCSHR[c,a,r]*v.qfm[c,a,r] for a in 1:nA) +
                  C.PMCSHR[c,r]*v.qpm[c,r] + C.GMCSHR[c,r]*v.qgm[c,r] + C.IMCSHR[c,r]*v.qim[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_qxs: regional demand for disaggregated imports (Armington)
    # (dd = destination region, avoids shadowing the data struct d)
    R[:E_qxs] = [v.qxs[c,ss,dd] + v.ams[c,ss,dd] - v.qms[c,dd] +
                 d.ESUBM[c]*(v.pmds[c,ss,dd] - v.ams[c,ss,dd] - v.pms[c,dd])
                 for c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_pms: composite import price
    R[:E_pms] = [v.pms[c,dd] - sum(C.MSHRS[c,ss,dd]*(v.pmds[c,ss,dd]-v.ams[c,ss,dd]) for ss in 1:nR)
                 for c in 1:nC, dd in 1:nR]

    # 6-2. International transport
    # E_atmfsd: transport tech change
    R[:E_atmfsd] = [v.atmfsd[m,c,ss,dd] - v.atm[m] - v.atf[c] - v.ats[ss] - v.atd[dd] - v.atall[m,c,ss,dd]
                    for m in 1:nM, c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_qtmfsd: bilateral demand for transport services
    R[:E_qtmfsd] = [v.qtmfsd[m,c,ss,dd] - v.qxs[c,ss,dd] + v.atmfsd[m,c,ss,dd]
                    for m in 1:nM, c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_qtm: global demand for margin m
    R[:E_qtm] = [v.qtm[m] - sum(C.VTMUSESHR[m,c,ss,dd]*v.qtmfsd[m,c,ss,dd]
                 for c in 1:nC, ss in 1:nR, dd in 1:nR) for m in 1:nM]

    # E_pt: price index for composite transport services
    R[:E_pt] = [v.pt[m] - sum(C.VTSUPPSHR[m,r]*v.pds[iC(s.MARG[m]),r] for r in 1:nR)
                for m in 1:nM]

    # E_qst: regional supply of margin m
    R[:E_qst] = [v.qst[m,r] - v.qtm[m] + d.ESUBS[m]*(v.pt[m]-v.pds[iC(s.MARG[m]),r])
                 for m in 1:nM, r in 1:nR]

    # E_ptrans: transport cost index
    R[:E_ptrans] = [v.ptrans[c,ss,dd] -
                    sum(C.VTFSD_MSH[m,c,ss,dd]*(v.pt[m]-v.atmfsd[m,c,ss,dd]) for m in 1:nM)
                    for c in 1:nC, ss in 1:nR, dd in 1:nR]

    # 6-3. Trade prices
    # E_pfob: FOB export prices
    R[:E_pfob] = [v.pfob[c,ss,dd] - v.pds[c,ss] + v.tx[c,ss] + v.txs[c,ss,dd]
                  for c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_pcif: FOB to CIF
    R[:E_pcif] = [v.pcif[c,ss,dd] - C.FOBSHR[c,ss,dd]*v.pfob[c,ss,dd] - C.TRNSHR[c,ss,dd]*v.ptrans[c,ss,dd]
                  for c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_pmds: domestic import price (CIF + tariff)
    R[:E_pmds] = [v.pmds[c,ss,dd] - v.pcif[c,ss,dd] - v.tm[c,dd] - v.tms[c,ss,dd]
                  for c in 1:nC, ss in 1:nR, dd in 1:nR]

    # E_pr: domestic to import price ratio
    R[:E_pr] = [v.pr[c,r] - v.pds[c,r] + v.pms[c,r] for c in 1:nC, r in 1:nR]

    # 6-4. Goods market equilibrium (domestic commodity)
    R[:E_qds] = [v.qds[c,r] -
                 (sum(C.FDCSHR[c,a,r]*v.qfd[c,a,r] for a in 1:nA) +
                  C.PDCSHR[c,r]*v.qpd[c,r] + C.GDCSHR[c,r]*v.qgd[c,r] + C.IDCSHR[c,r]*v.qid[c,r])
                 for c in 1:nC, r in 1:nR]

    # E_pds: commodity market clearing (qc = domestic + exports + transport)
    R[:E_pds] = [v.qc[c,r] - C.DSSHR[c,r]*v.qds[c,r] -
                 sum(C.XSSHR[c,r,d]*v.qxs[c,r,d] for d in 1:nR) -
                 (C.isMARG[c] ? C.STSHR[iM(s.COMM[c])::Int,r]*v.qst[iM(s.COMM[c])::Int,r] : 0.0) -
                 v.tradslack[c,r]
                 for c in 1:nC, r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 7 – FACTOR MARKET EQUILIBRIUM (3 endowment types)
    # ════════════════════════════════════════════════════════════════════

    # 7-1. Mobile endowments: perfect mobility (one economy-wide rental)
    # E_pe1: market clearing for mobile endowments
    R[:E_pe1] = [begin
        ei = iE(s.ENDWM[mi])
        v.qe[iEMS(s.ENDWM[mi]),r] - sum(C.ENDWMSHR[mi,a,r]*v.qfe[ei,a,r] for a in 1:nA) -
        v.endwslack[iEMS(s.ENDWM[mi]),r]
    end for (mi,_) in enumerate(s.ENDWM), r in 1:nR]

    # E_qes1: basic supply price of mobile endowments = economy-wide price
    R[:E_qes1] = [begin
        ei = iE(s.ENDWM[mi])
        v.pes[ei,a,r] - v.pe[iEMS(s.ENDWM[mi]),r]
    end for (mi,_) in enumerate(s.ENDWM), a in 1:nA, r in 1:nR]

    # 7-2. Sluggish endowments: CET allocation across activities
    # E_qes2: CET allocation of sluggish endowments
    R[:E_qes2] = [begin
        ei = iE(s.ENDWS[si])
        v.qes[ei,a,r] - v.qe[iEMS(s.ENDWS[si]),r] +
        d.ETRAE[ei,r]*(v.pes[ei,a,r]-v.pe[iEMS(s.ENDWS[si]),r]) +
        v.endwslack[iEMS(s.ENDWS[si]),r]
    end for (si,_) in enumerate(s.ENDWS), a in 1:nA, r in 1:nR]

    # E_pe2: composite price for sluggish endowments
    R[:E_pe2] = [begin
        ei = iE(s.ENDWS[si])
        v.pe[iEMS(s.ENDWS[si]),r] - sum(C.REVSHR[ei,a,r]*v.pes[ei,a,r] for a in 1:nA)
    end for (si,_) in enumerate(s.ENDWS), r in 1:nR]

    # 7-3. Fixed/sector-specific endowments: qes = qesf (exogenous)
    R[:E_qes3] = [begin
        ei = iE(s.ENDWF[fi])
        v.qes[ei,a,r] - v.qesf[fi,a,r]
    end for (fi,_) in enumerate(s.ENDWF), a in 1:nA, r in 1:nR]

    # 7-4. Endowment prices
    # E_pfe: firm demand price = basic endowment price + factor income tax
    R[:E_pfe] = [v.pfe[e,a,r] - v.peb[e,a,r] - v.tfe[e,a,r]
                 for e in 1:nE, a in 1:nA, r in 1:nR]

    # E_pes: basic endowment price = supply price + income tax (tinc)
    R[:E_pes_link] = [v.peb[e,a,r] - v.pes[e,a,r] - v.tinc[e,a,r]
                      for e in 1:nE, a in 1:nA, r in 1:nR]

    # E_peb (market clearing for endowments): qfe(e,a,r) = qes(e,a,r)
    R[:E_peb] = [v.qfe[e,a,r] - v.qes[e,a,r] for e in 1:nE, a in 1:nA, r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 8 – CAPITAL AND INVESTMENT
    # ════════════════════════════════════════════════════════════════════

    # E_rental: weighted capital return
    R[:E_rental] = [v.rental[r] - sum((C.VES[ec,r]/max(C.GROSSCAP[r],1e-10))*
                    v.pe[iEMS(s.ENDWC[k]),r] for (k,ec) in enumerate(iE.(s.ENDWC)))
                    for r in 1:nR]

    # E_ke: ending capital = beginning + net investment
    R[:E_ke] = [v.ke[r] - C.INVKERATIO[r]*v.qinv[r] - (1-C.INVKERATIO[r])*v.kb[r]
                for r in 1:nR]

    # E_kb: beginning capital stock services
    R[:E_kb] = [v.kb[r] - sum((C.VES[iE(s.ENDWC[k]),r]/max(C.GROSSCAP[r],1e-10))*
                v.qe[iEMS(s.ENDWC[k]),r] for k in 1:nEC)
                for r in 1:nR]

    # E_rorc: current net rate of return on capital
    R[:E_rorc] = [v.rorc[r] - C.GRNETRATIO[r]*(v.rental[r]-v.pinv[r]) for r in 1:nR]

    # E_rore: expected rate of return (depends on current return and investment)
    R[:E_rore] = [v.rore[r] - v.rorc[r] + d.RORFLEX[r]*(v.ke[r]-v.kb[r]) for r in 1:nR]

    # E_qinv: investment allocation (ROR-sensitive or fixed composition)
    R[:E_qinv] = [d.RORDELTA*v.rore[r] +
                  (1-d.RORDELTA)*((C.REGINV[r]/max(C.NETINV[r],1e-10))*v.qinv[r] -
                                  (d.VDEP[r]/max(C.NETINV[r],1e-10))*v.kb[r]) -
                  d.RORDELTA*v.rorg[1] - (1-d.RORDELTA)*v.globalcgds[1] - v.cgdslack[r]
                  for r in 1:nR]

    # E_expand: change in investment relative to capital stock
    R[:E_expand] = [v.expand[k,r] - v.qinv[r] + v.qe[iEMS(s.ENDWC[k]),r]
                    for k in 1:nEC, r in 1:nR]

    # E_globalcgds: global net investment or global rate of return
    R[:E_globalcgds] = [d.RORDELTA*v.globalcgds[1] + (1-d.RORDELTA)*v.rorg[1] -
                        d.RORDELTA*sum((C.REGINV[r]/max(C.GLOBINV,1e-10))*v.qinv[r] -
                                       (d.VDEP[r]/max(C.GLOBINV,1e-10))*v.kb[r] for r in 1:nR) -
                        (1-d.RORDELTA)*sum((C.NETINV[r]/max(C.GLOBINV,1e-10))*v.rore[r] for r in 1:nR)]

    # E_psave: saving price
    R[:E_psave] = [v.psave[r] - v.pinv[r] -
                   sum(((C.NETINV[ss]-d.SAVE[ss])/max(C.GLOBINV,1e-10))*v.pinv[ss] for ss in 1:nR) -
                   v.psaveslack[r] for r in 1:nR]

    # E_pcgdswld: world capital goods price
    R[:E_pcgdswld] = [v.pcgdswld[1] - sum((C.NETINV[r]/max(C.GLOBINV,1e-10))*v.pinv[r] for r in 1:nR)]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 9 – TAX REVENUE (simplified: del_indtaxr only)
    # ════════════════════════════════════════════════════════════════════

    # del_indtaxr is treated as a slack in the income equation; here we just
    # require it to satisfy: INCOME*del_indtaxr ≈ 0 in standard experiments
    # (Full tax revenue accounting can be added later)
    R[:E_del_indtaxr] = [v.del_indtaxr[r] for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 10 – NUMERAIRE AND WALRAS
    # ════════════════════════════════════════════════════════════════════

    # pfactor(r): regional factor price index (for numeraire chain)
    VENDWREG = [sum(C.VES[iE(e),r] for e in s.ENDW) for r in 1:nR]
    VENDWWLD = sum(VENDWREG)

    R[:E_pfactor] = [VENDWREG[r]*v.pfactor[r] -
                     sum(C.VES[iE(e),r]*v.pe[iEMS(e),r] for e in s.ENDWMS)
                     # Fixed endowments do not contribute to numeraire (prices determined otherwise)
                     for r in 1:nR]

    # pfactwld: world factor price index (numeraire)
    R[:E_pfactwld] = [VENDWWLD * v.pfactwld -
                      sum(VENDWREG[r]*v.pfactor[r] for r in 1:nR)]

    # Walras: global saving = global investment
    R[:E_walras_sup] = [v.walras_sup[1] - v.pcgdswld[1] - v.globalcgds[1]]
    R[:E_walras_dem] = [C.GLOBINV*v.walras_dem[1] -
                        sum(d.SAVE[r]*(v.psave[r]+v.qsave[r]) for r in 1:nR)]
    R[:E_walras]     = [v.walras_sup[1] - v.walras_dem[1] - v.walraslack]

    return R
end
