# =============================================================================
# gtap_v62.jl
# Complete Julia translation of the standard GTAP model, Version 6.2 (2003)
#
# Reference: Hertel & Tsigas, "Structure of the Standard GTAP Model", Ch. 2 in
#            Hertel (ed.) "Global Trade Analysis", CUP, 1997.
#            McDougall, "A New Regional Household Demand System for GTAP", TP#20.
#
# This file contains:
#   Part 1 – Sets, Coefficient structs, Derived coefficients
#   Part 2 – Equation residuals (all 8 modules + Appendices A–C)
#
# TABLO convention: UPPER_CASE = coefficient, lower_case = variable (%change)
# =============================================================================

using LinearAlgebra, SparseArrays

# =============================================================================
# PART 1 – SETS AND COEFFICIENTS
# =============================================================================

struct GTAPSets
    REG        ::Vector{String}
    TRAD_COMM  ::Vector{String}   # traded commodities
    MARG_COMM  ::Vector{String}   # margin commodities  ⊆ TRAD_COMM
    NMRG_COMM  ::Vector{String}   # non-margin traded   = TRAD_COMM – MARG_COMM
    CGDS_COMM  ::Vector{String}   # capital goods (single element)
    ENDW_COMM  ::Vector{String}   # endowment commodities
    ENDWS_COMM ::Vector{String}   # sluggish endowments ⊆ ENDW_COMM
    ENDWM_COMM ::Vector{String}   # mobile endowments   ⊆ ENDW_COMM
    ENDWC_COMM ::Vector{String}   # capital endowment   ⊆ ENDW_COMM (|·|=1)
    PROD_COMM  ::Vector{String}   # TRAD_COMM ∪ CGDS_COMM
    DEMD_COMM  ::Vector{String}   # ENDW_COMM ∪ TRAD_COMM
    NSAV_COMM  ::Vector{String}   # DEMD_COMM ∪ CGDS_COMM
end

"""Build sets from raw vectors + SLUG integer array (>0 → sluggish)."""
function build_sets(REG, TRAD_COMM, MARG_COMM, CGDS_COMM, ENDW_COMM,
                    SLUG::Vector{Int}, ENDWC_COMM)
    NMRG_COMM  = [x for x in TRAD_COMM if x ∉ MARG_COMM]
    ENDWS_COMM = ENDW_COMM[SLUG .> 0]
    ENDWM_COMM = [x for x in ENDW_COMM if x ∉ ENDWS_COMM]
    PROD_COMM  = vcat(TRAD_COMM, CGDS_COMM)
    DEMD_COMM  = vcat(ENDW_COMM, TRAD_COMM)
    NSAV_COMM  = vcat(DEMD_COMM, CGDS_COMM)
    return GTAPSets(REG, TRAD_COMM, MARG_COMM, NMRG_COMM, CGDS_COMM,
                    ENDW_COMM, ENDWS_COMM, ENDWM_COMM, ENDWC_COMM,
                    PROD_COMM, DEMD_COMM, NSAV_COMM)
end

"""Raw base-data and parameter arrays (read from HAR files)."""
struct GTAPData
    # ── GTAPDATA ──────────────────────────────────────────────────────────
    SAVE   ::Vector{Float64}       # (r)
    VDGA   ::Matrix{Float64}       # (nT,nR)
    VDGM   ::Matrix{Float64}
    VIGA   ::Matrix{Float64}
    VIGM   ::Matrix{Float64}
    VDPA   ::Matrix{Float64}
    VDPM   ::Matrix{Float64}
    VIPA   ::Matrix{Float64}
    VIPM   ::Matrix{Float64}
    EVOA   ::Matrix{Float64}       # (nE,nR)
    EVFA   ::Array{Float64,3}      # (nE,nP,nR)
    VDFA   ::Array{Float64,3}      # (nT,nP,nR)
    VDFM   ::Array{Float64,3}
    VIFA   ::Array{Float64,3}
    VIFM   ::Array{Float64,3}
    VFM    ::Array{Float64,3}      # (nE,nP,nR)
    VXMD   ::Array{Float64,3}      # (nT,nR,nR)
    VXWD   ::Array{Float64,3}
    VIWS   ::Array{Float64,3}
    VIMS   ::Array{Float64,3}
    VST    ::Matrix{Float64}       # (nM,nR)
    VTMFSD ::Array{Float64,4}      # (nM,nT,nR,nR)
    VKB    ::Vector{Float64}       # (r)
    VDEP   ::Vector{Float64}
    DPARSUM::Vector{Float64}
    # ── GTAPPARM ──────────────────────────────────────────────────────────
    ESUBD  ::Vector{Float64}       # (nT)
    ESUBT  ::Vector{Float64}       # (nP)
    ESUBVA ::Vector{Float64}       # (nP)
    ESUBM  ::Vector{Float64}       # (nT)
    ETRAE  ::Vector{Float64}       # (nE)
    INCPAR ::Matrix{Float64}       # (nT,nR)
    SUBPAR ::Matrix{Float64}       # (nT,nR)  CDE substitution parameter
    RORFLEX::Vector{Float64}       # (r)
    RORDELTA::Int                  # 0 or 1
    SLUG   ::Vector{Int}           # (nE)
end

# =============================================================================
# DERIVED COEFFICIENTS  (computed once from GTAPData before solving)
# =============================================================================

"""
Compute all formula-derived coefficients.
Returns a NamedTuple used by both the residual function and the JuMP model.
"""
function compute_derived(d::GTAPData, s::GTAPSets)
    nT  = length(s.TRAD_COMM);  nE  = length(s.ENDW_COMM)
    nR  = length(s.REG);        nP  = length(s.PROD_COMM)
    nM  = length(s.MARG_COMM);  nES = length(s.ENDWS_COMM)
    nEM = length(s.ENDWM_COMM); nND = length(s.DEMD_COMM)
    nNS = length(s.NSAV_COMM)

    # index helpers
    iT(x) = findfirst(==(x), s.TRAD_COMM)::Int
    iE(x) = findfirst(==(x), s.ENDW_COMM)::Int
    iP(x) = findfirst(==(x), s.PROD_COMM)::Int
    iD(x) = findfirst(==(x), s.DEMD_COMM)::Int
    iN(x) = findfirst(==(x), s.NSAV_COMM)::Int
    iM(x) = findfirst(==(x), s.MARG_COMM)::Int
    iR(x) = findfirst(==(x), s.REG)::Int

    isMARG  = Bool[x ∈ s.MARG_COMM  for x in s.TRAD_COMM]
    isNMRG  = .!isMARG
    isENDWS = Bool[x ∈ s.ENDWS_COMM for x in s.ENDW_COMM]
    isENDWM = Bool[x ∈ s.ENDWM_COMM for x in s.ENDW_COMM]
    isENDWC = Bool[x ∈ s.ENDWC_COMM for x in s.ENDW_COMM]

    # ── VFA  (nD,nP,nR) ───────────────────────────────────────────────────
    VFA = zeros(nND, nP, nR)
    for (ei,e) in enumerate(s.ENDW_COMM); VFA[iD(e),:,:] = d.EVFA[ei,:,:]; end
    for (ti,t) in enumerate(s.TRAD_COMM); VFA[iD(t),:,:] = d.VDFA[ti,:,:] .+ d.VIFA[ti,:,:]; end

    # ── VOA  (nNS,nR) ─────────────────────────────────────────────────────
    VOA = zeros(nNS, nR)
    for (ei,e) in enumerate(s.ENDW_COMM); VOA[iN(e),:] = d.EVOA[ei,:]; end
    for (pi,p) in enumerate(s.PROD_COMM); VOA[iN(p),:] = vec(sum(VFA[:,pi,:], dims=1)); end

    # ── VDM  (nT,nR) ──────────────────────────────────────────────────────
    VDM = zeros(nT, nR)
    for i in 1:nT, r in 1:nR
        VDM[i,r] = d.VDPM[i,r] + d.VDGM[i,r] + sum(d.VDFM[i,:,r])
    end

    # ── VOM  (nNS,nR) ─────────────────────────────────────────────────────
    VOM = zeros(nNS, nR)
    for (ei,e) in enumerate(s.ENDW_COMM)
        VOM[iN(e),:] = vec(sum(d.VFM[ei,:,:], dims=1))
    end
    for (mi,m) in enumerate(s.MARG_COMM), r in 1:nR
        VOM[iN(m),r] = VDM[iT(m),r] + sum(d.VXMD[iT(m),r,:]) + d.VST[mi,r]
    end
    for i in 1:nT
        isNMRG[i] || continue
        t = s.TRAD_COMM[i]
        for r in 1:nR; VOM[iN(t),r] = VDM[i,r] + sum(d.VXMD[i,r,:]); end
    end
    for c in s.CGDS_COMM, r in 1:nR; VOM[iN(c),r] = VOA[iN(c),r]; end

    # ── Expenditure aggregates ─────────────────────────────────────────────
    VGA     = d.VDGA .+ d.VIGA                            # (nT,nR)
    VPA     = d.VDPA .+ d.VIPA
    GOVEXP  = vec(sum(VGA,  dims=1))
    PRIVEXP = vec(sum(VPA,  dims=1))
    INCOME  = PRIVEXP .+ GOVEXP .+ d.SAVE

    # ── Tax receipt components ─────────────────────────────────────────────
    DGTAX = d.VDGA .- d.VDGM;   IGTAX = d.VIGA .- d.VIGM
    TGC   = vec(sum(DGTAX .+ IGTAX, dims=1))
    DPTAX = d.VDPA .- d.VDPM;   IPTAX = d.VIPA .- d.VIPM
    TPC   = vec(sum(DPTAX .+ IPTAX, dims=1))
    DFTAX = d.VDFA .- d.VDFM;   IFTAX = d.VIFA .- d.VIFM
    TIU   = [sum(DFTAX[:,:,r] .+ IFTAX[:,:,r]) for r in 1:nR]
    ETAX  = d.EVFA .- d.VFM
    TFU   = [sum(ETAX[:,:,r]) for r in 1:nR]
    PTAX  = zeros(nNS, nR)
    for n in 1:nNS; PTAX[n,:] = VOM[n,:] .- VOA[n,:]; end
    TOUT  = zeros(nR)
    for p in s.PROD_COMM; TOUT .+= PTAX[iN(p),:]; end
    XTAXD = d.VXWD .- d.VXMD
    TEX   = [sum(XTAXD[:,r,:]) for r in 1:nR]
    MTAX  = d.VIMS .- d.VIWS
    TIM   = [sum(MTAX[:,:,r]) for r in 1:nR]
    TINC  = zeros(nR)
    for e in s.ENDW_COMM; TINC .+= PTAX[iN(e),:]; end
    INDTAX = TPC .+ TGC .+ TIU .+ TFU .+ TOUT .+ TEX .+ TIM

    # ── Private-consumption CDE parameters ───────────────────────────────
    ALPHA     = 1.0 .- d.SUBPAR                            # (nT,nR)
    CONSHR    = VPA ./ max.(sum(VPA,dims=1), 1e-10)
    sumCALPHA = vec(sum(CONSHR .* ALPHA, dims=1))          # (nR)
    UELASPRIV = vec(sum(CONSHR .* d.INCPAR, dims=1))
    XWCONSHR  = CONSHR .* d.INCPAR ./ max.(UELASPRIV', 1e-10)

    # Allen partials APE(i,k,r) and price elasticities EP(i,k,r)
    APE = zeros(nT, nT, nR)
    for i in 1:nT, k in 1:nT, r in 1:nR
        APE[i,k,r] = ALPHA[i,r] + ALPHA[k,r] - sumCALPHA[r]
    end
    for i in 1:nT, r in 1:nR          # diagonal override
        APE[i,i,r] = 2*ALPHA[i,r] - sumCALPHA[r] - ALPHA[i,r]/max(CONSHR[i,r],1e-10)
    end
    sumCIALPHA = vec(sum(CONSHR .* d.INCPAR .* ALPHA, dims=1))   # (nR)
    EY = zeros(nT, nR)
    for i in 1:nT, r in 1:nR
        EY[i,r] = (1/max(UELASPRIV[r],1e-10)) *
                  (d.INCPAR[i,r]*(1-ALPHA[i,r]) + sumCIALPHA[r]) +
                  ALPHA[i,r] - sumCALPHA[r]
    end
    EP = zeros(nT, nT, nR)
    for i in 1:nT, k in 1:nT, r in 1:nR
        EP[i,k,r] = (APE[i,k,r] - EY[i,r]) * CONSHR[k,r]
    end
    PMSHR = d.VIPA ./ max.(VPA, 1e-10)                    # (nT,nR)

    # ── Firm shares ───────────────────────────────────────────────────────
    VFA_trad = VFA[nE+1:nE+nT,:,:]                        # (nT,nP,nR)
    FMSHR    = d.VIFA ./ max.(VFA_trad, 1e-10)
    VVA      = zeros(nP, nR)
    for j in 1:nP, r in 1:nR
        for e in s.ENDW_COMM; VVA[j,r] += VFA[iD(e),j,r]; end
    end
    SVADEFAULT = zeros(nE)
    denom_sva  = max(sum(VVA), 1e-10)
    for i in 1:nE; SVADEFAULT[i] = sum(VFA[iD(s.ENDW_COMM[i]),:,:]) / denom_sva; end
    SVA = zeros(nE, nP, nR)
    for i in 1:nE, j in 1:nP, r in 1:nR
        SVA[i,j,r] = VVA[j,r]>0 ? VFA[iD(s.ENDW_COMM[i]),j,r]/VVA[j,r] : SVADEFAULT[i]
    end
    STC = zeros(nND, nP, nR)
    for j in 1:nP, r in 1:nR
        tot = sum(VFA[:,j,r]); STC[:,j,r] = VFA[:,j,r] ./ max(tot,1e-10)
    end
    REVSHR = zeros(nE, nP, nR)
    for i in 1:nE, r in 1:nR
        tot = sum(d.VFM[i,:,r]); REVSHR[i,:,r] = d.VFM[i,:,r] ./ max(tot,1e-10)
    end

    # ── Investment & capital ──────────────────────────────────────────────
    REGINV    = zeros(nR)
    for c in s.CGDS_COMM; REGINV .+= VOA[iN(c),:]; end
    NETINV    = REGINV .- d.VDEP
    GLOBINV   = sum(NETINV)
    INVKERATIO= REGINV ./ max.(d.VKB .+ NETINV, 1e-10)
    VOAcap    = zeros(nR)
    for c in s.ENDWC_COMM; VOAcap .+= VOA[iN(c),:]; end
    GRNETRATIO= VOAcap ./ max.(VOAcap .- d.VDEP, 1e-10)

    # ── Regional household ────────────────────────────────────────────────
    XSHRPRIV  = PRIVEXP ./ INCOME
    XSHRGOV   = GOVEXP  ./ INCOME
    XSHRSAVE  = d.SAVE  ./ INCOME
    UTILELAS  = (UELASPRIV .* XSHRPRIV .+ XSHRGOV .+ XSHRSAVE) ./ max.(d.DPARSUM, 1e-10)
    DPARPRIV  = UELASPRIV .* XSHRPRIV ./ max.(UTILELAS, 1e-10)
    DPARGOV   = XSHRGOV   ./ max.(UTILELAS, 1e-10)
    DPARSAVE  = XSHRSAVE  ./ max.(UTILELAS, 1e-10)
    FY = zeros(nR)
    for e in s.ENDW_COMM; FY .+= VOM[iN(e),:]; end
    FY .-= d.VDEP

    # ── International transport ───────────────────────────────────────────
    VTFSD  = zeros(nT, nR, nR)
    for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR
        VTFSD[i,r,ss] += d.VTMFSD[m,i,r,ss]
    end
    VTMUSE  = [sum(d.VTMFSD[m,:,:,:]) for m in 1:nM]
    VTMPROV = [sum(d.VST[m,:]) for m in 1:nM]
    VTRPROV = [sum(d.VST[:,r]) for r in 1:nR]
    VT      = max(sum(d.VST), 1e-10)
    VTUSE   = max(sum(d.VTMFSD), 1e-10)
    VTMUSESHR = zeros(nM, nT, nR, nR)
    for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR
        VTMUSESHR[m,i,r,ss] = VTMUSE[m]>0 ? d.VTMFSD[m,i,r,ss]/VTMUSE[m] :
                                              VTFSD[i,r,ss]/VT
    end
    VTSUPPSHR = zeros(nM, nR)
    for m in 1:nM, r in 1:nR
        VTSUPPSHR[m,r] = VTMPROV[m]>0 ? d.VST[m,r]/VTMPROV[m] : VTRPROV[r]/VT
    end
    VTFSD_MSH = zeros(nM, nT, nR, nR)
    for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR
        VTFSD_MSH[m,i,r,ss] = VTFSD[i,r,ss]>0 ? d.VTMFSD[m,i,r,ss]/VTFSD[i,r,ss] :
                                                   VTMUSE[m]/VTUSE
    end
    VIWSCOST = d.VXWD .+ VTFSD
    FOBSHR   = d.VXWD ./ max.(VIWSCOST, 1e-10)
    TRNSHR   = VTFSD  ./ max.(VIWSCOST, 1e-10)

    # ── Trade & market-clearing shares ────────────────────────────────────
    MSHRS  = zeros(nT, nR, nR)
    for i in 1:nT, ss in 1:nR
        tot = max(sum(d.VIMS[i,:,ss]), 1e-10)
        MSHRS[i,:,ss] = d.VIMS[i,:,ss] ./ tot
    end
    GMSHR  = d.VIGA ./ max.(VGA, 1e-10)
    VIM    = zeros(nT, nR)
    for i in 1:nT, r in 1:nR
        VIM[i,r] = sum(d.VIFM[i,:,r]) + d.VIPM[i,r] + d.VIGM[i,r]
    end
    SHRIFM = zeros(nT, nP, nR)
    for i in 1:nT, j in 1:nP, r in 1:nR
        SHRIFM[i,j,r] = d.VIFM[i,j,r] / max(VIM[i,r],1e-10)
    end
    SHRIPM = d.VIPM ./ max.(VIM, 1e-10)
    SHRIGM = d.VIGM ./ max.(VIM, 1e-10)
    SHRDFM = zeros(nT, nP, nR)
    for i in 1:nT, j in 1:nP, r in 1:nR
        SHRDFM[i,j,r] = d.VDFM[i,j,r] / max(VDM[i,r],1e-10)
    end
    SHRDPM = d.VDPM ./ max.(VDM, 1e-10)
    SHRDGM = d.VDGM ./ max.(VDM, 1e-10)
    SHRDM  = zeros(nT, nR)
    for i in 1:nT, r in 1:nR
        SHRDM[i,r] = VDM[i,r] / max(VOM[iN(s.TRAD_COMM[i]),r],1e-10)
    end
    SHRST  = zeros(nM, nR)
    for m in 1:nM, r in 1:nR
        SHRST[m,r] = d.VST[m,r] / max(VOM[iN(s.MARG_COMM[m]),r],1e-10)
    end
    SHRXMD = zeros(nT, nR, nR)
    for i in 1:nT, r in 1:nR
        denom = max(VOM[iN(s.TRAD_COMM[i]),r],1e-10)
        SHRXMD[i,r,:] = d.VXMD[i,r,:] ./ denom
    end
    SHREM  = zeros(nEM, nP, nR)
    for (mi2,m2) in enumerate(s.ENDWM_COMM)
        ei = iE(m2); ni = iN(m2)
        for j in 1:nP, r in 1:nR
            SHREM[mi2,j,r] = d.VFM[ei,j,r] / max(VOM[ni,r],1e-10)
        end
    end

    # ── Summary-index coefficients ────────────────────────────────────────
    VXW = zeros(nT, nR)
    for (mi,m) in enumerate(s.MARG_COMM), r in 1:nR
        VXW[iT(m),r] = sum(d.VXWD[iT(m),r,:]) + d.VST[mi,r]
    end
    for i in 1:nT; isNMRG[i] || continue
        for r in 1:nR; VXW[i,r] = sum(d.VXWD[i,r,:]); end
    end
    VIW        = zeros(nT, nR)
    for i in 1:nT, r in 1:nR; VIW[i,r] = sum(d.VIWS[i,:,r]); end
    VXWREGION  = vec(sum(VXW, dims=1))
    VIWREGION  = vec(sum(VIW, dims=1))
    VENDWREG   = [sum(VOM[iN(e),r] for e in s.ENDW_COMM) for r in 1:nR]
    VENDWWLD   = sum(VENDWREG)
    GDP        = zeros(nR)
    for r in 1:nR
        GDP[r] = sum(VPA[:,r]) + sum(VGA[:,r]) +
                 sum(VOA[iN(c),r] for c in s.CGDS_COMM) +
                 sum(d.VXWD[:,r,:]) + sum(d.VST[:,r]) -
                 sum(d.VIWS[:,r,:])
    end
    VXWCOMMOD  = vec(sum(VXW, dims=2))
    VIWCOMMOD  = vec(sum(VIW, dims=2))
    VXWLD      = sum(VXWREGION)
    PW_PM      = zeros(nT, nR)
    for i in 1:nT, r in 1:nR
        denom = max(sum(d.VXMD[i,r,:]),1e-10)
        PW_PM[i,r] = sum(d.VXWD[i,r,:]) / denom
    end
    VOW = zeros(nT, nR)
    for (mi,m) in enumerate(s.MARG_COMM), r in 1:nR
        ti = iT(m)
        VOW[ti,r] = VDM[ti,r]*PW_PM[ti,r] + sum(d.VXWD[ti,r,:]) + d.VST[mi,r]
    end
    for i in 1:nT; isNMRG[i] || continue
        for r in 1:nR; VOW[i,r] = VDM[i,r]*PW_PM[i,r] + sum(d.VXWD[i,r,:]); end
    end
    VWOW  = vec(sum(VOW, dims=2))
    VWOU  = zeros(nT)
    for i in 1:nT
        di = iD(s.TRAD_COMM[i])
        VWOU[i] = sum(VPA[i,:]) + sum(VGA[i,:]) + sum(VFA[di,:,:])
    end
    TBAL  = VXWREGION .- VIWREGION

    # ── EV / welfare-decomp coefficients  (initial = base) ───────────────
    EVSCALFACT = ones(nR)           # updated each step (= UTILELASEV/UTILELAS * INCOMEEV/INCOME)
    VTMD = zeros(nM, nR)
    for m in 1:nM, r in 1:nR
        for i in 1:nT, ss in 1:nR; VTMD[m,r] += d.VTMFSD[m,i,ss,r]; end
    end

    return (;
        iT, iE, iP, iD, iN, iM, iR,
        isMARG, isNMRG, isENDWS, isENDWM, isENDWC,
        VFA, VOA, VOM, VDM, VGA, VPA,
        GOVEXP, PRIVEXP, INCOME,
        DGTAX, IGTAX, TGC, DPTAX, IPTAX, TPC,
        DFTAX, IFTAX, TIU, ETAX, TFU, PTAX, TOUT,
        XTAXD, TEX, MTAX, TIM, TINC, INDTAX,
        ALPHA, CONSHR, UELASPRIV, XWCONSHR,
        APE, EY, EP, PMSHR,
        GMSHR, FMSHR, VVA, SVA, STC, REVSHR,
        REGINV, NETINV, GLOBINV, INVKERATIO, VOAcap, GRNETRATIO,
        XSHRPRIV, XSHRGOV, XSHRSAVE,
        UTILELAS, DPARPRIV, DPARGOV, DPARSAVE, FY,
        VTFSD, VTMUSE, VTMPROV, VTRPROV, VT, VTUSE,
        VTMUSESHR, VTSUPPSHR, VTFSD_MSH,
        VIWSCOST, FOBSHR, TRNSHR,
        MSHRS, VIM,
        SHRIFM, SHRIPM, SHRIGM, SHRDFM, SHRDPM, SHRDGM,
        SHRDM, SHRST, SHRXMD, SHREM,
        VXW, VIW, VXWREGION, VIWREGION, VENDWREG, VENDWWLD,
        GDP, VXWCOMMOD, VIWCOMMOD, VXWLD,
        PW_PM, VOW, VWOW, VWOU, TBAL,
        EVSCALFACT, VTMD
    )
end


# =============================================================================
# PART 2 – EQUATION RESIDUALS
# All equations from gtap.tab Modules 1–8 and Appendices A–C
# =============================================================================

"""
    gtap_residuals(v, d, s, C) → Dict{Symbol,Array}

Evaluate all GTAP v6.2 equation residuals.
`v` is a NamedTuple of variable arrays; `d` raw data; `s` sets; `C` derived coefficients.
Returns Dict mapping equation name → residual array (zero at solution).
"""
function gtap_residuals(v, d::GTAPData, s::GTAPSets, C; skip_appendix::Bool=false)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM); nR=length(s.REG)
    nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEM=length(s.ENDWM_COMM)
    nN=length(s.NMRG_COMM)

    # shorthand index helpers from C
    iT=C.iT; iE=C.iE; iP=C.iP; iD=C.iD; iN=C.iN; iM=C.iM

    # pm indexed on NSAV_COMM; helper for TRAD_COMM element
    pm_t(i,r) = v.pm[iN(s.TRAD_COMM[i]), r]
    pm_e(i,r) = v.pm[iN(s.ENDW_COMM[i]), r]
    ps_n(i,r) = v.ps[iN(s.PROD_COMM[i]), r]
    qo_n(i,r) = v.qo[iN(s.PROD_COMM[i]), r]

    R = Dict{Symbol,Any}()

    # ════════════════════════════════════════════════════════════════════
    # MODULE 1 – GOVERNMENT CONSUMPTION
    # ════════════════════════════════════════════════════════════════════

    # GPRICEINDEX  (HT 40)
    R[:GPRICEINDEX] = [v.pgov[r] - sum(C.VGA[i,r]/max(C.GOVEXP[r],1e-10)*v.pg[i,r]
                        for i in 1:nT) for r in 1:nR]

    # GOVDMNDS  (HT 41)
    R[:GOVDMNDS] = [v.qg[i,r] - v.pop[r] - v.ug[r] + v.pg[i,r] - v.pgov[r]
                    for i in 1:nT, r in 1:nR]

    # GOVU
    R[:GOVU] = [v.yg[r] - v.pop[r] - v.pgov[r] - v.ug[r] for r in 1:nR]

    # GHHDPRICE  (HT 19)
    R[:GHHDPRICE] = [v.pgd[i,r] - v.tgd[i,r] - pm_t(i,r) for i in 1:nT, r in 1:nR]

    # GHHIPRICES  (HT 22)
    R[:GHHIPRICES] = [v.pgm[i,r] - v.tgm[i,r] - v.pim[i,r] for i in 1:nT, r in 1:nR]

    # GCOMPRICE  (HT 42)
    R[:GCOMPRICE] = [v.pg[i,r] - C.GMSHR[i,r]*v.pgm[i,r] - (1-C.GMSHR[i,r])*v.pgd[i,r]
                     for i in 1:nT, r in 1:nR]

    # GHHLDAGRIMP  (HT 43)
    R[:GHHLDAGRIMP] = [v.qgm[i,r] - v.qg[i,r] - d.ESUBD[i]*(v.pg[i,r]-v.pgm[i,r])
                       for i in 1:nT, r in 1:nR]

    # GHHLDDOM  (HT 44)
    R[:GHHLDDOM] = [v.qgd[i,r] - v.qg[i,r] - d.ESUBD[i]*(v.pg[i,r]-v.pgd[i,r])
                    for i in 1:nT, r in 1:nR]

    # TGCRATIO  (change equation)
    R[:TGCRATIO] = [100*C.INCOME[r]*v.del_taxrgc[r] + C.TGC[r]*v.y[r] -
                    sum(d.VDGA[i,r]*v.tgd[i,r] + C.DGTAX[i,r]*(pm_t(i,r)+v.qgd[i,r]) +
                        d.VIGA[i,r]*v.tgm[i,r] + C.IGTAX[i,r]*(v.pim[i,r]+v.qgm[i,r])
                        for i in 1:nT) for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 2 – PRIVATE CONSUMPTION
    # ════════════════════════════════════════════════════════════════════

    # PHHLDINDEX
    R[:PHHLDINDEX] = [v.ppriv[r] - sum(C.CONSHR[i,r]*v.pp[i,r] for i in 1:nT) for r in 1:nR]

    # PRIVATEU  (HT 45)
    R[:PRIVATEU] = [v.yp[r] - v.pop[r] - v.ppriv[r] - C.UELASPRIV[r]*v.up[r] for r in 1:nR]

    # UTILELASPRIV
    R[:UTILELASPRIV] = [v.uepriv[r] -
                        sum(C.XWCONSHR[i,r]*(v.pp[i,r]+v.qp[i,r]-v.yp[r]) for i in 1:nT)
                        for r in 1:nR]

    # PRIVDMNDS  (HT 46)
    R[:PRIVDMNDS] = [v.qp[i,r] - v.pop[r] -
                     sum(C.EP[i,k,r]*v.pp[k,r] for k in 1:nT) -
                     C.EY[i,r]*(v.yp[r]-v.pop[r])
                     for i in 1:nT, r in 1:nR]

    # TPDSHIFT
    R[:TPDSHIFT] = [v.atpd[i,r] - v.tpd[i,r] - v.tp[r] for i in 1:nT, r in 1:nR]

    # PHHDPRICE  (HT 18)
    R[:PHHDPRICE] = [v.ppd[i,r] - v.atpd[i,r] - pm_t(i,r) for i in 1:nT, r in 1:nR]

    # TPMSHIFT
    R[:TPMSHIFT] = [v.atpm[i,r] - v.tpm[i,r] - v.tp[r] for i in 1:nT, r in 1:nR]

    # PHHIPRICES  (HT 21)
    R[:PHHIPRICES] = [v.ppm[i,r] - v.atpm[i,r] - v.pim[i,r] for i in 1:nT, r in 1:nR]

    # PCOMPRICE  (HT 47)
    R[:PCOMPRICE] = [v.pp[i,r] - C.PMSHR[i,r]*v.ppm[i,r] - (1-C.PMSHR[i,r])*v.ppd[i,r]
                     for i in 1:nT, r in 1:nR]

    # PHHLDDOM  (HT 48)
    R[:PHHLDDOM] = [v.qpd[i,r] - v.qp[i,r] - d.ESUBD[i]*(v.pp[i,r]-v.ppd[i,r])
                    for i in 1:nT, r in 1:nR]

    # PHHLDAGRIMP  (HT 49)
    R[:PHHLDAGRIMP] = [v.qpm[i,r] - v.qp[i,r] - d.ESUBD[i]*(v.pp[i,r]-v.ppm[i,r])
                       for i in 1:nT, r in 1:nR]

    # TPCRATIO
    R[:TPCRATIO] = [100*C.INCOME[r]*v.del_taxrpc[r] + C.TPC[r]*v.y[r] -
                    sum(d.VDPA[i,r]*v.atpd[i,r] + C.DPTAX[i,r]*(pm_t(i,r)+v.qpd[i,r]) +
                        d.VIPA[i,r]*v.atpm[i,r] + C.IPTAX[i,r]*(v.pim[i,r]+v.qpm[i,r])
                        for i in 1:nT) for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 3 – FIRMS
    # ════════════════════════════════════════════════════════════════════

    # AOWORLD
    R[:AOWORLD] = [v.ao[j,r] - v.aosec[j] - v.aoreg[r] - v.aoall[j,r]
                   for j in 1:nP, r in 1:nR]

    # AVAWORLD
    R[:AVAWORLD] = [v.ava[j,r] - v.avasec[j] - v.avareg[r] - v.avaall[j,r]
                    for j in 1:nP, r in 1:nR]

    # VADEMAND  (HT 35)
    R[:VADEMAND] = [v.qva[j,r] + v.ava[j,r] - qo_n(j,r) + v.ao[j,r] +
                    d.ESUBT[j]*(v.pva[j,r] - v.ava[j,r] - ps_n(j,r) - v.ao[j,r])
                    for j in 1:nP, r in 1:nR]

    # AFWORLD
    R[:AFWORLD] = [v.af[i,j,r] - v.afcom[i] - v.afsec[j] - v.afreg[r] - v.afall[i,j,r]
                   for i in 1:nT, j in 1:nP, r in 1:nR]

    # INTDEMAND  (HT 36)
    R[:INTDEMAND] = [v.qf[i,j,r] + v.af[i,j,r] - qo_n(j,r) + v.ao[j,r] +
                     d.ESUBT[j]*(v.pf[i,j,r] - v.af[i,j,r] - ps_n(j,r) - v.ao[j,r])
                     for i in 1:nT, j in 1:nP, r in 1:nR]

    # DMNDDPRICE  (HT 20)
    R[:DMNDDPRICE] = [v.pfd[i,j,r] - v.tfd[i,j,r] - pm_t(i,r) for i in 1:nT, j in 1:nP, r in 1:nR]

    # DMNDIPRICES  (HT 23)
    R[:DMNDIPRICES] = [v.pfm[i,j,r] - v.tfm[i,j,r] - v.pim[i,r] for i in 1:nT, j in 1:nP, r in 1:nR]

    # ICOMPRICE  (HT 30)
    R[:ICOMPRICE] = [v.pf[i,j,r] - C.FMSHR[i,j,r]*v.pfm[i,j,r] - (1-C.FMSHR[i,j,r])*v.pfd[i,j,r]
                     for i in 1:nT, j in 1:nP, r in 1:nR]

    # INDIMP  (HT 31): qfm = qf - ESUBD*(pfm - pf)  [higher import price → less import demand]
    R[:INDIMP] = [v.qfm[i,j,r] - v.qf[i,j,r] + d.ESUBD[i]*(v.pfm[i,j,r]-v.pf[i,j,r])
                  for i in 1:nT, j in 1:nP, r in 1:nR]

    # INDDOM  (HT 32): qfd = qf - ESUBD*(pfd - pf)  [higher domestic price → less domestic demand]
    R[:INDDOM] = [v.qfd[i,j,r] - v.qf[i,j,r] + d.ESUBD[i]*(v.pfd[i,j,r]-v.pf[i,j,r])
                  for i in 1:nT, j in 1:nP, r in 1:nR]

    # AFEWORLD
    R[:AFEWORLD] = [v.afe[i,j,r] - v.afecom[i] - v.afesec[j] - v.afereg[r] - v.afeall[i,j,r]
                    for i in 1:nE, j in 1:nP, r in 1:nR]

    # VAPRICE  (HT 33)
    R[:VAPRICE] = [v.pva[j,r] - sum(C.SVA[k,j,r]*(v.pfe[k,j,r]-v.afe[k,j,r]) for k in 1:nE)
                   for j in 1:nP, r in 1:nR]

    # ENDWDEMAND  (HT 34)
    R[:ENDWDEMAND] = [v.qfe[i,j,r] + v.afe[i,j,r] - v.qva[j,r] +
                      d.ESUBVA[j]*(v.pfe[i,j,r] - v.afe[i,j,r] - v.pva[j,r])
                      for i in 1:nE, j in 1:nP, r in 1:nR]

    # MPFACTPRICE  (HT 16)  – mobile endowments
    R[:MPFACTPRICE] = [begin
        ei = iE(s.ENDWM_COMM[mi2]); ni = iN(s.ENDWM_COMM[mi2])
        v.pfe[ei,j,r] - v.tf[ei,j,r] - v.pm[ni,r]
    end for (mi2,_) in enumerate(s.ENDWM_COMM), j in 1:nP, r in 1:nR]

    # SPFACTPRICE  (HT 17)  – sluggish endowments
    R[:SPFACTPRICE] = [begin
        ei = iE(s.ENDWS_COMM[si2])
        v.pfe[ei,j,r] - v.tf[ei,j,r] - v.pmes[ei,j,r]
    end for (si2,_) in enumerate(s.ENDWS_COMM), j in 1:nP, r in 1:nR]

    # OUTPUTPRICES  (HT 15)
    R[:OUTPUTPRICES] = [begin
        ni = iN(s.PROD_COMM[pi])
        v.ps[ni,r] - v.to[ni,r] - v.pm[ni,r]
    end for pi in 1:nP, r in 1:nR]

    # TIURATIO
    R[:TIURATIO] = [100*C.INCOME[r]*v.del_taxriu[r] + C.TIU[r]*v.y[r] -
                    sum(d.VDFA[i,j,r]*v.tfd[i,j,r] + C.DFTAX[i,j,r]*(pm_t(i,r)+v.qfd[i,j,r]) +
                        d.VIFA[i,j,r]*v.tfm[i,j,r] + C.IFTAX[i,j,r]*(v.pim[i,r]+v.qfm[i,j,r])
                        for i in 1:nT, j in 1:nP) for r in 1:nR]

    # TFURATIO
    R[:TFURATIO] = [100*C.INCOME[r]*v.del_taxrfu[r] + C.TFU[r]*v.y[r] -
                    (sum(C.VFA[iD(e),j,r]*v.tf[iE(e),j,r] + C.ETAX[iE(e),j,r]*(v.pm[iN(e),r]+v.qfe[iE(e),j,r])
                         for e in s.ENDWM_COMM, j in 1:nP) +
                     sum(C.VFA[iD(e),j,r]*v.tf[iE(e),j,r] + C.ETAX[iE(e),j,r]*(v.pmes[iE(e),j,r]+v.qfe[iE(e),j,r])
                         for e in s.ENDWS_COMM, j in 1:nP)) for r in 1:nR]

    # ZEROPROFITS  (HT 6)
    R[:ZEROPROFITS] = [begin
        ni = iN(s.PROD_COMM[j])
        v.ps[ni,r] + v.ao[j,r] -
        sum(C.STC[iD(e),j,r]*(v.pfe[iE(e),j,r]-v.afe[iE(e),j,r]-v.ava[j,r]) for e in s.ENDW_COMM) -
        sum(C.STC[iD(t),j,r]*(v.pf[iT(t),j,r]-v.af[iT(t),j,r]) for t in s.TRAD_COMM) -
        v.profitslack[j,r]
    end for j in 1:nP, r in 1:nR]

    # TOUTRATIO
    R[:TOUTRATIO] = [100*C.INCOME[r]*v.del_taxrout[r] + C.TOUT[r]*v.y[r] -
                     sum(C.VOA[iN(p),r]*(-v.to[iN(p),r]) + C.PTAX[iN(p),r]*(v.pm[iN(p),r]+qo_n(iP(p),r))
                         for p in s.PROD_COMM) for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 4 – INVESTMENT, GLOBAL BANK & SAVINGS
    # ════════════════════════════════════════════════════════════════════

    # KAPSVCES  (HT 52)
    R[:KAPSVCES] = [v.ksvces[r] - sum((C.VOA[iN(h),r]/max(C.VOAcap[r],1e-10))*
                    v.qo[iN(h),r] for h in s.ENDWC_COMM) for r in 1:nR]

    # KAPRENTAL  (HT 53)
    R[:KAPRENTAL] = [v.rental[r] - sum((C.VOA[iN(h),r]/max(C.VOAcap[r],1e-10))*
                     v.ps[iN(h),r] for h in s.ENDWC_COMM) for r in 1:nR]

    # CAPGOODS  (HT 54)
    R[:CAPGOODS] = [v.qcgds[r] - sum((C.VOA[iN(h),r]/max(C.REGINV[r],1e-10))*
                    v.qo[iN(h),r] for h in s.CGDS_COMM) for r in 1:nR]

    # PRCGOODS  (HT 55)
    R[:PRCGOODS] = [v.pcgds[r] - sum((C.VOA[iN(h),r]/max(C.REGINV[r],1e-10))*
                    v.ps[iN(h),r] for h in s.CGDS_COMM) for r in 1:nR]

    # KBEGINNING  (HT 56)
    R[:KBEGINNING] = [v.kb[r] - v.ksvces[r] for r in 1:nR]

    # KEND  (HT 10)
    R[:KEND] = [v.ke[r] - C.INVKERATIO[r]*v.qcgds[r] - (1-C.INVKERATIO[r])*v.kb[r] for r in 1:nR]

    # RORCURRENT  (HT 57)
    R[:RORCURRENT] = [v.rorc[r] - C.GRNETRATIO[r]*(v.rental[r]-v.pcgds[r]) for r in 1:nR]

    # ROREXPECTED  (HT 58)
    R[:ROREXPECTED] = [v.rore[r] - v.rorc[r] + d.RORFLEX[r]*(v.ke[r]-v.kb[r]) for r in 1:nR]

    # BALDWIN – capital accumulation
    R[:BALDWIN] = [v.EXPAND[iE(h),r] - v.qcgds[r] + v.qo[iN(h),r]
                   for h in s.ENDWC_COMM, (ei,_) in enumerate(s.ENDWC_COMM), r in 1:nR
                   if h == s.ENDWC_COMM[ei]]
    # simpler scalar version (ENDWC has one element):
    R[:BALDWIN] = [v.EXPAND[1,r] - v.qcgds[r] + v.qo[iN(s.ENDWC_COMM[1]),r] for r in 1:nR]

    # RORGLOBAL  (HT 59)
    R[:RORGLOBAL] = [d.RORDELTA*v.rore[r] +
                     (1-d.RORDELTA)*((C.REGINV[r]/max(C.NETINV[r],1e-10))*v.qcgds[r] -
                                     (d.VDEP[r]/max(C.NETINV[r],1e-10))*v.kb[r]) -
                     d.RORDELTA*v.rorg - (1-d.RORDELTA)*v.globalcgds - v.cgdslack[r]
                     for r in 1:nR]

    # GLOBALINV  (HT 11)
    R[:GLOBALINV] = [d.RORDELTA*v.globalcgds + (1-d.RORDELTA)*v.rorg -
                     d.RORDELTA*sum((C.REGINV[r]/max(C.GLOBINV,1e-10))*v.qcgds[r] -
                                    (d.VDEP[r]/max(C.GLOBINV,1e-10))*v.kb[r] for r in 1:nR) -
                     (1-d.RORDELTA)*sum((C.NETINV[r]/max(C.GLOBINV,1e-10))*v.rore[r] for r in 1:nR)]

    # PRICGDS  (HT 60)
    R[:PRICGDS] = [v.pcgdswld - sum((C.NETINV[r]/max(C.GLOBINV,1e-10))*v.pcgds[r] for r in 1:nR)]

    # SAVEPRICE
    R[:SAVEPRICE] = [v.psave[r] - v.pcgds[r] -
                     sum(((C.NETINV[ss]-d.SAVE[ss])/max(C.GLOBINV,1e-10))*v.pcgds[ss] for ss in 1:nR) -
                     v.psaveslack[r] for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 5 – INTERNATIONAL TRADE
    # ════════════════════════════════════════════════════════════════════

    # EXPRICES  (HT 27)
    R[:EXPRICES] = [v.pfob[i,r,ss] - pm_t(i,r) + v.tx[i,r] + v.txs[i,r,ss]
                    for i in 1:nT, r in 1:nR, ss in 1:nR]

    # TEXPRATIO
    R[:TEXPRATIO] = [100*C.INCOME[r]*v.del_taxrexp[r] + C.TEX[r]*v.y[r] -
                     sum(d.VXMD[i,r,ss]*(-v.tx[i,r]-v.txs[i,r,ss]) +
                         C.XTAXD[i,r,ss]*(v.pfob[i,r,ss]+v.qxs[i,r,ss])
                         for i in 1:nT, ss in 1:nR) for r in 1:nR]

    # MKTPRICES  (HT 24)
    R[:MKTPRICES] = [v.pms[i,r,ss] - v.tm[i,ss] - v.tms[i,r,ss] - v.pcif[i,r,ss]
                     for i in 1:nT, r in 1:nR, ss in 1:nR]

    # DPRICEIMP  (HT 28)
    R[:DPRICEIMP] = [v.pim[i,ss] - sum(C.MSHRS[i,k,ss]*(v.pms[i,k,ss]-v.ams[i,k,ss]) for k in 1:nR)
                     for i in 1:nT, ss in 1:nR]

    # PRICETGT  (HT 25)
    R[:PRICETGT] = [v.pr[i,r] - pm_t(i,r) + v.pim[i,r] for i in 1:nT, r in 1:nR]

    # IMPORTDEMAND  (HT 29)
    R[:IMPORTDEMAND] = [v.qxs[i,r,ss] + v.ams[i,r,ss] - v.qim[i,ss] +
                        d.ESUBM[i]*(v.pms[i,r,ss]-v.ams[i,r,ss]-v.pim[i,ss])
                        for i in 1:nT, r in 1:nR, ss in 1:nR]

    # TIMPRATIO
    R[:TIMPRATIO] = [100*C.INCOME[r]*v.del_taxrimp[r] + C.TIM[r]*v.y[r] -
                     sum(d.VIMS[i,ss,r]*(v.tm[i,r]+v.tms[i,ss,r]) +
                         C.MTAX[i,ss,r]*(v.pcif[i,ss,r]+v.qxs[i,ss,r])
                         for i in 1:nT, ss in 1:nR) for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 6 – INTERNATIONAL TRANSPORT SERVICES
    # ════════════════════════════════════════════════════════════════════

    # TRANSTECHANGE
    R[:TRANSTECHANGE] = [v.atmfsd[m,i,r,ss] - v.atm[m] - v.atf[i] - v.ats[r] - v.atd[ss] - v.atall[m,i,r,ss]
                         for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR]

    # QTRANS_MFSD
    R[:QTRANS_MFSD] = [v.qtmfsd[m,i,r,ss] - v.qxs[i,r,ss] + v.atmfsd[m,i,r,ss]
                       for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR]

    # TRANS_DEMAND
    R[:TRANS_DEMAND] = [v.qtm[m] - sum(C.VTMUSESHR[m,i,r,ss]*v.qtmfsd[m,i,r,ss]
                        for i in 1:nT, r in 1:nR, ss in 1:nR) for m in 1:nM]

    # PTRANSPORT
    R[:PTRANSPORT] = [v.pt[m] - sum(C.VTSUPPSHR[m,r]*v.pm[iN(s.MARG_COMM[m]),r] for r in 1:nR)
                      for m in 1:nM]

    # TRANSCOSTINDEX
    R[:TRANSCOSTINDEX] = [v.ptrans[i,r,ss] -
                          sum(C.VTFSD_MSH[m,i,r,ss]*(v.pt[m]-v.atmfsd[m,i,r,ss]) for m in 1:nM)
                          for i in 1:nT, r in 1:nR, ss in 1:nR]

    # TRANSVCES  (HT 61)
    R[:TRANSVCES] = [v.qst[m,r] - v.qtm[m] - (v.pt[m] - v.pm[iN(s.MARG_COMM[m]),r])
                     for m in 1:nM, r in 1:nR]

    # FOBCIF  (HT 26')
    R[:FOBCIF] = [v.pcif[i,r,ss] - C.FOBSHR[i,r,ss]*v.pfob[i,r,ss] - C.TRNSHR[i,r,ss]*v.ptrans[i,r,ss]
                  for i in 1:nT, r in 1:nR, ss in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 7 – REGIONAL HOUSEHOLD
    # ════════════════════════════════════════════════════════════════════

    # FACTORINCPRICES  (HT 15, endowments)
    R[:FACTORINCPRICES] = [begin ni=iN(s.ENDW_COMM[ei])
        v.ps[ni,r] - v.to[ni,r] - v.pm[ni,r] end for ei in 1:nE, r in 1:nR]

    # TINCRATIO
    R[:TINCRATIO] = [100*C.INCOME[r]*v.del_taxrinc[r] + C.TINC[r]*v.y[r] -
                     sum(C.VOA[iN(e),r]*(-v.to[iN(e),r]) + C.PTAX[iN(e),r]*(v.pm[iN(e),r]+v.qo[iN(e),r])
                         for e in s.ENDW_COMM) for r in 1:nR]

    # ENDW_PRICE  (HT 50)  – sluggish
    R[:ENDW_PRICE] = [begin ei=iE(s.ENDWS_COMM[si2]); ni=iN(s.ENDWS_COMM[si2])
        v.pm[ni,r] - sum(C.REVSHR[ei,j,r]*v.pmes[ei,j,r] for j in 1:nP) end
        for (si2,_) in enumerate(s.ENDWS_COMM), r in 1:nR]

    # ENDW_SUPPLY  (HT 51)
    R[:ENDW_SUPPLY] = [begin ei=iE(s.ENDWS_COMM[si2]); ni=iN(s.ENDWS_COMM[si2])
        v.qoes[ei,j,r] - v.qo[ni,r] + v.endwslack[ei,r] - d.ETRAE[ei]*(v.pm[ni,r]-v.pmes[ei,j,r]) end
        for (si2,_) in enumerate(s.ENDWS_COMM), j in 1:nP, r in 1:nR]

    # FACTORINCOME
    R[:FACTORINCOME] = [C.FY[r]*v.fincome[r] -
                        (sum(C.VOM[iN(e),r]*(v.pm[iN(e),r]+v.qo[iN(e),r]) for e in s.ENDW_COMM) -
                         d.VDEP[r]*(v.pcgds[r]+v.kb[r])) for r in 1:nR]

    # DINDTAXRATIO
    R[:DINDTAXRATIO] = [v.del_indtaxr[r] - (v.del_taxrpc[r]+v.del_taxrgc[r]+v.del_taxriu[r]+
                        v.del_taxrfu[r]+v.del_taxrout[r]+v.del_taxrexp[r]+v.del_taxrimp[r])
                        for r in 1:nR]

    # DTAXRATIO
    R[:DTAXRATIO] = [v.del_ttaxr[r] - (v.del_taxrpc[r]+v.del_taxrgc[r]+v.del_taxriu[r]+
                     v.del_taxrfu[r]+v.del_taxrout[r]+v.del_taxrexp[r]+v.del_taxrimp[r]+v.del_taxrinc[r])
                     for r in 1:nR]

    # REGIONALINCOME
    # INCOME(r)*y(r) = FY(r)*fincome(r) + 100*INCOME(r)*del_indtaxr(r) + INDTAX(r)*y(r) + INCOME(r)*incomeslack(r)
    # Equivalently: FY(r)*y(r) = FY(r)*fincome(r) + 100*INCOME(r)*del_indtaxr(r) + INCOME(r)*incomeslack(r)
    R[:REGIONALINCOME] = [C.FY[r]*v.y[r] - C.FY[r]*v.fincome[r] -
                          100*C.INCOME[r]*v.del_indtaxr[r] -
                          C.INCOME[r]*v.incomeslack[r] for r in 1:nR]

    # DPARAV
    R[:DPARAV] = [v.dpav[r] - C.XSHRPRIV[r]*v.dppriv[r] - C.XSHRGOV[r]*v.dpgov[r] -
                  C.XSHRSAVE[r]*v.dpsave[r] for r in 1:nR]

    # UTILITELASTIC
    R[:UTILITELASTIC] = [v.uelas[r] - C.XSHRPRIV[r]*v.uepriv[r] + v.dpav[r] for r in 1:nR]

    # PRIVCONSEXP
    R[:PRIVCONSEXP] = [v.yp[r] - v.y[r] + (v.uepriv[r]-v.uelas[r]) - v.dppriv[r] for r in 1:nR]

    # GOVCONSEXP
    R[:GOVCONSEXP] = [v.yg[r] - v.y[r] - v.uelas[r] - v.dpgov[r] for r in 1:nR]

    # SAVING
    R[:SAVING] = [v.psave[r] + v.qsave[r] - v.y[r] - v.uelas[r] - v.dpsave[r] for r in 1:nR]

    # PRICEINDEXREG
    R[:PRICEINDEXREG] = [v.p[r] - C.XSHRPRIV[r]*v.ppriv[r] - C.XSHRGOV[r]*v.pgov[r] -
                         C.XSHRSAVE[r]*v.psave[r] for r in 1:nR]

    # UTILITY  (linearised, CDE log terms require updating UTILPRIV etc.)
    R[:UTILITY] = [v.u[r] - v.au[r] - (1/max(C.UTILELAS[r],1e-10))*(v.y[r]-v.pop[r]-v.p[r])
                   for r in 1:nR]

    # DISTPARSUM
    R[:DISTPARSUM] = [d.DPARSUM[r]*v.dpsum[r] - (C.DPARPRIV[r]*v.dppriv[r] +
                       C.DPARGOV[r]*v.dpgov[r] + C.DPARSAVE[r]*v.dpsave[r]) for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # MODULE 8 – EQUILIBRIUM CONDITIONS
    # ════════════════════════════════════════════════════════════════════

    # MKTCLDOM  (HT 3)
    R[:MKTCLDOM] = [v.qds[i,r] - sum(C.SHRDFM[i,j,r]*v.qfd[i,j,r] for j in 1:nP) -
                    C.SHRDPM[i,r]*v.qpd[i,r] - C.SHRDGM[i,r]*v.qgd[i,r]
                    for i in 1:nT, r in 1:nR]

    # MKTCLTRD_MARG  (HT 1)
    R[:MKTCLTRD_MARG] = [begin ti=iT(s.MARG_COMM[mi]); ni=iN(s.MARG_COMM[mi])
        v.qo[ni,r] - C.SHRDM[ti,r]*v.qds[ti,r] - C.SHRST[mi,r]*v.qst[mi,r] -
        sum(C.SHRXMD[ti,r,ss]*v.qxs[ti,r,ss] for ss in 1:nR) end
        for (mi,_) in enumerate(s.MARG_COMM), r in 1:nR]

    # MKTCLTRD_NMRG  (HT 1)
    R[:MKTCLTRD_NMRG] = [begin ti=iT(s.NMRG_COMM[ni2]); ni=iN(s.NMRG_COMM[ni2])
        v.qo[ni,r] - C.SHRDM[ti,r]*v.qds[ti,r] -
        sum(C.SHRXMD[ti,r,ss]*v.qxs[ti,r,ss] for ss in 1:nR) end
        for (ni2,_) in enumerate(s.NMRG_COMM), r in 1:nR]

    # MKTCLIMP  (HT 2)
    R[:MKTCLIMP] = [v.qim[i,r] - sum(C.SHRIFM[i,j,r]*v.qfm[i,j,r] for j in 1:nP) -
                    C.SHRIPM[i,r]*v.qpm[i,r] - C.SHRIGM[i,r]*v.qgm[i,r]
                    for i in 1:nT, r in 1:nR]

    # MKTCLENDWM  (HT 4)
    R[:MKTCLENDWM] = [begin ei=iE(s.ENDWM_COMM[mi2]); ni=iN(s.ENDWM_COMM[mi2])
        v.qo[ni,r] - sum(C.SHREM[mi2,j,r]*v.qfe[ei,j,r] for j in 1:nP) - v.endwslack[ei,r] end
        for (mi2,_) in enumerate(s.ENDWM_COMM), r in 1:nR]

    # MKTCLENDWS  (HT 5)
    R[:MKTCLENDWS] = [begin ei=iE(s.ENDWS_COMM[si2])
        v.qoes[ei,j,r] - v.qfe[ei,j,r] end
        for (si2,_) in enumerate(s.ENDWS_COMM), j in 1:nP, r in 1:nR]

    # WALRAS_S / WALRAS_D / WALRAS  (HT 12–14)
    R[:WALRAS_S] = v.walras_sup - v.pcgdswld - v.globalcgds
    R[:WALRAS_D] = C.GLOBINV*v.walras_dem - sum(d.SAVE[r]*(v.psave[r]+v.qsave[r]) for r in 1:nR)
    R[:WALRAS]   = v.walras_sup - v.walras_dem - v.walraslack

    skip_appendix && return R

    # ════════════════════════════════════════════════════════════════════
    # APPENDIX A – SUMMARY INDICES
    # ════════════════════════════════════════════════════════════════════

    # REALRETURN
    R[:REALRETURN] = [v.pfactreal[ei,r] - v.pm[iN(s.ENDW_COMM[ei]),r] + v.ppriv[r]
                      for ei in 1:nE, r in 1:nR]

    # PRIMFACTPR
    R[:PRIMFACTPR] = [C.VENDWREG[r]*v.pfactor[r] -
                      sum(C.VOM[iN(e),r]*v.pm[iN(e),r] for e in s.ENDW_COMM) for r in 1:nR]

    # PRIMFACTPRWLD
    R[:PRIMFACTPRWLD] = C.VENDWWLD*v.pfactwld - sum(C.VENDWREG[r]*v.pfactor[r] for r in 1:nR)

    # REGSUPRICE  (HT 64)
    R[:REGSUPRICE] = [C.VXWREGION[r]*v.psw[r] -
                      sum(d.VXWD[i,r,ss]*v.pfob[i,r,ss] for i in 1:nT, ss in 1:nR) -
                      sum(d.VST[m,r]*v.pm[iN(s.MARG_COMM[m]),r] for m in 1:nM) for r in 1:nR]

    # REGDEMPRICE  (HT 65)
    R[:REGDEMPRICE] = [C.VIWREGION[r]*v.pdw[r] -
                       sum(d.VIWS[i,ss,r]*v.pcif[i,ss,r] for i in 1:nT, ss in 1:nR) for r in 1:nR]

    # TOTeq  (HT 66)
    R[:TOTeq] = [v.tot[r] - v.psw[r] + v.pdw[r] for r in 1:nR]

    # VGDP_r  (HT 70)
    R[:VGDP_r] = [C.GDP[r]*v.vgdp[r] -
                  (sum(C.VGA[i,r]*(v.qg[i,r]+v.pg[i,r]) + C.VPA[i,r]*(v.qp[i,r]+v.pp[i,r]) for i in 1:nT) +
                   C.REGINV[r]*(v.qcgds[r]+v.pcgds[r]) +
                   sum(d.VXWD[i,r,ss]*(v.qxs[i,r,ss]+v.pfob[i,r,ss]) for i in 1:nT, ss in 1:nR) +
                   sum(d.VST[m,r]*(v.qst[m,r]+v.pm[iN(s.MARG_COMM[m]),r]) for m in 1:nM) -
                   sum(d.VIWS[i,ss,r]*(v.qxs[i,ss,r]+v.pcif[i,ss,r]) for i in 1:nT, ss in 1:nR))
                  for r in 1:nR]

    # PGDP_r  (HT 71)
    R[:PGDP_r] = [C.GDP[r]*v.pgdp[r] -
                  (sum(C.VGA[i,r]*v.pg[i,r] + C.VPA[i,r]*v.pp[i,r] for i in 1:nT) +
                   C.REGINV[r]*v.pcgds[r] +
                   sum(d.VXWD[i,r,ss]*v.pfob[i,r,ss] for i in 1:nT, ss in 1:nR) +
                   sum(d.VST[m,r]*v.pm[iN(s.MARG_COMM[m]),r] for m in 1:nM) -
                   sum(d.VIWS[i,ss,r]*v.pcif[i,ss,r] for i in 1:nT, ss in 1:nR))
                  for r in 1:nR]

    # QGDP_r
    R[:QGDP_r] = [C.GDP[r]*v.qgdp[r] -
                  (sum(C.VGA[i,r]*v.qg[i,r] + C.VPA[i,r]*v.qp[i,r] for i in 1:nT) +
                   C.REGINV[r]*v.qcgds[r] +
                   sum(d.VXWD[i,r,ss]*v.qxs[i,r,ss] for i in 1:nT, ss in 1:nR) +
                   sum(d.VST[m,r]*v.qst[m,r] for m in 1:nM) -
                   sum(d.VIWS[i,ss,r]*v.qxs[i,ss,r] for i in 1:nT, ss in 1:nR))
                  for r in 1:nR]

    # COMPVALADEQ
    R[:COMPVALADEQ] = [v.compvalad[pi,r] - qo_n(pi,r) + v.qgdp[r] for pi in 1:nP, r in 1:nR]

    # Trade value indices (HT 73–80)
    R[:VREGEX_ir] = [begin w=C.VXW[i,r]; rhs = sum(d.VXWD[i,r,ss]*(v.qxs[i,r,ss]+v.pfob[i,r,ss]) for ss in 1:nR)
        C.isMARG[i] ? w*v.vxwfob[i,r] - rhs - d.VST[iM(s.TRAD_COMM[i]),r]*(v.qst[iM(s.TRAD_COMM[i]),r]+v.pm[iN(s.TRAD_COMM[i]),r]) :
                      w*v.vxwfob[i,r] - rhs end for i in 1:nT, r in 1:nR]
    R[:VREGEX_r]  = [C.VXWREGION[r]*v.vxwreg[r] - sum(C.VXW[i,r]*v.vxwfob[i,r] for i in 1:nT) for r in 1:nR]
    R[:VREGIM_is] = [C.VIW[i,ss]*v.viwcif[i,ss] - sum(d.VIWS[i,r,ss]*(v.pcif[i,r,ss]+v.qxs[i,r,ss]) for r in 1:nR)
                     for i in 1:nT, ss in 1:nR]
    R[:VREGIM_s]  = [C.VIWREGION[r]*v.viwreg[r] - sum(C.VIW[i,r]*v.viwcif[i,r] for i in 1:nT) for r in 1:nR]

    # Trade balance  (HT 97–98)
    R[:TRADEBAL_i] = [v.DTBALi[i,r] - (C.VXW[i,r]/100)*v.vxwfob[i,r] + (C.VIW[i,r]/100)*v.viwcif[i,r]
                      for i in 1:nT, r in 1:nR]
    R[:TRADEBALANCE]= [v.DTBAL[r] - (C.VXWREGION[r]/100)*v.vxwreg[r] + (C.VIWREGION[r]/100)*v.viwreg[r]
                       for r in 1:nR]
    R[:DTBALRATIO] = [100*C.INCOME[r]*v.DTBALR[r] - 100*v.DTBAL[r] + C.TBAL[r]*v.y[r] for r in 1:nR]

    # ════════════════════════════════════════════════════════════════════
    # APPENDIX B – EQUIVALENT VARIATION
    # ════════════════════════════════════════════════════════════════════

    R[:GOVUSHD]    = [v.ygev[r] - v.pop[r] - v.ugev[r] for r in 1:nR]
    R[:PRIVDMNDSEV]= [v.qpev[i,r] - v.pop[r] - C.EY[i,r]*(v.ypev[r]-v.pop[r])
                      for i in 1:nT, r in 1:nR]       # prices fixed in EV shadow system
    R[:PRIVATEUEV] = [v.ypev[r] - v.pop[r] - C.UELASPRIV[r]*v.upev[r] for r in 1:nR]
    R[:UTILELASPRIVEV] = [v.ueprivev[r] - sum(C.XWCONSHR[i,r]*(v.qpev[i,r]-v.ypev[r]) for i in 1:nT)
                          for r in 1:nR]
    R[:DPARAVEV]   = [v.dpavev[r] - C.XSHRPRIV[r]*v.dppriv[r] -
                      C.XSHRGOV[r]*v.dpgov[r] - C.XSHRSAVE[r]*v.dpsave[r] for r in 1:nR]
    R[:UTILITELASTICEV] = [v.uelasev[r] - C.XSHRPRIV[r]*v.ueprivev[r] + v.dpavev[r] for r in 1:nR]
    R[:PCONSEXPEV] = [v.ypev[r] - v.yev[r] + (v.ueprivev[r]-v.uelasev[r]) - v.dppriv[r] for r in 1:nR]
    R[:GOVCONSEXPEV] = [v.ygev[r] - v.yev[r] - v.uelasev[r] - v.dpgov[r] for r in 1:nR]
    R[:SAVINGEV]   = [v.ysaveev[r] - v.yev[r] - v.uelasev[r] - v.dpsave[r] for r in 1:nR]
    R[:SAVEUEV]    = [v.qsaveev[r] - v.ysaveev[r] for r in 1:nR]
    # INCOME_EQUIV: u(r) = ... + [1/UTILELASEV]*[yev(r)-pop(r)]  (log terms omitted in linear approx)
    R[:INCOME_EQUIV] = [v.u[r] - v.au[r] - (1/max(C.UTILELAS[r],1e-10))*(v.yev[r]-v.pop[r])
                        for r in 1:nR]
    R[:EVREG] = [v.EV[r] - (C.INCOME[r]/100)*v.yev[r] for r in 1:nR]
    R[:EVWLD] = [v.WEV - sum(v.EV[r] for r in 1:nR)]

    # ════════════════════════════════════════════════════════════════════
    # APPENDIX C – WELFARE DECOMPOSITION (EV_ALT)
    # ════════════════════════════════════════════════════════════════════
    R[:EV_DECOMPOSITION] = [begin
        tax_eff = C.EVSCALFACT[r] * 0.01 * (
            sum(C.PTAX[iN(i),r]*(v.qo[iN(i),r]-v.pop[r]) for i in s.NSAV_COMM) +
            sum(C.ETAX[iE(e),j,r]*(v.qfe[iE(e),j,r]-v.pop[r]) for e in s.ENDW_COMM, j in 1:nP) +
            sum(C.IFTAX[i,j,r]*(v.qfm[i,j,r]-v.pop[r]) + C.DFTAX[i,j,r]*(v.qfd[i,j,r]-v.pop[r])
                for i in 1:nT, j in 1:nP) +
            sum(C.IPTAX[i,r]*(v.qpm[i,r]-v.pop[r]) + C.DPTAX[i,r]*(v.qpd[i,r]-v.pop[r]) +
                C.IGTAX[i,r]*(v.qgm[i,r]-v.pop[r]) + C.DGTAX[i,r]*(v.qgd[i,r]-v.pop[r]) for i in 1:nT) +
            sum(C.XTAXD[i,r,ss]*(v.qxs[i,r,ss]-v.pop[r]) + C.MTAX[i,ss,r]*(v.qxs[i,ss,r]-v.pop[r])
                for i in 1:nT, ss in 1:nR) +
            sum(C.VOA[iN(e),r]*(v.qo[iN(e),r]-v.pop[r]) for e in s.ENDW_COMM) -
            d.VDEP[r]*(v.kb[r]-v.pop[r]) +
            sum(C.VOA[iN(p),r]*v.ao[iP(p),r] for p in s.PROD_COMM) +
            sum(C.VVA[j,r]*v.ava[j,r] for j in 1:nP) +
            sum(C.VFA[iD(e),j,r]*v.afe[iE(e),j,r] for e in s.ENDW_COMM, j in 1:nP) +
            sum(C.VFA[iD(t),j,r]*v.af[iT(t),j,r] for t in s.TRAD_COMM, j in 1:nP) +
            sum(d.VTMFSD[m,i,ss,r]*v.atmfsd[m,i,ss,r] for m in 1:nM, i in 1:nT, ss in 1:nR) +
            sum(d.VIMS[i,ss,r]*v.ams[i,ss,r] for i in 1:nT, ss in 1:nR) +
            sum(d.VXWD[i,r,ss]*v.pfob[i,r,ss] for i in 1:nT, ss in 1:nR) +
            sum(d.VST[m,r]*v.pm[iN(s.MARG_COMM[m]),r] for m in 1:nM) +
            C.NETINV[r]*v.pcgds[r] -
            sum(d.VXWD[i,ss,r]*v.pfob[i,ss,r] for i in 1:nT, ss in 1:nR) -
            sum(C.VTMD[m,r]*v.pt[m] for m in 1:nM) -
            d.SAVE[r]*v.psave[r])
        v.EV_ALT[r] - tax_eff - 0.01*C.INCOME[r]*v.pop[r]
    end for r in 1:nR]

    R[:WORLDEV] = [v.WEV_ALT - sum(v.EV_ALT[r] for r in 1:nR)]

    return R
end


