# gtap_analytical.jl
# Analytical sparse Jacobian for the linearised GTAP v6.2 model.
# For a linear (TABLO-style) model, A[i,j] = coefficient of x_endo[j] in equation i.
# This avoids automatic differentiation entirely.
#
# Usage:
#   include("gtap_analytical.jl")
#   A = build_A_analytical(d, s, C)           # sparse matrix, ~seconds
#   b = -F_core(zeros(n_endog(s)), make_exog_zero(s), d, s, C)  # zero at benchmark
#   # Apply shock to exog, recompute b, then solve: x = lu(A) \ b

if !isdefined(Main, :GTAPSets); include("gtap_v62.jl"); end
if !isdefined(Main, :CORE_EQ_NAMES); include("gtap_pack.jl"); end
using SparseArrays

# ─────────────────────────────────────────────────────────────────────────────
# 1. Equation row-offset table
# ─────────────────────────────────────────────────────────────────────────────

function core_eq_sizes(s::GTAPSets)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEM=length(s.ENDWM_COMM)
    nMARG=length(s.MARG_COMM); nNMRG=length(s.NMRG_COMM)
    Dict{Symbol,Int}(
        # Module 1
        :GPRICEINDEX  => nR,    :GOVDMNDS => nT*nR,   :GOVU       => nR,
        :GHHDPRICE    => nT*nR, :GHHIPRICES=> nT*nR,  :GCOMPRICE  => nT*nR,
        :GHHLDAGRIMP  => nT*nR, :GHHLDDOM => nT*nR,   :TGCRATIO   => nR,
        # Module 2
        :PHHLDINDEX   => nR,    :PRIVATEU => nR,       :UTILELASPRIV=> nR,
        :PRIVDMNDS    => nT*nR, :TPDSHIFT => nT*nR,    :PHHDPRICE  => nT*nR,
        :TPMSHIFT     => nT*nR, :PHHIPRICES=> nT*nR,   :PCOMPRICE  => nT*nR,
        :PHHLDDOM     => nT*nR, :PHHLDAGRIMP=> nT*nR,  :TPCRATIO   => nR,
        # Module 3
        :VADEMAND     => nP*nR,     :INTDEMAND  => nT*nP*nR,
        :DMNDDPRICE   => nT*nP*nR,  :DMNDIPRICES=> nT*nP*nR,
        :ICOMPRICE    => nT*nP*nR,  :INDIMP     => nT*nP*nR,
        :INDDOM       => nT*nP*nR,
        :VAPRICE      => nP*nR,     :ENDWDEMAND => nE*nP*nR,
        :MPFACTPRICE  => nEM*nP*nR, :SPFACTPRICE=> nES*nP*nR,
        :OUTPUTPRICES => nP*nR,
        :TIURATIO     => nR,        :TFURATIO   => nR,
        :ZEROPROFITS  => nP*nR,     :TOUTRATIO  => nR,
        # Module 4
        :KAPSVCES  => nR, :KAPRENTAL => nR, :CAPGOODS  => nR, :PRCGOODS  => nR,
        :KBEGINNING=> nR, :KEND      => nR, :RORCURRENT=> nR, :ROREXPECTED=>nR,
        :BALDWIN   => nR, :RORGLOBAL => nR, :GLOBALINV => 1,
        :PRICGDS   => 1,  :SAVEPRICE => nR,
        # Module 5
        :EXPRICES    => nT*nR*nR, :TEXPRATIO => nR,
        :MKTPRICES   => nT*nR*nR, :DPRICEIMP => nT*nR,
        :PRICETGT    => nT*nR,    :IMPORTDEMAND=> nT*nR*nR,
        :TIMPRATIO   => nR,
        # Module 6
        :QTRANS_MFSD    => nM*nT*nR*nR, :TRANS_DEMAND  => nM,
        :PTRANSPORT     => nM,           :TRANSCOSTINDEX=> nT*nR*nR,
        :TRANSVCES      => nM*nR,        :FOBCIF        => nT*nR*nR,
        # Module 7
        :FACTORINCPRICES=> nE*nR,   :TINCRATIO  => nR,
        :ENDW_PRICE     => nES*nR,  :ENDW_SUPPLY=> nES*nP*nR,
        :FACTORINCOME   => nR,      :DINDTAXRATIO=> nR,
        :DTAXRATIO      => nR,      :REGIONALINCOME=> nR,
        :DPARAV         => nR,      :UTILITELASTIC=> nR,
        :PRIVCONSEXP    => nR,      :GOVCONSEXP => nR,
        :SAVING         => nR,      :PRICEINDEXREG=> nR,
        :UTILITY        => nR,      :DISTPARSUM => nR,
        # Module 8
        :MKTCLDOM      => nT*nR,    :MKTCLTRD_MARG=> nMARG*nR,
        :MKTCLTRD_NMRG => nNMRG*nR, :MKTCLIMP   => nT*nR,
        :MKTCLENDWM    => nEM*nR,   :MKTCLENDWS  => nES*nP*nR,
        :WALRAS_S      => 1,        :WALRAS_D    => 1,
        :WALRAS        => 1,
    )
end

function eq_row_offsets(s::GTAPSets)
    sizes = core_eq_sizes(s)
    offs  = Dict{Symbol,Int}()
    row   = 0
    for name in CORE_EQ_NAMES
        offs[name] = row + 1
        row += sizes[name]
    end
    return offs
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Main builder
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_A_analytical(d, s, C) → SparseMatrixCSC{Float64}

Construct the GTAP coefficient matrix analytically (no AD).
Runs in seconds for the full 1.3M-variable system.
"""
function build_A_analytical(d::GTAPData, s::GTAPSets, C)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEM=length(s.ENDWM_COMM)
    nEC=length(s.ENDWC_COMM); nNS=length(s.NSAV_COMM)
    nCGDS=length(s.CGDS_COMM)

    iT=C.iT; iE=C.iE; iP=C.iP; iN=C.iN; iD=C.iD

    _, voffs, n_var = endo_offsets(s)
    eoffs = eq_row_offsets(s)
    n_eq  = n_core_equations(s)

    # Pre-compute sluggish endowment mapping: ENDWS_COMM[si] → index in ENDW_COMM
    slug_in_endw  = [findfirst(==(e), s.ENDW_COMM) for e in s.ENDWS_COMM]
    endw_to_slug  = Dict(ei => si for (si,ei) in enumerate(slug_in_endw))

    # ── Column helpers (all 1-based) ─────────────────────────────────────────
    # 1D shapes
    c1(nm, k)          = voffs[nm] + k - 1
    # 2D shape (n1,n2): [i,r] → col-major = (r-1)*n1 + i
    c2(nm, i, r, n1)   = voffs[nm] + (r-1)*n1 + i - 1
    # 3D shape (n1,n2,n3): [i,j,r]
    c3(nm,i,j,r,n1,n2) = voffs[nm] + (r-1)*n1*n2 + (j-1)*n1 + i - 1
    # 4D shape (n1,n2,n3,n4): [m,i,r,ss]
    c4(nm,m,i,r,ss,n1,n2,n3) = voffs[nm] + (ss-1)*n1*n2*n3 + (r-1)*n1*n2 + (i-1)*n1 + m - 1

    # Frequently used variable column shortcuts
    col_pm(ni,r)  = c2(:pm,  ni, r, nNS)
    col_ps(ni,r)  = c2(:ps,  ni, r, nNS)
    col_pfe(ei,j,r) = c3(:pfe, ei, j, r, nE, nP)
    col_qfe(ei,j,r) = c3(:qfe, ei, j, r, nE, nP)
    col_pmes(si,j,r)= c3(:pmes_slug, si, j, r, nES, nP)
    col_qoes(si,j,r)= c3(:qoes_slug, si, j, r, nES, nP)
    # qo_prod_cgds shape (nT+nCGDS, nR); index k = iN(p)-nE for p ∈ PROD_COMM
    col_qo(ns_idx,r)= c2(:qo_prod_cgds, ns_idx-nE, r, nT+nCGDS)
    # For 3D (nT,nR,nR): [i,r,ss] → (ss-1)*nT*nR + (r-1)*nT + i
    c3b(nm,i,r,ss) = voffs[nm] + (ss-1)*nT*nR + (r-1)*nT + i - 1
    # For qtmfsd (nM,nT,nR,nR): [m,i,r,ss] → (ss-1)*nM*nT*nR + (r-1)*nM*nT + (i-1)*nM + m
    c4b(nm,m,i,r,ss) = voffs[nm] + (ss-1)*nM*nT*nR + (r-1)*nM*nT + (i-1)*nM + m - 1

    # Pre-allocate (upper bound on nnz; resize at end)
    cap = 30_000_000
    II = Vector{Int32}(undef, cap)
    JJ = Vector{Int32}(undef, cap)
    VV = Vector{Float64}(undef, cap)
    ptr = Ref(0)

    @inline function ae!(row::Int, col::Int, val::Real)
        iszero(val) && return
        k = ptr[] + 1
        II[k] = row; JJ[k] = col; VV[k] = Float64(val)
        ptr[] = k
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 1 – GOVERNMENT
    # ═════════════════════════════════════════════════════════════════════════

    # GPRICEINDEX[r]: pgov[r] - Σ_i VGA[i,r]/GOVEXP[r]*pg[i,r] = 0
    let er = eoffs[:GPRICEINDEX]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:pgov, r), 1.0)
            g = max(C.GOVEXP[r], 1e-10)
            for i in 1:nT
                ae!(row, c2(:pg, i, r, nT), -C.VGA[i,r]/g)
            end
        end
    end

    # GOVDMNDS[i,r]: qg[i,r] - ug[r] + pg[i,r] - pgov[r] = pop[r]
    let er = eoffs[:GOVDMNDS]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:qg,   i, r, nT), 1.0)
            ae!(row, c1(:ug,   r),        -1.0)
            ae!(row, c2(:pg,   i, r, nT), 1.0)
            ae!(row, c1(:pgov, r),        -1.0)
        end
    end

    # GOVU[r]: yg[r] - pgov[r] - ug[r] = pop[r]
    let er = eoffs[:GOVU]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:yg,   r),  1.0)
            ae!(row, c1(:pgov, r), -1.0)
            ae!(row, c1(:ug,   r), -1.0)
        end
    end

    # GHHDPRICE[i,r]: pgd[i,r] - pm[iN(TRAD[i]),r] = tgd[i,r]  (exog)
    let er = eoffs[:GHHDPRICE]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:pgd, i, r, nT),             1.0)
            ae!(row, col_pm(iN(s.TRAD_COMM[i]), r), -1.0)
        end
    end

    # GHHIPRICES[i,r]: pgm[i,r] - pim[i,r] = tgm[i,r]
    let er = eoffs[:GHHIPRICES]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:pgm, i, r, nT), 1.0)
            ae!(row, c2(:pim, i, r, nT), -1.0)
        end
    end

    # GCOMPRICE[i,r]: pg[i,r] - GMSHR*pgm - (1-GMSHR)*pgd = 0
    let er = eoffs[:GCOMPRICE]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            gm = C.GMSHR[i,r]
            ae!(row, c2(:pg,  i, r, nT),  1.0)
            ae!(row, c2(:pgm, i, r, nT), -gm)
            ae!(row, c2(:pgd, i, r, nT), -(1-gm))
        end
    end

    # GHHLDAGRIMP[i,r]: qgm - qg - ESUBD*(pg - pgm) = 0
    let er = eoffs[:GHHLDAGRIMP]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            σ = d.ESUBD[i]
            ae!(row, c2(:qgm, i, r, nT),  1.0)
            ae!(row, c2(:qg,  i, r, nT), -1.0)
            ae!(row, c2(:pg,  i, r, nT), -σ)
            ae!(row, c2(:pgm, i, r, nT),  σ)
        end
    end

    # GHHLDDOM[i,r]: qgd - qg - ESUBD*(pg - pgd) = 0
    let er = eoffs[:GHHLDDOM]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            σ = d.ESUBD[i]
            ae!(row, c2(:qgd, i, r, nT),  1.0)
            ae!(row, c2(:qg,  i, r, nT), -1.0)
            ae!(row, c2(:pg,  i, r, nT), -σ)
            ae!(row, c2(:pgd, i, r, nT),  σ)
        end
    end

    # TGCRATIO[r]: 100*INCOME*del_taxrgc + TGC*y - Σ_i[DGTAX*(pm+qgd) + IGTAX*(pim+qgm)] = exog
    let er = eoffs[:TGCRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrgc, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),           C.TGC[r])
            for i in 1:nT
                ni = iN(s.TRAD_COMM[i])
                ae!(row, col_pm(ni, r),         -C.DGTAX[i,r])
                ae!(row, c2(:qgd, i, r, nT),    -C.DGTAX[i,r])
                ae!(row, c2(:pim, i, r, nT),    -C.IGTAX[i,r])
                ae!(row, c2(:qgm, i, r, nT),    -C.IGTAX[i,r])
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 2 – PRIVATE CONSUMPTION
    # ═════════════════════════════════════════════════════════════════════════

    # PHHLDINDEX[r]: ppriv[r] - Σ_i CONSHR[i,r]*pp[i,r] = 0
    let er = eoffs[:PHHLDINDEX]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:ppriv, r), 1.0)
            for i in 1:nT
                ae!(row, c2(:pp, i, r, nT), -C.CONSHR[i,r])
            end
        end
    end

    # PRIVATEU[r]: yp[r] - ppriv[r] - UELASPRIV*up[r] = pop[r]
    let er = eoffs[:PRIVATEU]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:yp,    r),  1.0)
            ae!(row, c1(:ppriv, r), -1.0)
            ae!(row, c1(:up,    r), -C.UELASPRIV[r])
        end
    end

    # UTILELASPRIV[r]: uepriv[r] - Σ_i XWCONSHR*(pp+qp) + (Σ_i XWCONSHR)*yp = 0
    let er = eoffs[:UTILELASPRIV]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:uepriv, r), 1.0)
            s_xw = 0.0
            for i in 1:nT
                w = C.XWCONSHR[i,r]
                ae!(row, c2(:pp, i, r, nT), -w)
                ae!(row, c2(:qp, i, r, nT), -w)
                s_xw += w
            end
            ae!(row, c1(:yp, r), s_xw)
        end
    end

    # PRIVDMNDS[i,r]: qp[i,r] - Σ_k EP[i,k,r]*pp[k,r] - EY[i,r]*yp[r] = pop terms
    let er = eoffs[:PRIVDMNDS]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:qp, i, r, nT), 1.0)
            for k in 1:nT
                ae!(row, c2(:pp, k, r, nT), -C.EP[i,k,r])
            end
            ae!(row, c1(:yp, r), -C.EY[i,r])
        end
    end

    # TPDSHIFT[i,r]: atpd[i,r] = tpd + tp  (both exog) → only atpd is endo
    let er = eoffs[:TPDSHIFT]
        for r in 1:nR, i in 1:nT
            ae!(er+(r-1)*nT+i-1, c2(:atpd, i, r, nT), 1.0)
        end
    end

    # PHHDPRICE[i,r]: ppd[i,r] - atpd[i,r] - pm[iN(TRAD[i]),r] = 0
    let er = eoffs[:PHHDPRICE]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:ppd,  i, r, nT),             1.0)
            ae!(row, c2(:atpd, i, r, nT),            -1.0)
            ae!(row, col_pm(iN(s.TRAD_COMM[i]), r), -1.0)
        end
    end

    # TPMSHIFT[i,r]: atpm[i,r] = tpm + tp  (both exog)
    let er = eoffs[:TPMSHIFT]
        for r in 1:nR, i in 1:nT
            ae!(er+(r-1)*nT+i-1, c2(:atpm, i, r, nT), 1.0)
        end
    end

    # PHHIPRICES[i,r]: ppm[i,r] - atpm[i,r] - pim[i,r] = 0
    let er = eoffs[:PHHIPRICES]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:ppm,  i, r, nT), 1.0)
            ae!(row, c2(:atpm, i, r, nT), -1.0)
            ae!(row, c2(:pim,  i, r, nT), -1.0)
        end
    end

    # PCOMPRICE[i,r]: pp[i,r] - PMSHR*ppm - (1-PMSHR)*ppd = 0
    let er = eoffs[:PCOMPRICE]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            pm_ = C.PMSHR[i,r]
            ae!(row, c2(:pp,  i, r, nT),  1.0)
            ae!(row, c2(:ppm, i, r, nT), -pm_)
            ae!(row, c2(:ppd, i, r, nT), -(1-pm_))
        end
    end

    # PHHLDDOM[i,r]: qpd - qp - ESUBD*(pp - ppd) = 0
    let er = eoffs[:PHHLDDOM]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            σ = d.ESUBD[i]
            ae!(row, c2(:qpd, i, r, nT),  1.0)
            ae!(row, c2(:qp,  i, r, nT), -1.0)
            ae!(row, c2(:pp,  i, r, nT), -σ)
            ae!(row, c2(:ppd, i, r, nT),  σ)
        end
    end

    # PHHLDAGRIMP[i,r]: qpm - qp - ESUBD*(pp - ppm) = 0
    let er = eoffs[:PHHLDAGRIMP]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            σ = d.ESUBD[i]
            ae!(row, c2(:qpm, i, r, nT),  1.0)
            ae!(row, c2(:qp,  i, r, nT), -1.0)
            ae!(row, c2(:pp,  i, r, nT), -σ)
            ae!(row, c2(:ppm, i, r, nT),  σ)
        end
    end

    # TPCRATIO[r]: 100*INCOME*del_taxrpc + TPC*y - Σ_i[DPTAX*(pm+qpd)+IPTAX*(pim+qpm)] = exog
    let er = eoffs[:TPCRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrpc, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),           C.TPC[r])
            for i in 1:nT
                ni = iN(s.TRAD_COMM[i])
                ae!(row, c2(:atpd, i, r, nT),   -d.VDPA[i,r])
                ae!(row, col_pm(ni, r),          -C.DPTAX[i,r])
                ae!(row, c2(:qpd, i, r, nT),     -C.DPTAX[i,r])
                ae!(row, c2(:atpm, i, r, nT),    -d.VIPA[i,r])
                ae!(row, c2(:pim, i, r, nT),     -C.IPTAX[i,r])
                ae!(row, c2(:qpm, i, r, nT),     -C.IPTAX[i,r])
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 3 – FIRMS
    # ═════════════════════════════════════════════════════════════════════════

    # VADEMAND[j,r]: qva[j,r] - qo[iN(j),r] + ESUBT[j]*(pva[j,r] - ps[iN(j),r]) = exog
    let er = eoffs[:VADEMAND]
        for r in 1:nR, j in 1:nP
            row = er + (r-1)*nP + j - 1
            ni  = iN(s.PROD_COMM[j])
            σ   = d.ESUBT[j]
            ae!(row, c2(:qva, j, r, nP),  1.0)
            ae!(row, col_qo(ni, r),       -1.0)
            ae!(row, c2(:pva, j, r, nP),   σ)
            ae!(row, col_ps(ni, r),       -σ)
        end
    end

    # INTDEMAND[i,j,r]: qf - qo + ESUBT*(pf - ps) = exog
    let er = eoffs[:INTDEMAND]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            ni  = iN(s.PROD_COMM[j])
            σ   = d.ESUBT[j]
            ae!(row, c3(:qf, i, j, r, nT, nP),  1.0)
            ae!(row, col_qo(ni, r),             -1.0)
            ae!(row, c3(:pf, i, j, r, nT, nP),   σ)
            ae!(row, col_ps(ni, r),             -σ)
        end
    end

    # DMNDDPRICE[i,j,r]: pfd - pm_t = tfd (exog)
    let er = eoffs[:DMNDDPRICE]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            ae!(row, c3(:pfd, i, j, r, nT, nP),        1.0)
            ae!(row, col_pm(iN(s.TRAD_COMM[i]), r),   -1.0)
        end
    end

    # DMNDIPRICES[i,j,r]: pfm - pim = tfm (exog)
    let er = eoffs[:DMNDIPRICES]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            ae!(row, c3(:pfm, i, j, r, nT, nP),  1.0)
            ae!(row, c2(:pim, i, r, nT),         -1.0)
        end
    end

    # ICOMPRICE[i,j,r]: pf - FMSHR*pfm - (1-FMSHR)*pfd = 0
    let er = eoffs[:ICOMPRICE]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            fm  = C.FMSHR[i,j,r]
            ae!(row, c3(:pf,  i, j, r, nT, nP),  1.0)
            ae!(row, c3(:pfm, i, j, r, nT, nP), -fm)
            ae!(row, c3(:pfd, i, j, r, nT, nP), -(1-fm))
        end
    end

    # INDIMP[i,j,r]: qfm - qf + ESUBD*(pfm - pf) = 0  →  qfm = qf - ESUBD*(pfm-pf)
    let er = eoffs[:INDIMP]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            σ   = d.ESUBD[i]
            ae!(row, c3(:qfm, i, j, r, nT, nP),  1.0)
            ae!(row, c3(:qf,  i, j, r, nT, nP), -1.0)
            ae!(row, c3(:pfm, i, j, r, nT, nP),  σ)
            ae!(row, c3(:pf,  i, j, r, nT, nP), -σ)
        end
    end

    # INDDOM[i,j,r]: qfd - qf + ESUBD*(pfd - pf) = 0  →  qfd = qf - ESUBD*(pfd-pf)
    let er = eoffs[:INDDOM]
        for r in 1:nR, j in 1:nP, i in 1:nT
            row = er + (r-1)*nT*nP + (j-1)*nT + i - 1
            σ   = d.ESUBD[i]
            ae!(row, c3(:qfd, i, j, r, nT, nP),  1.0)
            ae!(row, c3(:qf,  i, j, r, nT, nP), -1.0)
            ae!(row, c3(:pfd, i, j, r, nT, nP),  σ)
            ae!(row, c3(:pf,  i, j, r, nT, nP), -σ)
        end
    end

    # VAPRICE[j,r]: pva[j,r] - Σ_k SVA[k,j,r]*pfe[k,j,r] = exog(afe)
    let er = eoffs[:VAPRICE]
        for r in 1:nR, j in 1:nP
            row = er + (r-1)*nP + j - 1
            ae!(row, c2(:pva, j, r, nP), 1.0)
            for k in 1:nE
                ae!(row, col_pfe(k, j, r), -C.SVA[k,j,r])
            end
        end
    end

    # ENDWDEMAND[ei,j,r]: qfe - qva + ESUBVA*(pfe - pva) = exog(afe)
    let er = eoffs[:ENDWDEMAND]
        for r in 1:nR, j in 1:nP, ei in 1:nE
            row = er + (r-1)*nE*nP + (j-1)*nE + ei - 1
            σ   = d.ESUBVA[j]
            ae!(row, col_qfe(ei, j, r),   1.0)
            ae!(row, c2(:qva, j, r, nP), -1.0)
            ae!(row, col_pfe(ei, j, r),   σ)
            ae!(row, c2(:pva, j, r, nP), -σ)
        end
    end

    # MPFACTPRICE[mi2,j,r]: pfe[ei,j,r] - pm[ni,r] = tf (exog)
    let er = eoffs[:MPFACTPRICE]
        for r in 1:nR, j in 1:nP, (mi2,e) in enumerate(s.ENDWM_COMM)
            row = er + (r-1)*nEM*nP + (j-1)*nEM + mi2 - 1
            ei  = iE(e); ni = iN(e)
            ae!(row, col_pfe(ei, j, r),  1.0)
            ae!(row, col_pm(ni, r),     -1.0)
        end
    end

    # SPFACTPRICE[si2,j,r]: pfe[ei,j,r] - pmes_slug[si2,j,r] = tf (exog)
    let er = eoffs[:SPFACTPRICE]
        for r in 1:nR, j in 1:nP, (si2,e) in enumerate(s.ENDWS_COMM)
            row = er + (r-1)*nES*nP + (j-1)*nES + si2 - 1
            ei  = iE(e)
            ae!(row, col_pfe(ei, j, r),  1.0)
            ae!(row, col_pmes(si2, j, r),-1.0)
        end
    end

    # OUTPUTPRICES[pi,r]: ps[ni,r] - pm[ni,r] = to (exog)
    let er = eoffs[:OUTPUTPRICES]
        for r in 1:nR, pi in 1:nP
            row = er + (r-1)*nP + pi - 1
            ni  = iN(s.PROD_COMM[pi])
            ae!(row, col_ps(ni, r),  1.0)
            ae!(row, col_pm(ni, r), -1.0)
        end
    end

    # TIURATIO[r]: 100*INCOME*del_taxriu + TIU*y - Σ_{i,j}[DFTAX*(pm+qfd)+IFTAX*(pim+qfm)] = exog
    let er = eoffs[:TIURATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxriu, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),           C.TIU[r])
            for j in 1:nP, i in 1:nT
                ni = iN(s.TRAD_COMM[i])
                ae!(row, col_pm(ni, r),              -C.DFTAX[i,j,r])
                ae!(row, c3(:qfd, i, j, r, nT, nP),  -C.DFTAX[i,j,r])
                ae!(row, c2(:pim, i, r, nT),          -C.IFTAX[i,j,r])
                ae!(row, c3(:qfm, i, j, r, nT, nP),  -C.IFTAX[i,j,r])
            end
        end
    end

    # TFURATIO[r]: 100*INCOME*del_taxrfu + TFU*y - Σ_{e∈ENDWM,j}ETAX*(pm+qfe)
    #                                              - Σ_{e∈ENDWS,j}ETAX*(pmes+qfe) = exog(tf)
    let er = eoffs[:TFURATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrfu, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),           C.TFU[r])
            for j in 1:nP
                for (mi2,e) in enumerate(s.ENDWM_COMM)
                    ei = iE(e); ni = iN(e)
                    tax = C.ETAX[ei,j,r]
                    ae!(row, col_pm(ni, r),      -tax)
                    ae!(row, col_qfe(ei, j, r),  -tax)
                end
                for (si2,e) in enumerate(s.ENDWS_COMM)
                    ei = iE(e)
                    tax = C.ETAX[ei,j,r]
                    ae!(row, col_pmes(si2, j, r), -tax)
                    ae!(row, col_qfe(ei, j, r),   -tax)
                end
            end
        end
    end

    # ZEROPROFITS[j,r]: ps[ni,r] - Σ_{e∈ENDW} STC*(pfe) - Σ_{t∈TRAD} STC*(pf) = exog
    let er = eoffs[:ZEROPROFITS]
        for r in 1:nR, j in 1:nP
            row = er + (r-1)*nP + j - 1
            ni  = iN(s.PROD_COMM[j])
            ae!(row, col_ps(ni, r), 1.0)
            for e in s.ENDW_COMM
                ae!(row, col_pfe(iE(e), j, r), -C.STC[iD(e),j,r])
            end
            for t in s.TRAD_COMM
                ae!(row, c3(:pf, iT(t), j, r, nT, nP), -C.STC[iD(t),j,r])
            end
        end
    end

    # TOUTRATIO[r]: 100*INCOME*del_taxrout + TOUT*y - Σ_{p∈PROD} PTAX*(pm+qo) = exog(to)
    let er = eoffs[:TOUTRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrout, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),            C.TOUT[r])
            for p in s.PROD_COMM
                ni = iN(p)
                ae!(row, col_pm(ni, r),   -C.PTAX[ni,r])
                ae!(row, col_qo(ni, r),   -C.PTAX[ni,r])
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 4 – INVESTMENT, GLOBAL BANK & SAVINGS
    # ═════════════════════════════════════════════════════════════════════════

    # KAPSVCES[r]: ksvces[r] = Σ_{h∈ENDWC} (VOA[iN(h),r]/VOAcap[r])*qo[iN(h),r]  (qo exog)
    let er = eoffs[:KAPSVCES]
        for r in 1:nR
            ae!(er+r-1, c1(:ksvces, r), 1.0)
        end
    end

    # KAPRENTAL[r]: rental[r] = Σ_{h∈ENDWC} (VOA/VOAcap)*ps[iN(h),r]
    let er = eoffs[:KAPRENTAL]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:rental, r), 1.0)
            g = max(C.VOAcap[r], 1e-10)
            for h in s.ENDWC_COMM
                ni = iN(h)
                ae!(row, col_ps(ni, r), -C.VOA[ni,r]/g)
            end
        end
    end

    # CAPGOODS[r]: qcgds[r] = Σ_{h∈CGDS} (VOA/REGINV)*qo[iN(h),r]  (qo endogenous here)
    let er = eoffs[:CAPGOODS]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:qcgds, r), 1.0)
            g = max(C.REGINV[r], 1e-10)
            for h in s.CGDS_COMM
                ni = iN(h)
                ae!(row, col_qo(ni, r), -C.VOA[ni,r]/g)
            end
        end
    end

    # PRCGOODS[r]: pcgds[r] = Σ_{h∈CGDS} (VOA/REGINV)*ps[iN(h),r]
    let er = eoffs[:PRCGOODS]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:pcgds, r), 1.0)
            g = max(C.REGINV[r], 1e-10)
            for h in s.CGDS_COMM
                ni = iN(h)
                ae!(row, col_ps(ni, r), -C.VOA[ni,r]/g)
            end
        end
    end

    # KBEGINNING[r]: kb - ksvces = 0
    let er = eoffs[:KBEGINNING]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:kb,     r),  1.0)
            ae!(row, c1(:ksvces, r), -1.0)
        end
    end

    # KEND[r]: ke - INVKERATIO*qcgds - (1-INVKERATIO)*kb = 0
    let er = eoffs[:KEND]
        for r in 1:nR
            row = er + r - 1
            ik  = C.INVKERATIO[r]
            ae!(row, c1(:ke,    r),  1.0)
            ae!(row, c1(:qcgds, r), -ik)
            ae!(row, c1(:kb,    r), -(1-ik))
        end
    end

    # RORCURRENT[r]: rorc - GRNETRATIO*(rental - pcgds) = 0
    let er = eoffs[:RORCURRENT]
        for r in 1:nR
            row = er + r - 1
            gr  = C.GRNETRATIO[r]
            ae!(row, c1(:rorc,   r),  1.0)
            ae!(row, c1(:rental, r), -gr)
            ae!(row, c1(:pcgds,  r),  gr)
        end
    end

    # ROREXPECTED[r]: rore - rorc + RORFLEX*(ke - kb) = 0
    let er = eoffs[:ROREXPECTED]
        for r in 1:nR
            row = er + r - 1
            rf  = d.RORFLEX[r]
            ae!(row, c1(:rore, r),  1.0)
            ae!(row, c1(:rorc, r), -1.0)
            ae!(row, c1(:ke,   r),  rf)
            ae!(row, c1(:kb,   r), -rf)
        end
    end

    # BALDWIN[r]: EXPAND[1,r] - qcgds + qo[iN(capital),r] = 0  (qo exog → only EXPAND,qcgds endo)
    let er = eoffs[:BALDWIN]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c2(:EXPAND, 1, r, nEC), 1.0)
            ae!(row, c1(:qcgds, r),         -1.0)
        end
    end

    # RORGLOBAL[r]: RORDELTA*rore + (1-RORDELTA)*(REGINV/NETINV*qcgds - VDEP/NETINV*kb)
    #               - RORDELTA*rorg - (1-RORDELTA)*globalcgds = cgdslack (exog)
    let er = eoffs[:RORGLOBAL]
        rd = d.RORDELTA
        for r in 1:nR
            row = er + r - 1
            ni  = max(C.NETINV[r], 1e-10)
            ae!(row, c1(:rore,      r),  rd)
            ae!(row, c1(:qcgds,     r),  (1-rd)*C.REGINV[r]/ni)
            ae!(row, c1(:kb,        r), -(1-rd)*d.VDEP[r]/ni)
            ae!(row, voffs[:rorg],      -rd)
            ae!(row, voffs[:globalcgds],-(1-rd))
        end
    end

    # GLOBALINV (scalar): RORDELTA*globalcgds + (1-RORDELTA)*rorg
    #                    - RORDELTA*Σ_r(REGINV/GLOBINV*qcgds - VDEP/GLOBINV*kb)
    #                    - (1-RORDELTA)*Σ_r(NETINV/GLOBINV*rore) = 0
    let er = eoffs[:GLOBALINV]
        row = er
        rd  = d.RORDELTA
        gb  = max(C.GLOBINV, 1e-10)
        ae!(row, voffs[:globalcgds],  rd)
        ae!(row, voffs[:rorg],        1-rd)
        for r in 1:nR
            ae!(row, c1(:qcgds, r), -rd*C.REGINV[r]/gb)
            ae!(row, c1(:kb,    r),  rd*d.VDEP[r]/gb)
            ae!(row, c1(:rore,  r), -(1-rd)*C.NETINV[r]/gb)
        end
    end

    # PRICGDS (scalar): pcgdswld - Σ_r (NETINV/GLOBINV)*pcgds[r] = 0
    let er = eoffs[:PRICGDS]
        row = er
        gb  = max(C.GLOBINV, 1e-10)
        ae!(row, voffs[:pcgdswld], 1.0)
        for r in 1:nR
            ae!(row, c1(:pcgds, r), -C.NETINV[r]/gb)
        end
    end

    # SAVEPRICE[r]: psave[r] - pcgds[r] - Σ_ss ((NETINV[ss]-SAVE[ss])/GLOBINV)*pcgds[ss] = psaveslack
    let er = eoffs[:SAVEPRICE]
        gb  = max(C.GLOBINV, 1e-10)
        # Precompute column correction vector for pcgds
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:psave, r), 1.0)
            ae!(row, c1(:pcgds, r), -1.0)
            for ss in 1:nR
                ae!(row, c1(:pcgds, ss), -(C.NETINV[ss]-d.SAVE[ss])/gb)
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 5 – INTERNATIONAL TRADE
    # ═════════════════════════════════════════════════════════════════════════

    # EXPRICES[i,r,ss]: pfob[i,r,ss] - pm_t(i,r) = -tx - txs  (exog)
    let er = eoffs[:EXPRICES]
        for ss in 1:nR, r in 1:nR, i in 1:nT
            row = er + (ss-1)*nT*nR + (r-1)*nT + i - 1
            ae!(row, c3b(:pfob, i, r, ss),           1.0)
            ae!(row, col_pm(iN(s.TRAD_COMM[i]), r), -1.0)
        end
    end

    # TEXPRATIO[r]: 100*INCOME*del_taxrexp + TEX*y - Σ_{i,ss} XTAXD*(pfob+qxs) = exog(tx,txs)
    let er = eoffs[:TEXPRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrexp, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),            C.TEX[r])
            for ss in 1:nR, i in 1:nT
                ae!(row, c3b(:pfob, i, r, ss),  -C.XTAXD[i,r,ss])
                ae!(row, c3b(:qxs,  i, r, ss),  -C.XTAXD[i,r,ss])
            end
        end
    end

    # MKTPRICES[i,r,ss]: pms[i,r,ss] - pcif[i,r,ss] = tm + tms  (exog)
    let er = eoffs[:MKTPRICES]
        for ss in 1:nR, r in 1:nR, i in 1:nT
            row = er + (ss-1)*nT*nR + (r-1)*nT + i - 1
            ae!(row, c3b(:pms,  i, r, ss),  1.0)
            ae!(row, c3b(:pcif, i, r, ss), -1.0)
        end
    end

    # DPRICEIMP[i,ss]: pim[i,ss] - Σ_k MSHRS[i,k,ss]*pms[i,k,ss] = exog(ams)
    let er = eoffs[:DPRICEIMP]
        for ss in 1:nR, i in 1:nT
            row = er + (ss-1)*nT + i - 1
            ae!(row, c2(:pim, i, ss, nT), 1.0)
            for k in 1:nR
                ae!(row, c3b(:pms, i, k, ss), -C.MSHRS[i,k,ss])
            end
        end
    end

    # PRICETGT[i,r]: pr[i,r] - pm_t(i,r) + pim[i,r] = 0
    let er = eoffs[:PRICETGT]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:pr,  i, r, nT),             1.0)
            ae!(row, col_pm(iN(s.TRAD_COMM[i]), r), -1.0)
            ae!(row, c2(:pim, i, r, nT),             1.0)
        end
    end

    # IMPORTDEMAND[i,r,ss]: qxs - qim + ESUBM*(pms - pim) = exog(ams)
    let er = eoffs[:IMPORTDEMAND]
        for ss in 1:nR, r in 1:nR, i in 1:nT
            row = er + (ss-1)*nT*nR + (r-1)*nT + i - 1
            σ   = d.ESUBM[i]
            ae!(row, c3b(:qxs, i, r, ss),   1.0)
            ae!(row, c2(:qim, i, ss, nT),   -1.0)
            ae!(row, c3b(:pms, i, r, ss),    σ)
            ae!(row, c2(:pim, i, ss, nT),   -σ)
        end
    end

    # TIMPRATIO[r]: 100*INCOME*del_taxrimp + TIM*y - Σ_{i,ss} MTAX*(pcif+qxs) = exog(tm,tms)
    let er = eoffs[:TIMPRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrimp, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),            C.TIM[r])
            for ss in 1:nR, i in 1:nT
                ae!(row, c3b(:pcif, i, ss, r), -C.MTAX[i,ss,r])
                ae!(row, c3b(:qxs,  i, ss, r), -C.MTAX[i,ss,r])
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 6 – INTERNATIONAL TRANSPORT
    # ═════════════════════════════════════════════════════════════════════════

    # QTRANS_MFSD[m,i,r,ss]: qtmfsd - qxs = -atmfsd (exog)
    let er = eoffs[:QTRANS_MFSD]
        for ss in 1:nR, r in 1:nR, i in 1:nT, m in 1:nM
            row = er + (ss-1)*nM*nT*nR + (r-1)*nM*nT + (i-1)*nM + m - 1
            ae!(row, c4b(:qtmfsd, m, i, r, ss), 1.0)
            ae!(row, c3b(:qxs,   i,    r, ss), -1.0)
        end
    end

    # TRANS_DEMAND[m]: qtm[m] - Σ_{i,r,ss} VTMUSESHR*qtmfsd = 0
    let er = eoffs[:TRANS_DEMAND]
        for m in 1:nM
            row = er + m - 1
            ae!(row, c1(:qtm, m), 1.0)
            for ss in 1:nR, r in 1:nR, i in 1:nT
                ae!(row, c4b(:qtmfsd, m, i, r, ss), -C.VTMUSESHR[m,i,r,ss])
            end
        end
    end

    # PTRANSPORT[m]: pt[m] - Σ_r VTSUPPSHR[m,r]*pm[iN(MARG[m]),r] = 0
    let er = eoffs[:PTRANSPORT]
        for m in 1:nM
            row = er + m - 1
            ni  = iN(s.MARG_COMM[m])
            ae!(row, c1(:pt, m), 1.0)
            for r in 1:nR
                ae!(row, col_pm(ni, r), -C.VTSUPPSHR[m,r])
            end
        end
    end

    # TRANSCOSTINDEX[i,r,ss]: ptrans - Σ_m VTFSD_MSH[m,i,r,ss]*pt[m] = exog(atmfsd)
    let er = eoffs[:TRANSCOSTINDEX]
        for ss in 1:nR, r in 1:nR, i in 1:nT
            row = er + (ss-1)*nT*nR + (r-1)*nT + i - 1
            ae!(row, c3b(:ptrans, i, r, ss), 1.0)
            for m in 1:nM
                ae!(row, c1(:pt, m), -C.VTFSD_MSH[m,i,r,ss])
            end
        end
    end

    # TRANSVCES[m,r]: qst[m,r] - qtm[m] - pt[m] + pm[iN(MARG[m]),r] = 0
    let er = eoffs[:TRANSVCES]
        for r in 1:nR, m in 1:nM
            row = er + (r-1)*nM + m - 1
            ni  = iN(s.MARG_COMM[m])
            ae!(row, c2(:qst, m, r, nM), 1.0)
            ae!(row, c1(:qtm, m),       -1.0)
            ae!(row, c1(:pt,  m),       -1.0)
            ae!(row, col_pm(ni, r),      1.0)
        end
    end

    # FOBCIF[i,r,ss]: pcif - FOBSHR*pfob - TRNSHR*ptrans = 0
    let er = eoffs[:FOBCIF]
        for ss in 1:nR, r in 1:nR, i in 1:nT
            row = er + (ss-1)*nT*nR + (r-1)*nT + i - 1
            ae!(row, c3b(:pcif,   i, r, ss),  1.0)
            ae!(row, c3b(:pfob,   i, r, ss), -C.FOBSHR[i,r,ss])
            ae!(row, c3b(:ptrans, i, r, ss), -C.TRNSHR[i,r,ss])
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 7 – REGIONAL HOUSEHOLD
    # ═════════════════════════════════════════════════════════════════════════

    # FACTORINCPRICES[ei,r]: ps[ni,r] - pm[ni,r] = to (exog)
    let er = eoffs[:FACTORINCPRICES]
        for r in 1:nR, ei in 1:nE
            row = er + (r-1)*nE + ei - 1
            ni  = iN(s.ENDW_COMM[ei])
            ae!(row, col_ps(ni, r),  1.0)
            ae!(row, col_pm(ni, r), -1.0)
        end
    end

    # TINCRATIO[r]: 100*INCOME*del_taxrinc + TINC*y - Σ_{e∈ENDW} PTAX*pm = exog(to,qo_endw)
    let er = eoffs[:TINCRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_taxrinc, r), 100*C.INCOME[r])
            ae!(row, c1(:y, r),            C.TINC[r])
            for e in s.ENDW_COMM
                ni = iN(e)
                ae!(row, col_pm(ni, r), -C.PTAX[ni,r])
            end
        end
    end

    # ENDW_PRICE[si2,r]: pm[ni,r] - Σ_j REVSHR[ei,j,r]*pmes_slug[si2,j,r] = 0
    let er = eoffs[:ENDW_PRICE]
        for r in 1:nR, (si2,e) in enumerate(s.ENDWS_COMM)
            row = er + (r-1)*nES + si2 - 1
            ni  = iN(e); ei = iE(e)
            ae!(row, col_pm(ni, r), 1.0)
            for j in 1:nP
                ae!(row, col_pmes(si2, j, r), -C.REVSHR[ei,j,r])
            end
        end
    end

    # ENDW_SUPPLY[si2,j,r]: qoes_slug - ETRAE*(pm - pmes_slug) = exog(qo_endw,endwslack)
    let er = eoffs[:ENDW_SUPPLY]
        for r in 1:nR, j in 1:nP, (si2,e) in enumerate(s.ENDWS_COMM)
            row = er + (r-1)*nES*nP + (j-1)*nES + si2 - 1
            ni  = iN(e); ei = iE(e)
            σ   = d.ETRAE[ei]
            ae!(row, col_qoes(si2, j, r),  1.0)
            ae!(row, col_pm(ni, r),       -σ)
            ae!(row, col_pmes(si2, j, r),  σ)
        end
    end

    # FACTORINCOME[r]: FY*fincome - Σ_{e∈ENDW} VOM*(pm) + VDEP*(pcgds+kb) = exog(qo_endw)
    let er = eoffs[:FACTORINCOME]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:fincome, r),  C.FY[r])
            for e in s.ENDW_COMM
                ni = iN(e)
                ae!(row, col_pm(ni, r), -C.VOM[ni,r])
            end
            ae!(row, c1(:pcgds, r), d.VDEP[r])
            ae!(row, c1(:kb,    r), d.VDEP[r])
        end
    end

    # DINDTAXRATIO[r]: del_indtaxr - (del_taxrpc + del_taxrgc + del_taxriu + del_taxrfu
    #                                + del_taxrout + del_taxrexp + del_taxrimp) = 0
    let er = eoffs[:DINDTAXRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_indtaxr,  r),  1.0)
            ae!(row, c1(:del_taxrpc,   r), -1.0)
            ae!(row, c1(:del_taxrgc,   r), -1.0)
            ae!(row, c1(:del_taxriu,   r), -1.0)
            ae!(row, c1(:del_taxrfu,   r), -1.0)
            ae!(row, c1(:del_taxrout,  r), -1.0)
            ae!(row, c1(:del_taxrexp,  r), -1.0)
            ae!(row, c1(:del_taxrimp,  r), -1.0)
        end
    end

    # DTAXRATIO[r]: del_ttaxr - (all del_tax* + del_taxrinc) = 0
    let er = eoffs[:DTAXRATIO]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:del_ttaxr,    r),  1.0)
            ae!(row, c1(:del_taxrpc,   r), -1.0)
            ae!(row, c1(:del_taxrgc,   r), -1.0)
            ae!(row, c1(:del_taxriu,   r), -1.0)
            ae!(row, c1(:del_taxrfu,   r), -1.0)
            ae!(row, c1(:del_taxrout,  r), -1.0)
            ae!(row, c1(:del_taxrexp,  r), -1.0)
            ae!(row, c1(:del_taxrimp,  r), -1.0)
            ae!(row, c1(:del_taxrinc,  r), -1.0)
        end
    end

    # REGIONALINCOME[r]: (INCOME - INDTAX)*y - FY*fincome - 100*INCOME*del_indtaxr = incomeslack
    let er = eoffs[:REGIONALINCOME]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:y,           r), C.INCOME[r] - C.INDTAX[r])
            ae!(row, c1(:fincome,     r), -C.FY[r])
            ae!(row, c1(:del_indtaxr, r), -100*C.INCOME[r])
        end
    end

    # DPARAV[r]: dpav - XSHRPRIV*dppriv - XSHRGOV*dpgov - XSHRSAVE*dpsave = 0  (dp* exog)
    let er = eoffs[:DPARAV]
        for r in 1:nR
            ae!(er+r-1, c1(:dpav, r), 1.0)
        end
    end

    # UTILITELASTIC[r]: uelas - XSHRPRIV*uepriv + dpav = 0
    let er = eoffs[:UTILITELASTIC]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:uelas,  r),  1.0)
            ae!(row, c1(:uepriv, r), -C.XSHRPRIV[r])
            ae!(row, c1(:dpav,   r),  1.0)
        end
    end

    # PRIVCONSEXP[r]: yp - y + uepriv - uelas = dppriv (exog)
    let er = eoffs[:PRIVCONSEXP]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:yp,     r),  1.0)
            ae!(row, c1(:y,      r), -1.0)
            ae!(row, c1(:uepriv, r),  1.0)
            ae!(row, c1(:uelas,  r), -1.0)
        end
    end

    # GOVCONSEXP[r]: yg - y - uelas = dpgov (exog)
    let er = eoffs[:GOVCONSEXP]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:yg,    r),  1.0)
            ae!(row, c1(:y,     r), -1.0)
            ae!(row, c1(:uelas, r), -1.0)
        end
    end

    # SAVING[r]: psave + qsave - y - uelas = dpsave (exog)
    let er = eoffs[:SAVING]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:psave, r),  1.0)
            ae!(row, c1(:qsave, r),  1.0)
            ae!(row, c1(:y,     r), -1.0)
            ae!(row, c1(:uelas, r), -1.0)
        end
    end

    # PRICEINDEXREG[r]: p - XSHRPRIV*ppriv - XSHRGOV*pgov - XSHRSAVE*psave = 0
    let er = eoffs[:PRICEINDEXREG]
        for r in 1:nR
            row = er + r - 1
            ae!(row, c1(:p,     r),  1.0)
            ae!(row, c1(:ppriv, r), -C.XSHRPRIV[r])
            ae!(row, c1(:pgov,  r), -C.XSHRGOV[r])
            ae!(row, c1(:psave, r), -C.XSHRSAVE[r])
        end
    end

    # UTILITY[r]: u - (1/UTILELAS)*(y - p) = au + pop/UTILELAS  (au,pop exog)
    let er = eoffs[:UTILITY]
        for r in 1:nR
            row = er + r - 1
            ue  = max(C.UTILELAS[r], 1e-10)
            ae!(row, c1(:u, r),  1.0)
            ae!(row, c1(:y, r), -1/ue)
            ae!(row, c1(:p, r),  1/ue)
        end
    end

    # DISTPARSUM[r]: DPARSUM*dpsum - DPARPRIV*dppriv - DPARGOV*dpgov - DPARSAVE*dpsave = 0 (dp* exog)
    let er = eoffs[:DISTPARSUM]
        for r in 1:nR
            ae!(er+r-1, c1(:dpsum, r), d.DPARSUM[r])
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # MODULE 8 – EQUILIBRIUM
    # ═════════════════════════════════════════════════════════════════════════

    # MKTCLDOM[i,r]: qds - Σ_j SHRDFM*qfd - SHRDPM*qpd - SHRDGM*qgd = 0
    let er = eoffs[:MKTCLDOM]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:qds, i, r, nT), 1.0)
            for j in 1:nP
                ae!(row, c3(:qfd, i, j, r, nT, nP), -C.SHRDFM[i,j,r])
            end
            ae!(row, c2(:qpd, i, r, nT), -C.SHRDPM[i,r])
            ae!(row, c2(:qgd, i, r, nT), -C.SHRDGM[i,r])
        end
    end

    # MKTCLTRD_MARG[mi,r]: qo[ni,r] - SHRDM*qds - SHRST*qst - Σ_ss SHRXMD*qxs = 0
    let er = eoffs[:MKTCLTRD_MARG]
        nMARG = length(s.MARG_COMM)
        for r in 1:nR, (mi,mc) in enumerate(s.MARG_COMM)
            row = er + (r-1)*nMARG + mi - 1
            ti  = iT(mc); ni = iN(mc)
            ae!(row, col_qo(ni, r),         1.0)
            ae!(row, c2(:qds, ti, r, nT),  -C.SHRDM[ti,r])
            ae!(row, c2(:qst, mi, r, nM),  -C.SHRST[mi,r])
            for ss in 1:nR
                ae!(row, c3b(:qxs, ti, r, ss), -C.SHRXMD[ti,r,ss])
            end
        end
    end

    # MKTCLTRD_NMRG[ni2,r]: qo[ni,r] - SHRDM*qds - Σ_ss SHRXMD*qxs = 0
    let er = eoffs[:MKTCLTRD_NMRG]
        nNMRG = length(s.NMRG_COMM)
        for r in 1:nR, (ni2,nc) in enumerate(s.NMRG_COMM)
            row = er + (r-1)*nNMRG + ni2 - 1
            ti  = iT(nc); ni = iN(nc)
            ae!(row, col_qo(ni, r),        1.0)
            ae!(row, c2(:qds, ti, r, nT), -C.SHRDM[ti,r])
            for ss in 1:nR
                ae!(row, c3b(:qxs, ti, r, ss), -C.SHRXMD[ti,r,ss])
            end
        end
    end

    # MKTCLIMP[i,r]: qim - Σ_j SHRIFM*qfm - SHRIPM*qpm - SHRIGM*qgm = 0
    let er = eoffs[:MKTCLIMP]
        for r in 1:nR, i in 1:nT
            row = er + (r-1)*nT + i - 1
            ae!(row, c2(:qim, i, r, nT), 1.0)
            for j in 1:nP
                ae!(row, c3(:qfm, i, j, r, nT, nP), -C.SHRIFM[i,j,r])
            end
            ae!(row, c2(:qpm, i, r, nT), -C.SHRIPM[i,r])
            ae!(row, c2(:qgm, i, r, nT), -C.SHRIGM[i,r])
        end
    end

    # MKTCLENDWM[mi2,r]: -Σ_j SHREM*qfe = exog(qo_endw,endwslack)
    let er = eoffs[:MKTCLENDWM]
        for r in 1:nR, (mi2,e) in enumerate(s.ENDWM_COMM)
            row = er + (r-1)*nEM + mi2 - 1
            ei  = iE(e)
            for j in 1:nP
                ae!(row, col_qfe(ei, j, r), -C.SHREM[mi2,j,r])
            end
        end
    end

    # MKTCLENDWS[si2,j,r]: qoes_slug[si2,j,r] - qfe[ei,j,r] = 0
    let er = eoffs[:MKTCLENDWS]
        for r in 1:nR, j in 1:nP, (si2,e) in enumerate(s.ENDWS_COMM)
            row = er + (r-1)*nES*nP + (j-1)*nES + si2 - 1
            ei  = iE(e)
            ae!(row, col_qoes(si2, j, r),  1.0)
            ae!(row, col_qfe(ei, j, r),   -1.0)
        end
    end

    # WALRAS_S (scalar): walras_sup - pcgdswld - globalcgds = 0
    let row = eoffs[:WALRAS_S]
        ae!(row, voffs[:walras_sup],   1.0)
        ae!(row, voffs[:pcgdswld],    -1.0)
        ae!(row, voffs[:globalcgds],  -1.0)
    end

    # WALRAS_D (scalar): GLOBINV*walras_dem - Σ_r SAVE[r]*(psave+qsave) = 0
    let row = eoffs[:WALRAS_D]
        ae!(row, voffs[:walras_dem], C.GLOBINV)
        for r in 1:nR
            ae!(row, c1(:psave, r), -d.SAVE[r])
            ae!(row, c1(:qsave, r), -d.SAVE[r])
        end
    end

    # WALRAS (scalar): walras_sup - walras_dem = walraslack (exog)
    let row = eoffs[:WALRAS]
        ae!(row, voffs[:walras_sup],  1.0)
        ae!(row, voffs[:walras_dem], -1.0)
    end

    # ── Assemble sparse matrix ────────────────────────────────────────────────
    nnz = ptr[]
    return sparse(Int.(II[1:nnz]), Int.(JJ[1:nnz]), VV[1:nnz], n_eq, n_var)
end
