# gtap_jacobian_analytical_v7.jl
# Direct analytical Jacobian for the GTAPv7 linearised system.
#
# Model is exactly linear in endogenous variables, so A[i,j] = coefficient
# of x_endo[j] in residual equation i.  No automatic differentiation needed.
# Build time: < 1 second vs ~75 minutes for colored finite differences.
#
# Usage:
#   A = build_A_v7_analytical(d, s, C)   # drop-in replacement for build_A_v7

using SparseArrays

"""
    build_A_v7_analytical(d, s, C) -> SparseMatrixCSC{Float64}

Construct the GTAPv7 Jacobian matrix analytically (no AD / finite differences).
"""
function build_A_v7_analytical(d::GTAPDataV7, s::GTAPSetsV7, C)
    nC=length(s.COMM); nA=length(s.ACTS); nR=length(s.REG)
    nE=length(s.ENDW); nM=length(s.MARG)
    nEM=length(s.ENDWM); nES=length(s.ENDWS); nEF=length(s.ENDWF)
    nEMS=length(s.ENDWMS); nEC=length(s.ENDWC)

    iC=C.iC; iA=C.iA; iE=C.iE; iM=C.iM; iR=C.iR
    iEM=C.iEM; iES=C.iES; iEMS=C.iEMS; iEC=C.iEC

    voff  = endo_offsets_v7(s)
    eoff  = eq_row_offsets_v7(s)
    n_col = n_endog_v7(s)
    sz    = core_eq_sizes_v7(s)
    n_row = sum(prod(sz[nm]) for nm in CORE_EQ_NAMES_V7)
    n     = n_col   # column count alias

    # ── Column helpers: variable nm at local linear index idx (1-based) ──────
    @inline cv(nm::Symbol, idx::Int) = voff[nm] + idx - 1

    # Linear index helpers (col-major, first dim varies fastest)
    @inline ar(a,r)           = a + (r-1)*nA
    @inline car(c,a,r)        = c + (a-1)*nC + (r-1)*nC*nA
    @inline ear(e,a,r)        = e + (a-1)*nE + (r-1)*nE*nA
    @inline cr(c,r)           = c + (r-1)*nC
    @inline crr(c,s,d)        = c + (s-1)*nC + (d-1)*nC*nR
    @inline mcrr(m,c,s,d)     = m + (c-1)*nM + (s-1)*nM*nC + (d-1)*nM*nC*nR
    @inline mr(m,r)           = m + (r-1)*nM
    @inline emsr(e,r)         = e + (r-1)*nEMS
    @inline emr(e,r)          = e + (r-1)*nEM
    @inline esar(e,a,r)       = e + (a-1)*nES + (r-1)*nES*nA
    @inline efar(e,a,r)       = e + (a-1)*nEF + (r-1)*nEF*nA
    @inline ecr(k,r)          = k + (r-1)*nEC

    # ── Row helpers: equation nm at local index idx ──────────────────────────
    @inline er(nm::Symbol, idx::Int) = eoff[nm] + idx - 1

    # ── Triplet accumulator ───────────────────────────────────────────────────
    cap = 600_000
    II  = Vector{Int32}(undef, cap)
    JJ  = Vector{Int32}(undef, cap)
    VV  = Vector{Float64}(undef, cap)
    ptr = Ref(0)

    @inline function ae!(row::Int, nm::Symbol, idx::Int, val::Real)
        iszero(val) && return
        k      = ptr[] + 1
        II[k]  = row
        JJ[k]  = voff[nm] + idx - 1
        VV[k]  = Float64(val)
        ptr[]  = k
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 – FIRMS
    # ══════════════════════════════════════════════════════════════════════════

    # E_ao (nA,nR): ao - aosec - aoreg - aoall = 0  (aosec,aoreg,aoall exog)
    for a in 1:nA, r in 1:nR
        ae!(er(:E_ao, ar(a,r)), :ao, ar(a,r), 1.0)
    end

    # E_ava (nA,nR): ava - avasec - avareg - avaall = 0
    for a in 1:nA, r in 1:nR
        ae!(er(:E_ava, ar(a,r)), :ava, ar(a,r), 1.0)
    end

    # E_afa (nC,nA,nR): afa - afcom - afsec - afreg - afall = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        ae!(er(:E_afa, car(c,a,r)), :afa, car(c,a,r), 1.0)
    end

    # E_afe (nE,nA,nR): afe - afecom - afesec - afereg - afeall = 0
    for e in 1:nE, a in 1:nA, r in 1:nR
        ae!(er(:E_afe, ear(e,a,r)), :afe, ear(e,a,r), 1.0)
    end

    # E_aint (nA,nR): aint - aintsec - aintreg - aintall = 0
    for a in 1:nA, r in 1:nR
        ae!(er(:E_aint, ar(a,r)), :aint, ar(a,r), 1.0)
    end

    # E_qint (nA,nR): qint + aint - qo + ao + ESUBT*(pint - aint - po - ao) = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_qint, ar(a,r))
        e   = d.ESUBT[a,r]
        ae!(row, :qint,  ar(a,r),  1.0)
        ae!(row, :aint,  ar(a,r),  1.0 - e)
        ae!(row, :qo,    ar(a,r), -1.0)
        ae!(row, :ao,    ar(a,r),  1.0 - e)
        ae!(row, :pint,  ar(a,r),  e)
        ae!(row, :po,    ar(a,r), -e)
    end

    # E_qva (nA,nR): qva + ava - qo + ao + ESUBT*(pva - ava - po - ao) = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_qva, ar(a,r))
        e   = d.ESUBT[a,r]
        ae!(row, :qva,  ar(a,r),  1.0)
        ae!(row, :ava,  ar(a,r),  1.0 - e)
        ae!(row, :qo,   ar(a,r), -1.0)
        ae!(row, :ao,   ar(a,r),  1.0 - e)
        ae!(row, :pva,  ar(a,r),  e)
        ae!(row, :po,   ar(a,r), -e)
    end

    # E_qfa (nC,nA,nR): qfa + afa - qint + ESUBC*(pfa - afa - pint) = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_qfa, car(c,a,r))
        e   = d.ESUBC[a,r]
        ae!(row, :qfa,  car(c,a,r),  1.0)
        ae!(row, :afa,  car(c,a,r),  1.0 - e)
        ae!(row, :qint, ar(a,r),    -1.0)
        ae!(row, :pfa,  car(c,a,r),  e)
        ae!(row, :pint, ar(a,r),    -e)
    end

    # E_pint (nA,nR): pint - Σ_c INTSHR*(pfa - afa) = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_pint, ar(a,r))
        ae!(row, :pint, ar(a,r), 1.0)
        for c in 1:nC
            shr = C.INTSHR[c,a,r]
            ae!(row, :pfa, car(c,a,r), -shr)
            ae!(row, :afa, car(c,a,r),  shr)
        end
    end

    # E_qfe (nE,nA,nR): qfe + afe - qva + ESUBVA*(pfe - afe - pva) = 0
    for e in 1:nE, a in 1:nA, r in 1:nR
        row = er(:E_qfe, ear(e,a,r))
        ev  = d.ESUBVA[a,r]
        ae!(row, :qfe, ear(e,a,r),  1.0)
        ae!(row, :afe, ear(e,a,r),  1.0 - ev)
        ae!(row, :qva, ar(a,r),    -1.0)
        ae!(row, :pfe, ear(e,a,r),  ev)
        ae!(row, :pva, ar(a,r),    -ev)
    end

    # E_pva (nA,nR): pva - Σ_e SVA*(pfe - afe) = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_pva, ar(a,r))
        ae!(row, :pva, ar(a,r), 1.0)
        for e in 1:nE
            s2 = C.SVA[e,a,r]
            ae!(row, :pfe, ear(e,a,r), -s2)
            ae!(row, :afe, ear(e,a,r),  s2)
        end
    end

    # E_qfd (nC,nA,nR): qfd - qfa + ESUBD*(pfd - pfa) = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_qfd, car(c,a,r))
        e   = d.ESUBD[c]
        ae!(row, :qfd, car(c,a,r),  1.0)
        ae!(row, :qfa, car(c,a,r), -1.0)
        ae!(row, :pfd, car(c,a,r),  e)
        ae!(row, :pfa, car(c,a,r), -e)
    end

    # E_qfm (nC,nA,nR): qfm - qfa + ESUBD*(pfm - pfa) = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_qfm, car(c,a,r))
        e   = d.ESUBD[c]
        ae!(row, :qfm, car(c,a,r),  1.0)
        ae!(row, :qfa, car(c,a,r), -1.0)
        ae!(row, :pfm, car(c,a,r),  e)
        ae!(row, :pfa, car(c,a,r), -e)
    end

    # E_pfa (nC,nA,nR): pfa - (1-FMSHR)*pfd - FMSHR*pfm = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_pfa, car(c,a,r))
        fm  = C.FMSHR[c,a,r]
        ae!(row, :pfa, car(c,a,r),  1.0)
        ae!(row, :pfd, car(c,a,r), -(1.0 - fm))
        ae!(row, :pfm, car(c,a,r), -fm)
    end

    # E_pfd (nC,nA,nR): pfd - pds[c,r] - tfd = 0  (tfd exog)
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_pfd, car(c,a,r))
        ae!(row, :pfd, car(c,a,r),  1.0)
        ae!(row, :pds, cr(c,r),    -1.0)
    end

    # E_pfm (nC,nA,nR): pfm - pms[c,r] - tfm = 0  (tfm exog)
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_pfm, car(c,a,r))
        ae!(row, :pfm, car(c,a,r),  1.0)
        ae!(row, :pms, cr(c,r),    -1.0)
    end

    # E_qo_zero (nA,nR): zero-profit condition
    # po + ao - Σ_e STC_endw*(pfe-afe-ava) - Σ_c STC_comm*(pfa-afa-aint) - profitslack = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_qo_zero, ar(a,r))
        ae!(row, :po, ar(a,r), 1.0)
        ae!(row, :ao, ar(a,r), 1.0)
        sum_stc_e = 0.0
        for e in 1:nE
            stc = C.STC_endw[e,a,r]
            ae!(row, :pfe, ear(e,a,r), -stc)
            ae!(row, :afe, ear(e,a,r),  stc)
            sum_stc_e += stc
        end
        ae!(row, :ava, ar(a,r), sum_stc_e)
        sum_stc_c = 0.0
        for c in 1:nC
            stc = C.STC_comm[c,a,r]
            ae!(row, :pfa, car(c,a,r), -stc)
            ae!(row, :afa, car(c,a,r),  stc)
            sum_stc_c += stc
        end
        ae!(row, :aint, ar(a,r), sum_stc_c)
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 – COMMODITY SUPPLY
    # ══════════════════════════════════════════════════════════════════════════

    # E_qca (nC,nA,nR): if MAKES>0: qca - qo + ETRAQ*(ps - po) = 0; else: qca = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_qca, car(c,a,r))
        if d.MAKES[c,a,r] > 0
            et = d.ETRAQ[a,r]
            ae!(row, :qca, car(c,a,r),  1.0)
            ae!(row, :qo,  ar(a,r),    -1.0)
            ae!(row, :ps,  car(c,a,r),  et)
            ae!(row, :po,  ar(a,r),    -et)
        else
            ae!(row, :qca, car(c,a,r), 1.0)
        end
    end

    # E_po (nA,nR): po - Σ_c MAKESACTSHR*ps = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_po, ar(a,r))
        ae!(row, :po, ar(a,r), 1.0)
        for c in 1:nC
            ae!(row, :ps, car(c,a,r), -C.MAKESACTSHR[c,a,r])
        end
    end

    # E_ps (nC,nA,nR): pca - ps - to = 0  (to exog)
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_ps, car(c,a,r))
        ae!(row, :pca, car(c,a,r),  1.0)
        ae!(row, :ps,  car(c,a,r), -1.0)
    end

    # E_pb (nA,nR): pb - Σ_c MAKEBACTSHR*pca = 0
    for a in 1:nA, r in 1:nR
        row = er(:E_pb, ar(a,r))
        ae!(row, :pb, ar(a,r), 1.0)
        for c in 1:nC
            ae!(row, :pca, car(c,a,r), -C.MAKEBACTSHR[c,a,r])
        end
    end

    # E_pca (nC,nA,nR): if MAKEB>0: pca - pds + ESUBQ*(qca-qc) = 0; else: pca - pds = 0
    for c in 1:nC, a in 1:nA, r in 1:nR
        row = er(:E_pca, car(c,a,r))
        ae!(row, :pca, car(c,a,r),  1.0)
        ae!(row, :pds, cr(c,r),    -1.0)
        if d.MAKEB[c,a,r] > 0
            eq = d.ESUBQ[c,r]
            ae!(row, :qca, car(c,a,r),  eq)
            ae!(row, :qc,  cr(c,r),    -eq)
        end
    end

    # E_qc (nC,nR): qc - Σ_a MAKEBCOMSHR*qca = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qc, cr(c,r))
        ae!(row, :qc, cr(c,r), 1.0)
        for a in 1:nA
            ae!(row, :qca, car(c,a,r), -C.MAKEBCOMSHR[c,a,r])
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 3 – INCOME
    # ══════════════════════════════════════════════════════════════════════════

    # E_fincome (nR): FY*fincome - Σ_{e,a} EVFB*(peb+qes) + VDEP*(pinv+kb) = 0
    for r in 1:nR
        row = er(:E_fincome, r)
        ae!(row, :fincome, r,  C.FY[r])
        ae!(row, :pinv,    r,  d.VDEP[r])
        ae!(row, :kb,      r,  d.VDEP[r])
        for e in 1:nE, a in 1:nA
            evfb = d.EVFB[e,a,r]
            ae!(row, :peb, ear(e,a,r), -evfb)
            ae!(row, :qes, ear(e,a,r), -evfb)
        end
    end

    # E_y (nR): FY*y - FY*fincome - 100*INCOME*del_indtaxr - INCOME*incomeslack = 0
    # (incomeslack exog)
    for r in 1:nR
        row = er(:E_y, r)
        ae!(row, :y,           r,  C.FY[r])
        ae!(row, :fincome,     r, -C.FY[r])
        ae!(row, :del_indtaxr, r, -100.0 * C.INCOME[r])
    end

    # E_del_indtaxr (nR): 100*INCOME*del_indtaxr + TAXR_IND*y - RHS = 0
    # RHS = taxrout + taxrfu + taxriu_d + taxriu_m + taxrpc + taxrgc + taxric + taxrimp + taxrexp
    for r in 1:nR
        row = er(:E_del_indtaxr, r)
        ae!(row, :del_indtaxr, r,  100.0 * C.INCOME[r])
        ae!(row, :y,           r,  C.TAXR_IND[r])
        # taxrout: PTAX*(ps + qca)  (to is exog)
        for c in 1:nC, a in 1:nA
            ptax = C.PTAX[c,a,r]
            ae!(row, :ps,  car(c,a,r), -ptax)
            ae!(row, :qca, car(c,a,r), -ptax)
        end
        # taxrfu: ETAX*(peb + qfe)  (tfe is exog)
        for e in 1:nE, a in 1:nA
            etax = C.ETAX[e,a,r]
            ae!(row, :peb, ear(e,a,r), -etax)
            ae!(row, :qfe, ear(e,a,r), -etax)
        end
        # taxriu_d: DFTAX*(pds + qfd)  (tfd exog)
        for c in 1:nC
            pds_coef = 0.0
            for a in 1:nA
                dft = C.DFTAX[c,a,r]
                ae!(row, :qfd, car(c,a,r), -dft)
                pds_coef += dft
            end
            ae!(row, :pds, cr(c,r), -pds_coef)
        end
        # taxriu_m: MFTAX*(pms + qfm)  (tfm exog)
        for c in 1:nC
            pms_coef = 0.0
            for a in 1:nA
                mft = C.MFTAX[c,a,r]
                ae!(row, :qfm, car(c,a,r), -mft)
                pms_coef += mft
            end
            ae!(row, :pms, cr(c,r), -pms_coef)
        end
        # taxrpc: VDPP*tpd + DPTAX*(pds+qpd) + VMPP*tpm + MPTAX*(pms+qpm)
        for c in 1:nC
            ae!(row, :tpd, cr(c,r), -d.VDPP[c,r])
            ae!(row, :pds, cr(c,r), -C.DPTAX[c,r])   # accumulates into pds
            ae!(row, :qpd, cr(c,r), -C.DPTAX[c,r])
            ae!(row, :tpm, cr(c,r), -d.VMPP[c,r])
            ae!(row, :pms, cr(c,r), -C.MPTAX[c,r])   # accumulates into pms
            ae!(row, :qpm, cr(c,r), -C.MPTAX[c,r])
        end
        # taxrgc: DGTAX*(pds+qgd) + MGTAX*(pms+qgm)  (tgd,tgm exog)
        for c in 1:nC
            ae!(row, :pds, cr(c,r), -C.DGTAX[c,r])
            ae!(row, :qgd, cr(c,r), -C.DGTAX[c,r])
            ae!(row, :pms, cr(c,r), -C.MGTAX[c,r])
            ae!(row, :qgm, cr(c,r), -C.MGTAX[c,r])
        end
        # taxric: DITAX*(pds+qid) + MITAX*(pms+qim)  (tid,tim exog)
        for c in 1:nC
            ae!(row, :pds, cr(c,r), -C.DITAX[c,r])
            ae!(row, :qid, cr(c,r), -C.DITAX[c,r])
            ae!(row, :pms, cr(c,r), -C.MITAX[c,r])
            ae!(row, :qim, cr(c,r), -C.MITAX[c,r])
        end
        # taxrimp: MTAX*(pcif + qxs)  (tm,tms exog)  [imports INTO r from ss]
        for c in 1:nC, ss in 1:nR
            mtax = C.MTAX[c,ss,r]
            ae!(row, :pcif, crr(c,ss,r), -mtax)
            ae!(row, :qxs,  crr(c,ss,r), -mtax)
        end
        # taxrexp: XTAXD*(pds + qxs)  (tx,txs exog)  [exports FROM r to dd]
        for c in 1:nC
            xtaxd_sum = 0.0
            for dd in 1:nR
                xtaxd = C.XTAXD[c,r,dd]
                ae!(row, :qxs, crr(c,r,dd), -xtaxd)
                xtaxd_sum += xtaxd
            end
            ae!(row, :pds, cr(c,r), -xtaxd_sum)
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 4 – INCOME ALLOCATION
    # ══════════════════════════════════════════════════════════════════════════

    # E_qsave (nR): psave + qsave - y - uelas - dpsave = 0  (dpsave exog)
    for r in 1:nR
        row = er(:E_qsave, r)
        ae!(row, :psave, r,  1.0)
        ae!(row, :qsave, r,  1.0)
        ae!(row, :y,     r, -1.0)
        ae!(row, :uelas, r, -1.0)
    end

    # E_yg (nR): yg - y - uelas - dpgov = 0  (dpgov exog)
    for r in 1:nR
        row = er(:E_yg, r)
        ae!(row, :yg,    r,  1.0)
        ae!(row, :y,     r, -1.0)
        ae!(row, :uelas, r, -1.0)
    end

    # E_yp (nR): yp - y + (uepriv - uelas) - dppriv = 0  (dppriv exog)
    for r in 1:nR
        row = er(:E_yp, r)
        ae!(row, :yp,     r,  1.0)
        ae!(row, :y,      r, -1.0)
        ae!(row, :uepriv, r,  1.0)
        ae!(row, :uelas,  r, -1.0)
    end

    # E_uelas (nR): uelas - XSHRPRIV*uepriv + dpav = 0
    for r in 1:nR
        row = er(:E_uelas, r)
        ae!(row, :uelas,  r,  1.0)
        ae!(row, :uepriv, r, -C.XSHRPRIV[r])
        ae!(row, :dpav,   r,  1.0)
    end

    # E_dpav (nR): dpav - XSHRPRIV*dppriv - XSHRGOV*dpgov - XSHRSAVE*dpsave = 0
    # (dppriv, dpgov, dpsave exog)
    for r in 1:nR
        ae!(er(:E_dpav, r), :dpav, r, 1.0)
    end

    # E_p (nR): p - XSHRPRIV*ppriv - XSHRGOV*pgov - XSHRSAVE*psave = 0
    for r in 1:nR
        row = er(:E_p, r)
        ae!(row, :p,     r,  1.0)
        ae!(row, :ppriv, r, -C.XSHRPRIV[r])
        ae!(row, :pgov,  r, -C.XSHRGOV[r])
        ae!(row, :psave, r, -C.XSHRSAVE[r])
    end

    # E_u (nR): u - au - (1/UTILELAS)*(y - pop - p) = 0  (au, pop exog)
    for r in 1:nR
        row = er(:E_u, r)
        ue  = 1.0 / max(C.UTILELAS[r], 1e-10)
        ae!(row, :u, r,  1.0)
        ae!(row, :y, r, -ue)
        ae!(row, :p, r,  ue)
    end

    # E_dpsum (nR): DPARSUM*dpsum - DPARPRIV*dppriv - DPARGOV*dpgov - DPARSAVE*dpsave = 0
    # (dppriv, dpgov, dpsave exog)
    for r in 1:nR
        ae!(er(:E_dpsum, r), :dpsum, r, d.DPARSUM[r])
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 5 – DOMESTIC FINAL DEMAND
    # ══════════════════════════════════════════════════════════════════════════

    # E_qpa (nC,nR): qpa - pop - Σ_k EP[c,k]*ppa[k] - EY[c]*(yp-pop) = 0  (pop exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_qpa, cr(c,r))
        ae!(row, :qpa, cr(c,r),  1.0)
        ae!(row, :yp,  r,        -C.EY[c,r])
        for k in 1:nC
            ae!(row, :ppa, cr(k,r), -C.EP[c,k,r])
        end
    end

    # E_uepriv (nR): uepriv - Σ_c XWCONSHR*(ppa + qpa - yp) = 0
    for r in 1:nR
        row = er(:E_uepriv, r)
        ae!(row, :uepriv, r, 1.0)
        yp_coef = 0.0
        for c in 1:nC
            xw = C.XWCONSHR[c,r]
            ae!(row, :ppa, cr(c,r), -xw)
            ae!(row, :qpa, cr(c,r), -xw)
            yp_coef += xw
        end
        ae!(row, :yp, r, yp_coef)
    end

    # E_ppriv (nR): ppriv - Σ_c CONSHR*ppa = 0
    for r in 1:nR
        row = er(:E_ppriv, r)
        ae!(row, :ppriv, r, 1.0)
        for c in 1:nC
            ae!(row, :ppa, cr(c,r), -C.CONSHR[c,r])
        end
    end

    # E_up (nR): UELASPRIV*up - yp + ppriv + pop = 0  (pop exog)
    for r in 1:nR
        row = er(:E_up, r)
        ae!(row, :up,    r,  C.UELASPRIV[r])
        ae!(row, :yp,    r, -1.0)
        ae!(row, :ppriv, r,  1.0)
    end

    # E_qpd (nC,nR): qpd - qpa + ESUBD*(ppd - ppa) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qpd, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qpd, cr(c,r),  1.0)
        ae!(row, :qpa, cr(c,r), -1.0)
        ae!(row, :ppd, cr(c,r),  e)
        ae!(row, :ppa, cr(c,r), -e)
    end

    # E_qpm (nC,nR): qpm - qpa + ESUBD*(ppm - ppa) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qpm, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qpm, cr(c,r),  1.0)
        ae!(row, :qpa, cr(c,r), -1.0)
        ae!(row, :ppm, cr(c,r),  e)
        ae!(row, :ppa, cr(c,r), -e)
    end

    # E_ppa (nC,nR): ppa - (1-PMSHR)*ppd - PMSHR*ppm = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_ppa, cr(c,r))
        pm  = C.PMSHR[c,r]
        ae!(row, :ppa, cr(c,r),  1.0)
        ae!(row, :ppd, cr(c,r), -(1.0 - pm))
        ae!(row, :ppm, cr(c,r), -pm)
    end

    # E_ppd (nC,nR): ppd - pds - tpd = 0  (tpd endogenous here)
    for c in 1:nC, r in 1:nR
        row = er(:E_ppd, cr(c,r))
        ae!(row, :ppd, cr(c,r),  1.0)
        ae!(row, :pds, cr(c,r), -1.0)
        ae!(row, :tpd, cr(c,r), -1.0)
    end

    # E_ppm (nC,nR): ppm - pms - tpm = 0  (tpm endogenous)
    for c in 1:nC, r in 1:nR
        row = er(:E_ppm, cr(c,r))
        ae!(row, :ppm, cr(c,r),  1.0)
        ae!(row, :pms, cr(c,r), -1.0)
        ae!(row, :tpm, cr(c,r), -1.0)
    end

    # E_tpd (nC,nR): tpd - tpdall - tpreg = 0  (tpdall, tpreg exog)
    for c in 1:nC, r in 1:nR
        ae!(er(:E_tpd, cr(c,r)), :tpd, cr(c,r), 1.0)
    end

    # E_tpm (nC,nR): tpm - tpmall - tpreg = 0  (tpmall, tpreg exog)
    for c in 1:nC, r in 1:nR
        ae!(er(:E_tpm, cr(c,r)), :tpm, cr(c,r), 1.0)
    end

    # E_qga (nC,nR): qga - yg + pgov + ESUBG*(pga - pgov) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qga, cr(c,r))
        eg  = d.ESUBG[r]
        ae!(row, :qga,  cr(c,r),  1.0)
        ae!(row, :yg,   r,        -1.0)
        ae!(row, :pgov, r,         1.0 - eg)
        ae!(row, :pga,  cr(c,r),  eg)
    end

    # E_pgov (nR): pgov - Σ_c GOVSHR*pga = 0
    for r in 1:nR
        row = er(:E_pgov, r)
        ae!(row, :pgov, r, 1.0)
        for c in 1:nC
            ae!(row, :pga, cr(c,r), -C.GOVSHR[c,r])
        end
    end

    # E_ug (nR): ug - yg + pgov + pop = 0  (pop exog)
    for r in 1:nR
        row = er(:E_ug, r)
        ae!(row, :ug,   r,  1.0)
        ae!(row, :yg,   r, -1.0)
        ae!(row, :pgov, r,  1.0)
    end

    # E_qgd (nC,nR): qgd - qga + ESUBD*(pgd - pga) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qgd, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qgd, cr(c,r),  1.0)
        ae!(row, :qga, cr(c,r), -1.0)
        ae!(row, :pgd, cr(c,r),  e)
        ae!(row, :pga, cr(c,r), -e)
    end

    # E_qgm (nC,nR): qgm - qga + ESUBD*(pgm - pga) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qgm, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qgm, cr(c,r),  1.0)
        ae!(row, :qga, cr(c,r), -1.0)
        ae!(row, :pgm, cr(c,r),  e)
        ae!(row, :pga, cr(c,r), -e)
    end

    # E_pga (nC,nR): pga - (1-GMSHR)*pgd - GMSHR*pgm = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_pga, cr(c,r))
        gm  = C.GMSHR[c,r]
        ae!(row, :pga, cr(c,r),  1.0)
        ae!(row, :pgd, cr(c,r), -(1.0 - gm))
        ae!(row, :pgm, cr(c,r), -gm)
    end

    # E_pgd (nC,nR): pgd - pds - tgd = 0  (tgd exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_pgd, cr(c,r))
        ae!(row, :pgd, cr(c,r),  1.0)
        ae!(row, :pds, cr(c,r), -1.0)
    end

    # E_pgm (nC,nR): pgm - pms - tgm = 0  (tgm exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_pgm, cr(c,r))
        ae!(row, :pgm, cr(c,r),  1.0)
        ae!(row, :pms, cr(c,r), -1.0)
    end

    # E_qia (nC,nR): qia - qinv = 0  (Leontief investment)
    for c in 1:nC, r in 1:nR
        row = er(:E_qia, cr(c,r))
        ae!(row, :qia,  cr(c,r),  1.0)
        ae!(row, :qinv, r,        -1.0)
    end

    # E_pinv (nR): pinv - Σ_c INVSHR*pia = 0
    for r in 1:nR
        row = er(:E_pinv, r)
        ae!(row, :pinv, r, 1.0)
        for c in 1:nC
            ae!(row, :pia, cr(c,r), -C.INVSHR[c,r])
        end
    end

    # E_qid (nC,nR): qid - qia + ESUBD*(pid - pia) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qid, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qid, cr(c,r),  1.0)
        ae!(row, :qia, cr(c,r), -1.0)
        ae!(row, :pid, cr(c,r),  e)
        ae!(row, :pia, cr(c,r), -e)
    end

    # E_qim (nC,nR): qim - qia + ESUBD*(pim - pia) = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qim, cr(c,r))
        e   = d.ESUBD[c]
        ae!(row, :qim, cr(c,r),  1.0)
        ae!(row, :qia, cr(c,r), -1.0)
        ae!(row, :pim, cr(c,r),  e)
        ae!(row, :pia, cr(c,r), -e)
    end

    # E_pia (nC,nR): pia - (1-IMSHR)*pid - IMSHR*pim = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_pia, cr(c,r))
        im_ = C.IMSHR[c,r]
        ae!(row, :pia, cr(c,r),  1.0)
        ae!(row, :pid, cr(c,r), -(1.0 - im_))
        ae!(row, :pim, cr(c,r), -im_)
    end

    # E_pid (nC,nR): pid - pds - tid = 0  (tid exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_pid, cr(c,r))
        ae!(row, :pid, cr(c,r),  1.0)
        ae!(row, :pds, cr(c,r), -1.0)
    end

    # E_pim (nC,nR): pim - pms - tim = 0  (tim exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_pim, cr(c,r))
        ae!(row, :pim, cr(c,r),  1.0)
        ae!(row, :pms, cr(c,r), -1.0)
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 6 – TRADE
    # ══════════════════════════════════════════════════════════════════════════

    # E_qms (nC,nR): qms - Σ_a FMCSHR*qfm - PMCSHR*qpm - GMCSHR*qgm - IMCSHR*qim = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qms, cr(c,r))
        ae!(row, :qms, cr(c,r),  1.0)
        for a in 1:nA
            ae!(row, :qfm, car(c,a,r), -C.FMCSHR[c,a,r])
        end
        ae!(row, :qpm, cr(c,r), -C.PMCSHR[c,r])
        ae!(row, :qgm, cr(c,r), -C.GMCSHR[c,r])
        ae!(row, :qim, cr(c,r), -C.IMCSHR[c,r])
    end

    # E_pms (nC,nR): pms - Σ_ss MSHRS*(pmds - ams) = 0  (ams exog)
    for c in 1:nC, dd in 1:nR
        row = er(:E_pms, cr(c,dd))
        ae!(row, :pms, cr(c,dd), 1.0)
        for ss in 1:nR
            ae!(row, :pmds, crr(c,ss,dd), -C.MSHRS[c,ss,dd])
        end
    end

    # E_qxs (nC,nR,nR): qxs - ams - qms[dd] + ESUBM*(pmds - ams - pms[dd]) = 0  (ams exog)
    for c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_qxs, crr(c,ss,dd))
        e   = d.ESUBM[c]
        ae!(row, :qxs,  crr(c,ss,dd),  1.0)
        ae!(row, :qms,  cr(c,dd),      -1.0)
        ae!(row, :pmds, crr(c,ss,dd),  e)
        ae!(row, :pms,  cr(c,dd),      -e)
    end

    # E_pfob (nC,nR,nR): pfob - pds[c,ss] + tx + txs = 0  (tx,txs exog)
    for c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_pfob, crr(c,ss,dd))
        ae!(row, :pfob, crr(c,ss,dd),  1.0)
        ae!(row, :pds,  cr(c,ss),      -1.0)
    end

    # E_pcif (nC,nR,nR): pcif - FOBSHR*pfob - TRNSHR*ptrans = 0
    for c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_pcif, crr(c,ss,dd))
        ae!(row, :pcif,   crr(c,ss,dd),  1.0)
        ae!(row, :pfob,   crr(c,ss,dd), -C.FOBSHR[c,ss,dd])
        ae!(row, :ptrans, crr(c,ss,dd), -C.TRNSHR[c,ss,dd])
    end

    # E_pmds (nC,nR,nR): pmds - pcif - tm - tms = 0  (tm,tms exog)
    for c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_pmds, crr(c,ss,dd))
        ae!(row, :pmds, crr(c,ss,dd),  1.0)
        ae!(row, :pcif, crr(c,ss,dd), -1.0)
    end

    # E_ptrans (nC,nR,nR): ptrans - Σ_m VTFSD_MSH*(pt - atmfsd) = 0
    for c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_ptrans, crr(c,ss,dd))
        ae!(row, :ptrans, crr(c,ss,dd), 1.0)
        for m in 1:nM
            vtf = C.VTFSD_MSH[m,c,ss,dd]
            ae!(row, :pt,      m,              -vtf)
            ae!(row, :atmfsd,  mcrr(m,c,ss,dd), vtf)
        end
    end

    # E_pr (nC,nR): pr - pds + pms = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_pr, cr(c,r))
        ae!(row, :pr,  cr(c,r),  1.0)
        ae!(row, :pds, cr(c,r), -1.0)
        ae!(row, :pms, cr(c,r),  1.0)
    end

    # E_qds (nC,nR): qds - Σ_a FDCSHR*qfd - PDCSHR*qpd - GDCSHR*qgd - IDCSHR*qid = 0
    for c in 1:nC, r in 1:nR
        row = er(:E_qds, cr(c,r))
        ae!(row, :qds, cr(c,r),  1.0)
        for a in 1:nA
            ae!(row, :qfd, car(c,a,r), -C.FDCSHR[c,a,r])
        end
        ae!(row, :qpd, cr(c,r), -C.PDCSHR[c,r])
        ae!(row, :qgd, cr(c,r), -C.GDCSHR[c,r])
        ae!(row, :qid, cr(c,r), -C.IDCSHR[c,r])
    end

    # E_pds (nC,nR): qc - DSSHR*qds - Σ_d XSSHR*qxs - (STSHR*qst if MARG) - tradslack = 0
    # (tradslack exog)
    for c in 1:nC, r in 1:nR
        row = er(:E_pds, cr(c,r))
        ae!(row, :qc,  cr(c,r),  1.0)
        ae!(row, :qds, cr(c,r), -C.DSSHR[c,r])
        for dd in 1:nR
            ae!(row, :qxs, crr(c,r,dd), -C.XSSHR[c,r,dd])
        end
        if C.isMARG[c]
            m = iM(s.COMM[c])::Int
            ae!(row, :qst, mr(m,r), -C.STSHR[m,r])
        end
    end

    # ── Transport ────────────────────────────────────────────────────────────

    # E_atmfsd (nM,nC,nR,nR): atmfsd - atm - atf - ats - atd - atall = 0 (all exog)
    for m in 1:nM, c in 1:nC, ss in 1:nR, dd in 1:nR
        ae!(er(:E_atmfsd, mcrr(m,c,ss,dd)), :atmfsd, mcrr(m,c,ss,dd), 1.0)
    end

    # E_qtmfsd (nM,nC,nR,nR): qtmfsd - qxs + atmfsd = 0
    for m in 1:nM, c in 1:nC, ss in 1:nR, dd in 1:nR
        row = er(:E_qtmfsd, mcrr(m,c,ss,dd))
        ae!(row, :qtmfsd, mcrr(m,c,ss,dd),  1.0)
        ae!(row, :qxs,    crr(c,ss,dd),     -1.0)
        ae!(row, :atmfsd, mcrr(m,c,ss,dd),   1.0)
    end

    # E_qtm (nM): qtm - Σ_{c,ss,dd} VTMUSESHR*qtmfsd = 0
    for m in 1:nM
        row = er(:E_qtm, m)
        ae!(row, :qtm, m, 1.0)
        for c in 1:nC, ss in 1:nR, dd in 1:nR
            ae!(row, :qtmfsd, mcrr(m,c,ss,dd), -C.VTMUSESHR[m,c,ss,dd])
        end
    end

    # E_pt (nM): pt - Σ_r VTSUPPSHR*pds[iC(MARG[m]),r] = 0
    for m in 1:nM
        row = er(:E_pt, m)
        ae!(row, :pt, m, 1.0)
        cm  = iC(s.MARG[m])::Int
        for r in 1:nR
            ae!(row, :pds, cr(cm,r), -C.VTSUPPSHR[m,r])
        end
    end

    # E_qst (nM,nR): qst - qtm + ESUBS*(pt - pds[iC(MARG[m]),r]) = 0
    for m in 1:nM, r in 1:nR
        row = er(:E_qst, mr(m,r))
        cm  = iC(s.MARG[m])::Int
        e   = d.ESUBS[m]
        ae!(row, :qst, mr(m,r),  1.0)
        ae!(row, :qtm, m,        -1.0)
        ae!(row, :pt,  m,         e)
        ae!(row, :pds, cr(cm,r), -e)
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 7 – ENDOWMENTS
    # ══════════════════════════════════════════════════════════════════════════

    # E_pe1 (nEM,nR): qe[iEMS(m),r] - Σ_a ENDWMSHR*qfe[iE(m),a,r] - endwslack = 0
    # (qe, endwslack exog)
    for (mi, _) in enumerate(s.ENDWM), r in 1:nR
        row = er(:E_pe1, emr(mi,r))
        ei  = iE(s.ENDWM[mi])::Int
        for a in 1:nA
            ae!(row, :qfe, ear(ei,a,r), -C.ENDWMSHR[mi,a,r])
        end
    end

    # E_qes1 (nEM,nA,nR): pes[iE(m),a,r] - pe[iEMS(m),r] = 0
    for (mi, _) in enumerate(s.ENDWM), a in 1:nA, r in 1:nR
        local idx = mi + (a-1)*nEM + (r-1)*nEM*nA
        row = er(:E_qes1, idx)
        ei  = iE(s.ENDWM[mi])::Int
        ems = iEMS(s.ENDWM[mi])::Int
        ae!(row, :pes, ear(ei,a,r),   1.0)
        ae!(row, :pe,  emsr(ems,r),  -1.0)
    end

    # E_qes2 (nES,nA,nR): qes[iE(s),a,r] - qe[iEMS(s),r] + ETRAE*(pes[iE(s),a,r]-pe[iEMS(s),r]) + endwslack = 0
    # (qe, endwslack exog)
    for (si, _) in enumerate(s.ENDWS), a in 1:nA, r in 1:nR
        local idx = si + (a-1)*nES + (r-1)*nES*nA
        row = er(:E_qes2, idx)
        ei  = iE(s.ENDWS[si])::Int
        ems = iEMS(s.ENDWS[si])::Int
        et  = d.ETRAE[ei,r]
        ae!(row, :qes, ear(ei,a,r),   1.0)
        ae!(row, :pes, ear(ei,a,r),   et)
        ae!(row, :pe,  emsr(ems,r),  -et)
    end

    # E_pe2 (nES,nR): pe[iEMS(s),r] - Σ_a REVSHR*pes[iE(s),a,r] = 0
    for (si, _) in enumerate(s.ENDWS), r in 1:nR
        row = er(:E_pe2, si + (r-1)*nES)
        ei  = iE(s.ENDWS[si])::Int
        ems = iEMS(s.ENDWS[si])::Int
        ae!(row, :pe, emsr(ems,r), 1.0)
        for a in 1:nA
            ae!(row, :pes, ear(ei,a,r), -C.REVSHR[ei,a,r])
        end
    end

    # E_qes3 (nEF,nA,nR): qes[iE(f),a,r] - qesf[f,a,r] = 0  (qesf exog)
    for (fi, _) in enumerate(s.ENDWF), a in 1:nA, r in 1:nR
        local idx = fi + (a-1)*nEF + (r-1)*nEF*nA
        row = er(:E_qes3, idx)
        ei  = iE(s.ENDWF[fi])::Int
        ae!(row, :qes, ear(ei,a,r), 1.0)
    end

    # E_pfe (nE,nA,nR): pfe - peb - tfe = 0  (tfe exog)
    for e in 1:nE, a in 1:nA, r in 1:nR
        row = er(:E_pfe, ear(e,a,r))
        ae!(row, :pfe, ear(e,a,r),  1.0)
        ae!(row, :peb, ear(e,a,r), -1.0)
    end

    # E_pes_link (nE,nA,nR): peb - pes - tinc = 0  (tinc exog)
    for e in 1:nE, a in 1:nA, r in 1:nR
        row = er(:E_pes_link, ear(e,a,r))
        ae!(row, :peb, ear(e,a,r),  1.0)
        ae!(row, :pes, ear(e,a,r), -1.0)
    end

    # E_peb (nE,nA,nR): qfe - qes = 0
    for e in 1:nE, a in 1:nA, r in 1:nR
        row = er(:E_peb, ear(e,a,r))
        ae!(row, :qfe, ear(e,a,r),  1.0)
        ae!(row, :qes, ear(e,a,r), -1.0)
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 8 – CAPITAL AND INVESTMENT
    # ══════════════════════════════════════════════════════════════════════════

    # E_rental (nR): rental - Σ_k (VES[ec,r]/GROSSCAP)*pe[iEMS(ENDWC[k]),r] = 0
    for r in 1:nR
        row = er(:E_rental, r)
        ae!(row, :rental, r, 1.0)
        gc  = max(C.GROSSCAP[r], 1e-10)
        for (k, ec) in enumerate(iE.(s.ENDWC))
            ems_k = iEMS(s.ENDWC[k])::Int
            ae!(row, :pe, emsr(ems_k,r), -(C.VES[ec,r]/gc))
        end
    end

    # E_ke (nR): ke - INVKERATIO*qinv - (1-INVKERATIO)*kb = 0
    for r in 1:nR
        row = er(:E_ke, r)
        ik  = C.INVKERATIO[r]
        ae!(row, :ke,   r,  1.0)
        ae!(row, :qinv, r, -ik)
        ae!(row, :kb,   r, -(1.0 - ik))
    end

    # E_kb (nR): kb - Σ_k (VES[ec,r]/GROSSCAP)*qe[iEMS(ENDWC[k]),r] = 0
    # (qe exog → only kb appears)
    for r in 1:nR
        ae!(er(:E_kb, r), :kb, r, 1.0)
    end

    # E_rorc (nR): rorc - GRNETRATIO*(rental - pinv) = 0
    for r in 1:nR
        row = er(:E_rorc, r)
        g   = C.GRNETRATIO[r]
        ae!(row, :rorc,   r,  1.0)
        ae!(row, :rental, r, -g)
        ae!(row, :pinv,   r,  g)
    end

    # E_rore (nR): rore - rorc + RORFLEX*(ke - kb) = 0
    for r in 1:nR
        row = er(:E_rore, r)
        rf  = d.RORFLEX[r]
        ae!(row, :rore, r,  1.0)
        ae!(row, :rorc, r, -1.0)
        ae!(row, :ke,   r,  rf)
        ae!(row, :kb,   r, -rf)
    end

    # E_qinv (nR):
    # RORDELTA*rore + (1-RORDELTA)*(REGINV/NETINV*qinv - VDEP/NETINV*kb)
    # - RORDELTA*rorg - (1-RORDELTA)*globalcgds - cgdslack = 0  (cgdslack exog)
    for r in 1:nR
        row  = er(:E_qinv, r)
        rd   = d.RORDELTA
        ni   = max(C.NETINV[r], 1e-10)
        ae!(row, :rore,       r,   rd)
        ae!(row, :qinv,       r,   (1.0-rd)*C.REGINV[r]/ni)
        ae!(row, :kb,         r,  -(1.0-rd)*d.VDEP[r]/ni)
        ae!(row, :rorg,       1,  -rd)
        ae!(row, :globalcgds, 1,  -(1.0-rd))
    end

    # E_expand (nEC,nR): expand[k,r] - qinv[r] + qe[iEMS(ENDWC[k]),r] = 0
    # (qe exog)
    for k in 1:nEC, r in 1:nR
        row = er(:E_expand, ecr(k,r))
        ae!(row, :expand, ecr(k,r),  1.0)
        ae!(row, :qinv,   r,         -1.0)
    end

    # E_globalcgds (1):
    # RORDELTA*globalcgds + (1-RORDELTA)*rorg
    # - RORDELTA*Σ_r[(REGINV/GLOBINV)*qinv - (VDEP/GLOBINV)*kb]
    # - (1-RORDELTA)*Σ_r (NETINV/GLOBINV)*rore = 0
    let row = er(:E_globalcgds, 1)
        rd   = d.RORDELTA
        gl   = max(C.GLOBINV, 1e-10)
        ae!(row, :globalcgds, 1,  rd)
        ae!(row, :rorg,       1,  1.0-rd)
        for r in 1:nR
            ae!(row, :qinv, r,  -rd * C.REGINV[r] / gl)
            ae!(row, :kb,   r,   rd * d.VDEP[r]   / gl)
            ae!(row, :rore, r,  -(1.0-rd) * C.NETINV[r] / gl)
        end
    end

    # E_psave (nR):
    # psave - pinv - Σ_ss ((NETINV[ss]-SAVE[ss])/GLOBINV)*pinv[ss] - psaveslack = 0
    # (psaveslack exog)
    let gl = max(C.GLOBINV, 1e-10)
        for r in 1:nR
            row = er(:E_psave, r)
            ae!(row, :psave, r, 1.0)
            # coefficient of each pinv[ss] (including ss=r)
            for ss in 1:nR
                cf = (ss == r ? -1.0 : 0.0) - (C.NETINV[ss] - d.SAVE[ss]) / gl
                ae!(row, :pinv, ss, cf)
            end
        end
    end

    # E_pcgdswld (1): pcgdswld - Σ_r (NETINV/GLOBINV)*pinv = 0
    let row = er(:E_pcgdswld, 1), gl = max(C.GLOBINV, 1e-10)
        ae!(row, :pcgdswld, 1, 1.0)
        for r in 1:nR
            ae!(row, :pinv, r, -C.NETINV[r]/gl)
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 9/10 – NUMERAIRE AND WALRAS
    # ══════════════════════════════════════════════════════════════════════════

    # E_pfactor (nR): VENDWREG*pfactor - Σ_{e,a} EVFB*peb = 0
    let VENDWREG = [sum(d.EVFB[:,:,r]) for r in 1:nR]
        for r in 1:nR
            row = er(:E_pfactor, r)
            ae!(row, :pfactor, r, VENDWREG[r])
            for e in 1:nE, a in 1:nA
                ae!(row, :peb, ear(e,a,r), -d.EVFB[e,a,r])
            end
        end
    end

    # E_pfactwld (1): VENDWWLD*pfactwld - Σ_r VENDWREG*pfactor = 0
    # (pfactwld exog → only pfactor[r] entries)
    let VENDWREG = [sum(d.EVFB[:,:,r]) for r in 1:nR], row = er(:E_pfactwld, 1)
        for r in 1:nR
            ae!(row, :pfactor, r, -VENDWREG[r])
        end
    end

    # E_walras_sup (1): walras_sup - pcgdswld - globalcgds = 0
    let row = er(:E_walras_sup, 1)
        ae!(row, :walras_sup,  1,  1.0)
        ae!(row, :pcgdswld,    1, -1.0)
        ae!(row, :globalcgds,  1, -1.0)
    end

    # E_walras_dem (1): GLOBINV*walras_dem - Σ_r SAVE*(psave+qsave) = 0
    let row = er(:E_walras_dem, 1), gl = C.GLOBINV
        ae!(row, :walras_dem, 1, gl)
        for r in 1:nR
            sv = d.SAVE[r]
            ae!(row, :psave, r, -sv)
            ae!(row, :qsave, r, -sv)
        end
    end

    # E_walras (1): walras_sup - walras_dem - walraslack = 0  (walraslack exog)
    let row = er(:E_walras, 1)
        ae!(row, :walras_sup, 1,  1.0)
        ae!(row, :walras_dem, 1, -1.0)
    end

    # ── Assemble sparse matrix ────────────────────────────────────────────────
    nnz = ptr[]
    sparse(II[1:nnz], JJ[1:nnz], VV[1:nnz], n_row, n_col)
end
