# gtap_pack.jl
# Pack / unpack the endogenous variable vector for the direct sparse-matrix GTAP solver.

# GTAPData, GTAPSets, gtap_residuals are defined in gtap_v62.jl
if !isdefined(Main, :GTAPSets)
    include(joinpath(@__DIR__, "gtap_v62.jl"))
end
#
# Design:
#   x_endo  – flat Float64 vector of all endogenous (non-appendix) variables
#   exog    – NamedTuple of all exogenous variables (instruments + slacks + numeraire)
#   unpack_endo(x_endo, exog, s) returns the full NamedTuple `v` that gtap_residuals expects
#   pack_residuals_core(R, s) packs only the Module 1-8 equations into a flat vector
#
# Closure: standard GTAP (all tech/tax shocks = 0, pfactwld = 0 numeraire)
# Appendix A/B/C variables are excluded from x_endo; compute them post-solve if needed.

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXOGENOUS VARIABLE BUNDLE  (all zero = benchmark; modify to apply shocks)
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_exog_zero(s) → NamedTuple

Return a NamedTuple of all exogenous variables set to zero (benchmark).
Mutate individual fields to apply shocks before passing to `build_gtap_jacobian`.

Example:
    exog = make_exog_zero(s)
    exog_shocked = merge(exog, (; tm = copy(exog.tm)))
    exog_shocked.tm[1, 1] = 10.0   # 10% tariff on commodity 1 into region 1
"""
function make_exog_zero(s::GTAPSets)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nNS=length(s.NSAV_COMM)
    return (;
        # ── Technology shocks ──────────────────────────────────────────────
        ao       = zeros(nP,nR),
        ava      = zeros(nP,nR),
        af       = zeros(nT,nP,nR),
        afe      = zeros(nE,nP,nR),
        ams      = zeros(nT,nR,nR),
        # Tech decompositions (always zero)
        aosec    = zeros(nP),  aoreg  = zeros(nR),  aoall  = zeros(nP,nR),
        avasec   = zeros(nP),  avareg = zeros(nR),  avaall = zeros(nP,nR),
        afcom    = zeros(nT),  afsec  = zeros(nP),  afreg  = zeros(nR),
        afall    = zeros(nT,nP,nR),
        afecom   = zeros(nE),  afesec = zeros(nP),  afereg = zeros(nR),
        afeall   = zeros(nE,nP,nR),
        # ── Tax instruments ────────────────────────────────────────────────
        to       = zeros(nNS,nR),
        tgd      = zeros(nT,nR),   tgm = zeros(nT,nR),
        tfd      = zeros(nT,nP,nR), tfm = zeros(nT,nP,nR),
        tf       = zeros(nE,nP,nR),
        tx       = zeros(nT,nR),
        txs      = zeros(nT,nR,nR),
        tm       = zeros(nT,nR),
        tms      = zeros(nT,nR,nR),
        tpd      = zeros(nT,nR),   tpm = zeros(nT,nR),   tp = zeros(nR),
        # ── Transport tech ─────────────────────────────────────────────────
        atmfsd   = zeros(nM,nT,nR,nR),
        atm      = zeros(nM),  atf = zeros(nT),  ats = zeros(nR),
        atd      = zeros(nR),  atall = zeros(nM,nT,nR,nR),
        # ── Preference / demographic shocks ───────────────────────────────
        pop      = zeros(nR),
        dppriv   = zeros(nR),  dpgov = zeros(nR),  dpsave = zeros(nR),
        au       = zeros(nR),
        # ── Numeraire ─────────────────────────────────────────────────────
        pfactwld = 0.0,
        # ── Closure slacks (all zero → standard closure) ──────────────────
        psaveslack  = zeros(nR),
        cgdslack    = zeros(nR),
        endwslack   = zeros(nE,nR),
        profitslack = zeros(nP,nR),
        walraslack  = 0.0,
        incomeslack = zeros(nR),
        # ── Endowment supply (exogenous in standard short-run GTAP closure) ─
        # Shock qo_endw[ei,r] to simulate endowment changes (population, land supply…)
        qo_endw     = zeros(nE,nR),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENDOGENOUS VARIABLE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (name::Symbol, shape::Tuple)
# pmes and qoes are stored only for ENDWS_COMM (sluggish) rows → shape (nES,nP,nR)
# They are unpacked to full [nE,nP,nR] with zeros for non-sluggish rows.

function _endo_specs(s::GTAPSets)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEC=length(s.ENDWC_COMM)
    nNS=length(s.NSAV_COMM)
    return [
        # ── Module 1 – Government ─────────────────────────────────────────
        (:pgov,     (nR,)),
        (:pgd,      (nT,nR)),  (:pgm,  (nT,nR)),  (:pg,   (nT,nR)),
        (:ug,       (nR,)),    (:yg,   (nR,)),
        (:qg,       (nT,nR)),  (:qgd,  (nT,nR)),  (:qgm,  (nT,nR)),
        (:del_taxrgc,(nR,)),
        # ── Module 2 – Private ────────────────────────────────────────────
        (:ppriv,    (nR,)),
        (:ppd,      (nT,nR)),  (:ppm,  (nT,nR)),  (:pp,   (nT,nR)),
        (:up,       (nR,)),    (:yp,   (nR,)),    (:uepriv,(nR,)),
        (:qp,       (nT,nR)),  (:qpd,  (nT,nR)),  (:qpm,  (nT,nR)),
        (:pim,      (nT,nR)),  (:qim,  (nT,nR)),
        (:atpd,     (nT,nR)),  (:atpm, (nT,nR)),
        (:del_taxrpc,(nR,)),
        # ── Module 3 – Firms ──────────────────────────────────────────────
        (:pva,      (nP,nR)),  (:qva,  (nP,nR)),
        (:pf,       (nT,nP,nR)), (:qf, (nT,nP,nR)),
        (:pfd,      (nT,nP,nR)), (:qfd,(nT,nP,nR)),
        (:pfm,      (nT,nP,nR)), (:qfm,(nT,nP,nR)),
        (:pfe,      (nE,nP,nR)), (:qfe,(nE,nP,nR)),
        # qo: only production+CGDS sectors are endogenous (nE+1 : nNS in NSAV_COMM)
        # Endowment supplies qo[ENDW_COMM,r] are exogenous → see exog.qo_endw
        (:qo_prod_cgds, (nT+length(s.CGDS_COMM),nR)),
        (:qds,      (nT,nR)),
        # pmes/qoes: only sluggish rows stored; unpacked to [nE,nP,nR]
        (:pmes_slug,(nES,nP,nR)),
        (:qoes_slug,(nES,nP,nR)),
        (:ps,       (nNS,nR)),
        (:del_taxriu,(nR,)),  (:del_taxrfu,(nR,)),  (:del_taxrout,(nR,)),
        # ── Module 4 – Investment ─────────────────────────────────────────
        (:rental,   (nR,)),    (:kb,   (nR,)),    (:ke,   (nR,)),
        (:rorc,     (nR,)),    (:rore, (nR,)),    (:ksvces,(nR,)),
        (:qcgds,    (nR,)),    (:pcgds,(nR,)),
        (:pcgdswld, (1,)),     (:globalcgds,(1,)), (:rorg, (1,)),
        (:EXPAND,   (nEC,nR)),
        (:psave,    (nR,)),    (:qsave,(nR,)),
        # ── Module 5 – Trade ──────────────────────────────────────────────
        (:pfob,     (nT,nR,nR)), (:pcif,(nT,nR,nR)), (:pms,(nT,nR,nR)),
        (:pr,       (nT,nR)),
        (:qxs,      (nT,nR,nR)),
        (:del_taxrexp,(nR,)),  (:del_taxrimp,(nR,)),
        # ── Module 6 – Transport ──────────────────────────────────────────
        (:qtmfsd,   (nM,nT,nR,nR)),
        (:qtm,      (nM,)),    (:pt,   (nM,)),
        (:ptrans,   (nT,nR,nR)),
        (:qst,      (nM,nR)),
        # ── Module 7 – Regional household ────────────────────────────────
        (:pm,       (nNS,nR)),
        (:fincome,  (nR,)),
        (:del_taxrinc,(nR,)),  (:del_ttaxr,(nR,)),  (:del_indtaxr,(nR,)),
        (:dpav,     (nR,)),    (:uelas,(nR,)),     (:dpsum,(nR,)),
        (:y,        (nR,)),    (:p,    (nR,)),     (:u,    (nR,)),
        # ── Module 8 – Equilibrium ────────────────────────────────────────
        (:walras_sup,(1,)),    (:walras_dem,(1,)),
    ]
end

"""
    endo_offsets(s) → (specs, offsets::Dict{Symbol,Int}, n_total::Int)

Return the variable layout and flat-vector offsets for the endogenous system.
`offsets[name]` is the 1-based starting index of that variable block in x_endo.
"""
function endo_offsets(s::GTAPSets)
    specs = _endo_specs(s)
    offsets = Dict{Symbol,Int}()
    total = 0
    for (name, shape) in specs
        offsets[name] = total + 1
        total += prod(shape)
    end
    return specs, offsets, total
end

"""
    n_endog(s) → Int

Total number of endogenous variables in the core system.
"""
function n_endog(s::GTAPSets)
    _, _, n = endo_offsets(s)
    return n
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  UNPACK: flat vector  →  NamedTuple  (compatible with gtap_residuals)
# ─────────────────────────────────────────────────────────────────────────────

"""
    unpack_endo(x_endo, exog, s) → NamedTuple

Build the full variable NamedTuple `v` expected by `gtap_residuals` from the
flat endogenous vector `x_endo` and the exogenous NamedTuple `exog`.

Works with any element type (Float64, ForwardDiff.Dual, …).
"""
function unpack_endo(x_endo::AbstractVector{T}, exog, s::GTAPSets) where T
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEC=length(s.ENDWC_COMM)
    nNS=length(s.NSAV_COMM)

    _, offsets, _ = endo_offsets(s)

    # Helper: view a block of x_endo as an array with the given shape
    function blk(name::Symbol, shape::Tuple)
        off = offsets[name]
        n   = prod(shape)
        reshape(view(x_endo, off:off+n-1), shape)
    end
    # Scalar helper (1-element block → scalar)
    sc(name::Symbol) = x_endo[offsets[name]]

    # ── Expand pmes_slug / qoes_slug → full [nE,nP,nR] ───────────────────
    # Build mapping: ENDWS_COMM[si] → index in ENDW_COMM
    slug_in_endw = [findfirst(==(e), s.ENDW_COMM) for e in s.ENDWS_COMM]

    pmes_full = zeros(T, nE, nP, nR)
    qoes_full = zeros(T, nE, nP, nR)
    pmes_s = blk(:pmes_slug, (nES,nP,nR))
    qoes_s = blk(:qoes_slug, (nES,nP,nR))
    for (si, ei) in enumerate(slug_in_endw)
        pmes_full[ei, :, :] = pmes_s[si, :, :]
        qoes_full[ei, :, :] = qoes_s[si, :, :]
    end

    return (;
        # ── Endogenous ─────────────────────────────────────────────────────
        pm          = blk(:pm,         (nNS,nR)),
        ps          = blk(:ps,         (nNS,nR)),
        pgov        = blk(:pgov,       (nR,)),
        pgd         = blk(:pgd,        (nT,nR)),
        pgm         = blk(:pgm,        (nT,nR)),
        pg          = blk(:pg,         (nT,nR)),
        ug          = blk(:ug,         (nR,)),
        yg          = blk(:yg,         (nR,)),
        qg          = blk(:qg,         (nT,nR)),
        qgd         = blk(:qgd,        (nT,nR)),
        qgm         = blk(:qgm,        (nT,nR)),
        ppriv       = blk(:ppriv,      (nR,)),
        ppd         = blk(:ppd,        (nT,nR)),
        ppm         = blk(:ppm,        (nT,nR)),
        pp          = blk(:pp,         (nT,nR)),
        up          = blk(:up,         (nR,)),
        yp          = blk(:yp,         (nR,)),
        uepriv      = blk(:uepriv,     (nR,)),
        qp          = blk(:qp,         (nT,nR)),
        qpd         = blk(:qpd,        (nT,nR)),
        qpm         = blk(:qpm,        (nT,nR)),
        pim         = blk(:pim,        (nT,nR)),
        qim         = blk(:qim,        (nT,nR)),
        pva         = blk(:pva,        (nP,nR)),
        qva         = blk(:qva,        (nP,nR)),
        pf          = blk(:pf,         (nT,nP,nR)),
        qf          = blk(:qf,         (nT,nP,nR)),
        pfd         = blk(:pfd,        (nT,nP,nR)),
        qfd         = blk(:qfd,        (nT,nP,nR)),
        pfm         = blk(:pfm,        (nT,nP,nR)),
        qfm         = blk(:qfm,        (nT,nP,nR)),
        pfe         = blk(:pfe,        (nE,nP,nR)),
        qfe         = blk(:qfe,        (nE,nP,nR)),
        qo          = begin
            # Assemble full qo[nNS,nR]: ENDW rows from exog, TRAD+CGDS rows from x_endo
            # NSAV_COMM ordering: ENDW_COMM (1:nE) | TRAD_COMM (nE+1:nE+nT) | CGDS_COMM (nNS)
            nCGDS = length(s.CGDS_COMM)
            qo_full = zeros(T, nNS, nR)
            for ei in 1:nE, r in 1:nR
                qo_full[ei, r] = exog.qo_endw[ei, r]
            end
            qo_prod = blk(:qo_prod_cgds, (nT+nCGDS, nR))
            qo_full[nE+1:nNS, :] = qo_prod
            qo_full
        end,
        qds         = blk(:qds,        (nT,nR)),
        pmes        = pmes_full,   # [nE,nP,nR], zeros for mobile rows
        qoes        = qoes_full,   # [nE,nP,nR], zeros for mobile rows
        atpd        = blk(:atpd,       (nT,nR)),
        atpm        = blk(:atpm,       (nT,nR)),
        pfob        = blk(:pfob,       (nT,nR,nR)),
        pcif        = blk(:pcif,       (nT,nR,nR)),
        pms         = blk(:pms,        (nT,nR,nR)),  # field name 'pms' for gtap_residuals
        pr          = blk(:pr,         (nT,nR)),
        qxs         = blk(:qxs,        (nT,nR,nR)),
        qtmfsd      = blk(:qtmfsd,     (nM,nT,nR,nR)),
        qtm         = blk(:qtm,        (nM,)),
        pt          = blk(:pt,         (nM,)),
        ptrans      = blk(:ptrans,     (nT,nR,nR)),
        qst         = blk(:qst,        (nM,nR)),
        rental      = blk(:rental,     (nR,)),
        kb          = blk(:kb,         (nR,)),
        ke          = blk(:ke,         (nR,)),
        rorc        = blk(:rorc,       (nR,)),
        rore        = blk(:rore,       (nR,)),
        ksvces      = blk(:ksvces,     (nR,)),
        qcgds       = blk(:qcgds,      (nR,)),
        pcgds       = blk(:pcgds,      (nR,)),
        pcgdswld    = sc(:pcgdswld),
        globalcgds  = sc(:globalcgds),
        rorg        = sc(:rorg),
        EXPAND      = begin
            # gtap_residuals line 662 uses EXPAND[iE(h),r] where iE(h) is capital's
            # index in ENDW_COMM (up to nE), but the block is only (nEC,nR).
            # Expand to (nE,nR) so the dead-code generator doesn't bounds-error.
            cap_in_endw = [findfirst(==(c), s.ENDW_COMM) for c in s.ENDWC_COMM]
            ex_blk = blk(:EXPAND, (nEC,nR))
            ex_full = zeros(T, nE, nR)
            for (k, ei) in enumerate(cap_in_endw)
                ex_full[ei, :] = ex_blk[k, :]
            end
            ex_full
        end,
        psave       = blk(:psave,      (nR,)),
        qsave       = blk(:qsave,      (nR,)),
        y           = blk(:y,          (nR,)),
        p           = blk(:p,          (nR,)),
        u           = blk(:u,          (nR,)),
        dpav        = blk(:dpav,       (nR,)),
        uelas       = blk(:uelas,      (nR,)),
        dpsum       = blk(:dpsum,      (nR,)),
        fincome     = blk(:fincome,    (nR,)),
        del_taxrgc  = blk(:del_taxrgc, (nR,)),
        del_taxrpc  = blk(:del_taxrpc, (nR,)),
        del_taxriu  = blk(:del_taxriu, (nR,)),
        del_taxrfu  = blk(:del_taxrfu, (nR,)),
        del_taxrout = blk(:del_taxrout,(nR,)),
        del_taxrexp = blk(:del_taxrexp,(nR,)),
        del_taxrimp = blk(:del_taxrimp,(nR,)),
        del_taxrinc = blk(:del_taxrinc,(nR,)),
        del_ttaxr   = blk(:del_ttaxr,  (nR,)),
        del_indtaxr = blk(:del_indtaxr,(nR,)),
        walras_sup  = sc(:walras_sup),
        walras_dem  = sc(:walras_dem),
        # ── Appendix A/B/C variables: zero stubs so gtap_residuals doesn't crash ──
        # Their equations are excluded from CORE_EQ_NAMES; values here don't affect solve.
        pfactreal   = zeros(T, nE, nR),
        pfactor     = zeros(T, nR),
        psw         = zeros(T, nR),
        pdw         = zeros(T, nR),
        tot         = zeros(T, nR),
        vgdp        = zeros(T, nR),
        pgdp        = zeros(T, nR),
        qgdp        = zeros(T, nR),
        compvalad   = zeros(T, nP, nR),
        vxwfob      = zeros(T, nT, nR),
        vxwreg      = zeros(T, nR),
        viwcif      = zeros(T, nT, nR),
        viwreg      = zeros(T, nR),
        DTBALi      = zeros(T, nT, nR),
        DTBAL       = zeros(T, nR),
        DTBALR      = zeros(T, nR),
        ygev        = zeros(T, nR),
        ugev        = zeros(T, nR),
        qpev        = zeros(T, nT, nR),
        ypev        = zeros(T, nR),
        upev        = zeros(T, nR),
        ueprivev    = zeros(T, nR),
        dpavev      = zeros(T, nR),
        uelasev     = zeros(T, nR),
        yev         = zeros(T, nR),
        ysaveev     = zeros(T, nR),
        qsaveev     = zeros(T, nR),
        EV          = zeros(T, nR),
        WEV         = zero(T),
        EV_ALT      = zeros(T, nR),
        WEV_ALT     = zero(T),
        # ── Exogenous (from exog NamedTuple) ──────────────────────────────
        ao          = exog.ao,
        ava         = exog.ava,
        af          = exog.af,
        afe         = exog.afe,
        ams         = exog.ams,
        aosec       = exog.aosec,  aoreg  = exog.aoreg,  aoall  = exog.aoall,
        avasec      = exog.avasec, avareg = exog.avareg, avaall = exog.avaall,
        afcom       = exog.afcom,  afsec  = exog.afsec,  afreg  = exog.afreg,
        afall       = exog.afall,
        afecom      = exog.afecom, afesec = exog.afesec, afereg = exog.afereg,
        afeall      = exog.afeall,
        to          = exog.to,
        tgd         = exog.tgd,   tgm  = exog.tgm,
        tfd         = exog.tfd,   tfm  = exog.tfm,
        tf          = exog.tf,
        tx          = exog.tx,
        txs         = exog.txs,
        tm          = exog.tm,
        tms         = exog.tms,
        tpd         = exog.tpd,   tpm  = exog.tpm,   tp = exog.tp,
        atmfsd      = exog.atmfsd,
        atm         = exog.atm,   atf  = exog.atf,   ats = exog.ats,
        atd         = exog.atd,   atall= exog.atall,
        pop         = exog.pop,
        dppriv      = exog.dppriv, dpgov = exog.dpgov, dpsave = exog.dpsave,
        au          = exog.au,
        pfactwld    = exog.pfactwld,
        psaveslack  = exog.psaveslack,
        cgdslack    = exog.cgdslack,
        endwslack   = exog.endwslack,
        profitslack = exog.profitslack,
        walraslack  = exog.walraslack,
        incomeslack = exog.incomeslack,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PACK RESIDUALS: Dict  →  flat vector  (core equations only, Modules 1-8)
# ─────────────────────────────────────────────────────────────────────────────

# Equation names in Module 1-8 that contain at least one endogenous variable.
# AOWORLD / AVAWORLD / AFWORLD / AFEWORLD / TRANSTECHANGE are all-exogenous → omitted.
const CORE_EQ_NAMES = [
    # Module 1
    :GPRICEINDEX, :GOVDMNDS, :GOVU, :GHHDPRICE, :GHHIPRICES,
    :GCOMPRICE, :GHHLDAGRIMP, :GHHLDDOM, :TGCRATIO,
    # Module 2
    :PHHLDINDEX, :PRIVATEU, :UTILELASPRIV, :PRIVDMNDS,
    :TPDSHIFT, :PHHDPRICE, :TPMSHIFT, :PHHIPRICES,
    :PCOMPRICE, :PHHLDDOM, :PHHLDAGRIMP, :TPCRATIO,
    # Module 3
    :VADEMAND, :INTDEMAND, :DMNDDPRICE, :DMNDIPRICES,
    :ICOMPRICE, :INDIMP, :INDDOM,
    :VAPRICE, :ENDWDEMAND, :MPFACTPRICE, :SPFACTPRICE,
    :OUTPUTPRICES, :TIURATIO, :TFURATIO, :ZEROPROFITS, :TOUTRATIO,
    # Module 4
    :KAPSVCES, :KAPRENTAL, :CAPGOODS, :PRCGOODS,
    :KBEGINNING, :KEND, :RORCURRENT, :ROREXPECTED,
    :BALDWIN, :RORGLOBAL, :GLOBALINV, :PRICGDS, :SAVEPRICE,
    # Module 5
    :EXPRICES, :TEXPRATIO, :MKTPRICES, :DPRICEIMP,
    :PRICETGT, :IMPORTDEMAND, :TIMPRATIO,
    # Module 6
    :QTRANS_MFSD, :TRANS_DEMAND, :PTRANSPORT,
    :TRANSCOSTINDEX, :TRANSVCES, :FOBCIF,
    # Module 7
    :FACTORINCPRICES, :TINCRATIO, :ENDW_PRICE, :ENDW_SUPPLY,
    :FACTORINCOME, :DINDTAXRATIO, :DTAXRATIO, :REGIONALINCOME,
    :DPARAV, :UTILITELASTIC, :PRIVCONSEXP, :GOVCONSEXP,
    :SAVING, :PRICEINDEXREG, :UTILITY, :DISTPARSUM,
    # Module 8
    :MKTCLDOM, :MKTCLTRD_MARG, :MKTCLTRD_NMRG,
    :MKTCLIMP, :MKTCLENDWM, :MKTCLENDWS,
    :WALRAS_S, :WALRAS_D, :WALRAS,
]

"""
    pack_residuals_core(R, s) → Vector{Float64}

Pack the core (Modules 1-8) residuals from `gtap_residuals` into a flat vector.
Appendix A/B/C equations are excluded.
Works with any numeric element type (supports ForwardDiff).
"""
function pack_residuals_core(R::Dict{Symbol,Any}, s::GTAPSets)
    T = mapreduce(r -> eltype(vec(r isa Number ? [r] : r)), promote_type, values(R))
    result = T[]
    for name in CORE_EQ_NAMES
        r = R[name]
        if r isa Number
            push!(result, r)
        else
            append!(result, vec(r))
        end
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINED RESIDUAL FUNCTION  (used by SparseDiffTools)
# ─────────────────────────────────────────────────────────────────────────────

"""
    F_core(x_endo, exog, d, s, C) → Vector

Evaluate the core (Module 1-8) residuals as a flat vector.
At the benchmark (x_endo = 0, exog = zero), F_core = 0.
After a shock in exog, F_core(0, shocked_exog) = -b  (the RHS for the linear solve).
"""
function F_core(x_endo::AbstractVector, exog, d::GTAPData, s::GTAPSets, C)
    v = unpack_endo(x_endo, exog, s)
    R = gtap_residuals(v, d, s, C; skip_appendix=true)
    return pack_residuals_core(R, s)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    print_endo_layout(s)

Print the endogenous variable layout: name, shape, flat size, and cumulative offset.
"""
function print_endo_layout(s::GTAPSets)
    specs, _, total = endo_offsets(s)
    cum = 0
    println("Endogenous variable layout  (total = $total)")
    println("  $(rpad("name",20)) $(rpad("shape",18)) size   offset")
    println("  ", "─"^60)
    for (name, shape) in specs
        sz = prod(shape)
        cum += sz
        println("  $(rpad(String(name),20)) $(rpad(string(shape),18)) $(rpad(sz,6)) $(cum-sz+1)")
    end
end

"""
    n_core_equations(s) → Int

Count the total number of core (Module 1-8) residual equations.
"""
function n_core_equations(s::GTAPSets)
    nT=length(s.TRAD_COMM); nE=length(s.ENDW_COMM)
    nR=length(s.REG); nP=length(s.PROD_COMM); nM=length(s.MARG_COMM)
    nES=length(s.ENDWS_COMM); nEM=length(s.ENDWM_COMM)
    nNS=length(s.NSAV_COMM)
    nNMRG = nT - nM   # non-margin tradeable commodities
    return (
        # Module 1
        nR + nT*nR + nR + nT*nR + nT*nR + nT*nR + nT*nR + nT*nR + nR +
        # Module 2
        nR + nR + nR + nT*nR + nT*nR + nT*nR + nT*nR + nT*nR + nT*nR + nT*nR + nT*nR + nR +
        # Module 3
        nP*nR + nT*nP*nR + nT*nP*nR + nT*nP*nR + nT*nP*nR + nT*nP*nR + nT*nP*nR +
        nP*nR + nE*nP*nR + nEM*nP*nR + nES*nP*nR +
        nP*nR + nR + nR + nP*nR + nR +
        # Module 4
        nR + nR + nR + nR + nR + nR + nR + nR + nR + nR + 1 + 1 + nR +
        # Module 5
        nT*nR*nR + nR + nT*nR*nR + nT*nR + nT*nR + nT*nR*nR + nR +
        # Module 6
        nM*nT*nR*nR + nM + nM + nT*nR*nR + nM*nR + nT*nR*nR +
        # Module 7
        nE*nR + nR + nES*nR + nES*nP*nR + nR + nR + nR + nR +
        nR + nR + nR + nR + nR + nR + nR + nR +
        # Module 8
        nT*nR + nM*nR + nNMRG*nR + nT*nR + nEM*nR + nES*nP*nR +
        1 + 1 + 1
    )
end
