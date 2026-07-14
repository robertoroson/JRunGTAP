# =============================================================================
# gtap_pack_v7.jl
# Pack / unpack for the GTAP v7 linearised system.
#
# The endogenous variable vector x_endo is a flat Float64 array whose layout
# is defined by _endo_specs_v7(s).  Each entry is (name::Symbol, shape::Tuple).
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Exogenous variable NamedTuple (all zeros in the benchmark / standard closure)
# ─────────────────────────────────────────────────────────────────────────────
function make_exog_zero_v7(s::GTAPSetsV7)
    nC=length(s.COMM); nA=length(s.ACTS); nR=length(s.REG)
    nE=length(s.ENDW); nM=length(s.MARG)
    nEM=length(s.ENDWM); nES=length(s.ENDWS); nEF=length(s.ENDWF)
    nEMS=length(s.ENDWMS); nEC=length(s.ENDWC)
    (
    # ── Technology (all exogenous in standard closure) ─────────────────────
    aosec   = zeros(nA),         # sector-neutral output tech
    aoreg   = zeros(nR),         # region-neutral output tech
    aoall   = zeros(nA, nR),     # all output tech
    avasec  = zeros(nA),
    avareg  = zeros(nR),
    avaall  = zeros(nA, nR),
    afcom   = zeros(nC),
    afsec   = zeros(nA),
    afreg   = zeros(nR),
    afall   = zeros(nC, nA, nR),
    afecom  = zeros(nE),
    afesec  = zeros(nA),
    afereg  = zeros(nR),
    afeall  = zeros(nE, nA, nR),
    aintsec = zeros(nA),
    aintreg = zeros(nR),
    aintall = zeros(nA, nR),
    au      = zeros(nR),         # preference shifter
    atm     = zeros(nM),         # transport tech (margin type)
    atf     = zeros(nC),         # transport tech (commodity)
    ats     = zeros(nR),         # transport tech (source region)
    atd     = zeros(nR),         # transport tech (destination)
    atall   = zeros(nM, nC, nR, nR),
    # ── Output taxes ──────────────────────────────────────────────────────
    to      = zeros(nC, nA, nR), # output tax (c,a,r)
    tinc    = zeros(nE, nA, nR), # income tax on endowments (e,a,r)
    tfe     = zeros(nE, nA, nR), # factor income tax (firm-level, e,a,r)
    # ── Input taxes ───────────────────────────────────────────────────────
    tfd     = zeros(nC, nA, nR), # dom. interm. tax
    tfm     = zeros(nC, nA, nR), # imp. interm. tax
    tpdall  = zeros(nC, nR),     # private dom. consumption tax
    tpmall  = zeros(nC, nR),     # private imp. consumption tax
    tpreg   = zeros(nR),         # aggregate private consumption tax
    tgd     = zeros(nC, nR),
    tgm     = zeros(nC, nR),
    tid     = zeros(nC, nR),
    tim     = zeros(nC, nR),
    # ── Trade taxes ───────────────────────────────────────────────────────
    tx      = zeros(nC, nR),
    txs     = zeros(nC, nR, nR),
    tm      = zeros(nC, nR),
    tms     = zeros(nC, nR, nR),
    ams     = zeros(nC, nR, nR), # bilateral import preference
    # ── Factor supply (qe = total endowment stock, fixed short-run) ─────
    qe      = zeros(nEMS, nR),      # total endowment supply (mobile+sluggish)
    qesf    = zeros(nEF, nA, nR),
    # ── Population / preference ───────────────────────────────────────────
    pop     = zeros(nR),
    dppriv  = zeros(nR),
    dpgov   = zeros(nR),
    dpsave  = zeros(nR),
    # ── Slack variables (zero in standard closure) ────────────────────────
    endwslack  = zeros(nEMS, nR),
    tradslack  = zeros(nC, nR),
    profitslack= zeros(nA, nR),
    incomeslack= zeros(nR),
    cgdslack   = zeros(nR),
    psaveslack = zeros(nR),
    walraslack = 0.0,
    # ── Numeraire ─────────────────────────────────────────────────────────
    pfactwld   = 0.0,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Endogenous variable layout specification
# Each entry: (name::Symbol, shape::Tuple{Int...})
# ─────────────────────────────────────────────────────────────────────────────
function _endo_specs_v7(s::GTAPSetsV7)
    nC=length(s.COMM); nA=length(s.ACTS); nR=length(s.REG)
    nE=length(s.ENDW); nM=length(s.MARG)
    nEM=length(s.ENDWM); nES=length(s.ENDWS); nEF=length(s.ENDWF)
    nEMS=length(s.ENDWMS); nEC=length(s.ENDWC)
    [
    # ── Module 1: Firms ───────────────────────────────────────────────────
    (:ao,    (nA, nR)),          # output aug. tech index
    (:ava,   (nA, nR)),          # VA aug. tech index
    (:afa,   (nC, nA, nR)),      # interm. aug. tech index
    (:afe,   (nE, nA, nR)),      # endowment aug. tech index
    (:aint,  (nA, nR)),          # interm. composite aug. tech
    (:qo,    (nA, nR)),          # activity output
    (:po,    (nA, nR)),          # average supply price of activity
    (:qint,  (nA, nR)),          # composite intermediate demand
    (:pint,  (nA, nR)),          # composite intermediate price
    (:qva,   (nA, nR)),          # value added composite
    (:pva,   (nA, nR)),          # VA price
    (:qfa,   (nC, nA, nR)),      # composite intermediate demand (c,a,r)
    (:pfa,   (nC, nA, nR)),      # composite intermediate price
    (:qfe,   (nE, nA, nR)),      # endowment demand by firm
    (:pfe,   (nE, nA, nR)),      # endowment demand price (firm)
    (:qfd,   (nC, nA, nR)),      # domestic intermediate demand
    (:pfd,   (nC, nA, nR)),      # domestic intermediate price (firm)
    (:qfm,   (nC, nA, nR)),      # imported intermediate demand
    (:pfm,   (nC, nA, nR)),      # imported intermediate price (firm)
    # ── Module 2: Commodity supply ────────────────────────────────────────
    (:qca,   (nC, nA, nR)),      # commodity c supplied by activity a
    (:pca,   (nC, nA, nR)),      # commodity-specific supply price
    (:ps,    (nC, nA, nR)),      # supplier price (tax-exclusive)
    (:pb,    (nA, nR)),          # basic price index for activity a
    (:qc,    (nC, nR)),          # total commodity supply
    (:pds,   (nC, nR)),          # domestic commodity price
    # ── Module 3: Income ──────────────────────────────────────────────────
    (:fincome,(nR,)),
    (:y,     (nR,)),
    (:del_indtaxr,(nR,)),        # tax revenue (simplified)
    # ── Module 4: Income allocation ───────────────────────────────────────
    (:yp,    (nR,)),
    (:yg,    (nR,)),
    (:qsave, (nR,)),
    (:psave, (nR,)),
    (:u,     (nR,)),
    (:up,    (nR,)),
    (:ug,    (nR,)),
    (:uelas, (nR,)),
    (:uepriv,(nR,)),
    (:p,     (nR,)),
    (:ppriv, (nR,)),
    (:pgov,  (nR,)),
    (:dpav,  (nR,)),
    (:dpsum, (nR,)),
    # ── Module 5: Final demand ────────────────────────────────────────────
    (:qpa,   (nC, nR)),
    (:ppa,   (nC, nR)),
    (:qpd,   (nC, nR)),
    (:ppd,   (nC, nR)),
    (:qpm,   (nC, nR)),
    (:ppm,   (nC, nR)),
    (:tpd,   (nC, nR)),          # effective private dom. tax
    (:tpm,   (nC, nR)),          # effective private imp. tax
    (:qga,   (nC, nR)),
    (:pga,   (nC, nR)),
    (:qgd,   (nC, nR)),
    (:pgd,   (nC, nR)),
    (:qgm,   (nC, nR)),
    (:pgm,   (nC, nR)),
    (:qia,   (nC, nR)),
    (:pia,   (nC, nR)),
    (:qid,   (nC, nR)),
    (:pid,   (nC, nR)),
    (:qim,   (nC, nR)),
    (:pim,   (nC, nR)),
    (:pinv,  (nR,)),
    # ── Module 6: Trade ───────────────────────────────────────────────────
    (:qms,   (nC, nR)),
    (:pms,   (nC, nR)),
    (:qxs,   (nC, nR, nR)),
    (:pfob,  (nC, nR, nR)),
    (:pcif,  (nC, nR, nR)),
    (:pmds,  (nC, nR, nR)),
    (:ptrans,(nC, nR, nR)),
    (:pr,    (nC, nR)),
    (:qds,   (nC, nR)),
    (:qtmfsd,(nM, nC, nR, nR)),
    (:atmfsd,(nM, nC, nR, nR)),
    (:qtm,   (nM,)),
    (:pt,    (nM,)),
    (:qst,   (nM, nR)),
    # ── Module 7: Endowments ─────────────────────────────────────────────
    (:pe,    (nEMS, nR)),        # composite endowment price (mobile+sluggish)
    (:pes,   (nE, nA, nR)),      # endowment supply price by activity
    (:peb,   (nE, nA, nR)),      # basic endowment price by activity
    (:qes,   (nE, nA, nR)),      # endowment supply by activity
    (:rental,(nR,)),
    # ── Module 8: Capital ─────────────────────────────────────────────────
    (:ke,    (nR,)),
    (:kb,    (nR,)),
    (:rorc,  (nR,)),
    (:rore,  (nR,)),
    (:qinv,  (nR,)),
    (:expand,(nEC, nR)),
    (:globalcgds, (1,)),
    (:rorg,       (1,)),
    (:walras_sup, (1,)),
    (:walras_dem, (1,)),
    (:pcgdswld,   (1,)),
    (:pfactor,(nR,)),
    ]
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility: total number of endogenous variables
# ─────────────────────────────────────────────────────────────────────────────
function n_endog_v7(s::GTAPSetsV7)
    sum(prod(sh) for (_, sh) in _endo_specs_v7(s))
end

# ─────────────────────────────────────────────────────────────────────────────
# Compute offset dict for each variable name → start index in x_endo (1-based)
# ─────────────────────────────────────────────────────────────────────────────
function endo_offsets_v7(s::GTAPSetsV7)
    off = Dict{Symbol,Int}()
    cursor = 1
    for (nm, sh) in _endo_specs_v7(s)
        off[nm] = cursor
        cursor  += prod(sh)
    end
    off
end

# ─────────────────────────────────────────────────────────────────────────────
# Unpack flat vector x_endo into a NamedTuple of arrays + exogenous variables
# ─────────────────────────────────────────────────────────────────────────────
function unpack_endo_v7(x_endo::AbstractVector, exog, s::GTAPSetsV7)
    specs = _endo_specs_v7(s)
    off   = endo_offsets_v7(s)
    parts = []
    for (nm, sh) in specs
        i0 = off[nm]
        n  = prod(sh)
        push!(parts, nm => reshape(x_endo[i0:i0+n-1], sh))
    end
    endo_nt = NamedTuple(parts)
    # merge endogenous + exogenous (endo takes precedence for shared names)
    return merge(exog, endo_nt)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pack residuals from Dict{Symbol,Any} into a flat vector
# (The residual vector must match the endogenous variable ordering)
# ─────────────────────────────────────────────────────────────────────────────

# Map equation name → residual shape (mirrors _endo_specs_v7 structure)
function core_eq_sizes_v7(s::GTAPSetsV7)
    nC=length(s.COMM); nA=length(s.ACTS); nR=length(s.REG)
    nE=length(s.ENDW); nM=length(s.MARG)
    nEM=length(s.ENDWM); nES=length(s.ENDWS); nEF=length(s.ENDWF)
    nEMS=length(s.ENDWMS); nEC=length(s.ENDWC)
    Dict(
        :E_ao         => (nA, nR),
        :E_ava        => (nA, nR),
        :E_afa        => (nC, nA, nR),
        :E_afe        => (nE, nA, nR),
        :E_aint       => (nA, nR),
        :E_qint       => (nA, nR),
        :E_qva        => (nA, nR),
        :E_qfa        => (nC, nA, nR),
        :E_pint       => (nA, nR),
        :E_qfe        => (nE, nA, nR),
        :E_pva        => (nA, nR),
        :E_qfd        => (nC, nA, nR),
        :E_qfm        => (nC, nA, nR),
        :E_pfa        => (nC, nA, nR),
        :E_pfd        => (nC, nA, nR),
        :E_pfm        => (nC, nA, nR),
        :E_qo_zero    => (nA, nR),
        :E_qca        => (nC, nA, nR),
        :E_po         => (nA, nR),
        :E_ps         => (nC, nA, nR),
        :E_pb         => (nA, nR),
        :E_pca        => (nC, nA, nR),
        :E_qc         => (nC, nR),
        :E_fincome    => (nR,),
        :E_y          => (nR,),
        :E_del_indtaxr=> (nR,),
        :E_qsave      => (nR,),
        :E_yg         => (nR,),
        :E_yp         => (nR,),
        :E_uelas      => (nR,),
        :E_dpav       => (nR,),
        :E_p          => (nR,),
        :E_u          => (nR,),
        :E_dpsum      => (nR,),
        :E_qpa        => (nC, nR),
        :E_uepriv     => (nR,),
        :E_ppriv      => (nR,),
        :E_up         => (nR,),
        :E_qpd        => (nC, nR),
        :E_qpm        => (nC, nR),
        :E_ppa        => (nC, nR),
        :E_ppd        => (nC, nR),
        :E_ppm        => (nC, nR),
        :E_tpd        => (nC, nR),
        :E_tpm        => (nC, nR),
        :E_qga        => (nC, nR),
        :E_pgov       => (nR,),
        :E_ug         => (nR,),
        :E_qgd        => (nC, nR),
        :E_qgm        => (nC, nR),
        :E_pga        => (nC, nR),
        :E_pgd        => (nC, nR),
        :E_pgm        => (nC, nR),
        :E_qia        => (nC, nR),
        :E_pinv       => (nR,),
        :E_qid        => (nC, nR),
        :E_qim        => (nC, nR),
        :E_pia        => (nC, nR),
        :E_pid        => (nC, nR),
        :E_pim        => (nC, nR),
        :E_qms        => (nC, nR),
        :E_qxs        => (nC, nR, nR),
        :E_pms        => (nC, nR),
        :E_atmfsd     => (nM, nC, nR, nR),
        :E_qtmfsd     => (nM, nC, nR, nR),
        :E_qtm        => (nM,),
        :E_pt         => (nM,),
        :E_qst        => (nM, nR),
        :E_ptrans     => (nC, nR, nR),
        :E_pfob       => (nC, nR, nR),
        :E_pcif       => (nC, nR, nR),
        :E_pmds       => (nC, nR, nR),
        :E_pr         => (nC, nR),
        :E_qds        => (nC, nR),
        :E_pds        => (nC, nR),
        :E_pe1        => (nEM, nR),
        :E_qes1       => (nEM, nA, nR),
        :E_qes2       => (nES, nA, nR),
        :E_pe2        => (nES, nR),
        :E_qes3       => (nEF, nA, nR),
        :E_pfe        => (nE, nA, nR),
        :E_pes_link   => (nE, nA, nR),
        :E_peb        => (nE, nA, nR),
        :E_rental     => (nR,),
        :E_ke         => (nR,),
        :E_kb         => (nR,),
        :E_rorc       => (nR,),
        :E_rore       => (nR,),
        :E_qinv       => (nR,),
        :E_expand     => (nEC, nR),
        :E_globalcgds => (1,),
        :E_psave      => (nR,),
        :E_pcgdswld   => (1,),
        :E_pfactor    => (nR,),
        :E_pfactwld   => (1,),
        :E_walras_sup => (1,),
        :E_walras_dem => (1,),
        :E_walras     => (1,),
    )
end

# Ordered list of equations packed into the residual vector (must cover all
# endogenous variables — exactly one equation per degree of freedom).
const CORE_EQ_NAMES_V7 = [
    :E_ao, :E_ava, :E_afa, :E_afe, :E_aint,
    :E_qint, :E_qva, :E_qfa, :E_pint, :E_qfe, :E_pva,
    :E_qfd, :E_qfm, :E_pfa, :E_pfd, :E_pfm,
    :E_qo_zero,
    :E_qca, :E_po, :E_ps, :E_pb, :E_pca, :E_qc,
    :E_fincome, :E_y, :E_del_indtaxr,
    :E_qsave, :E_yg, :E_yp, :E_uelas, :E_dpav, :E_p, :E_u, :E_dpsum,
    :E_qpa, :E_uepriv, :E_ppriv, :E_up,
    :E_qpd, :E_qpm, :E_ppa, :E_ppd, :E_ppm, :E_tpd, :E_tpm,
    :E_qga, :E_pgov, :E_ug, :E_qgd, :E_qgm, :E_pga, :E_pgd, :E_pgm,
    :E_qia, :E_pinv, :E_qid, :E_qim, :E_pia, :E_pid, :E_pim,
    :E_qms, :E_pms, :E_qxs, :E_pfob, :E_pcif, :E_pmds, :E_ptrans, :E_pr, :E_qds, :E_pds,
    :E_atmfsd, :E_qtmfsd, :E_qtm, :E_pt, :E_qst,
    :E_pe1, :E_qes1, :E_qes2, :E_pe2, :E_qes3,
    :E_pfe, :E_pes_link, :E_peb,
    :E_rental, :E_ke, :E_kb, :E_rorc, :E_rore, :E_qinv, :E_expand,
    :E_globalcgds, :E_psave, :E_pcgdswld,
    :E_pfactor, :E_pfactwld,
    :E_walras_sup, :E_walras_dem, :E_walras,
]

function pack_residuals_core_v7(R::Dict{Symbol,Any}, s::GTAPSetsV7)
    sz = core_eq_sizes_v7(s)
    n  = sum(prod(sz[nm]) for nm in CORE_EQ_NAMES_V7)
    b  = zeros(n)
    cursor = 1
    for nm in CORE_EQ_NAMES_V7
        haskey(R, nm) || error("Missing equation $nm in residual dict")
        v  = R[nm]
        np = prod(sz[nm])
        b[cursor:cursor+np-1] = vec(v)
        cursor += np
    end
    b
end

# ─────────────────────────────────────────────────────────────────────────────
# Combined residual function (for solver and Jacobian)
# ─────────────────────────────────────────────────────────────────────────────
function F_core_v7(x_endo::AbstractVector, exog, d::GTAPDataV7, s::GTAPSetsV7, C)
    v = unpack_endo_v7(x_endo, exog, s)
    R = gtap_v7_residuals(v, d, s, C)
    pack_residuals_core_v7(R, s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Extract a named variable from the solution vector
# ─────────────────────────────────────────────────────────────────────────────
function get_variable_v7(x_endo::AbstractVector, name::Symbol, s::GTAPSetsV7)
    specs = _endo_specs_v7(s)
    off   = endo_offsets_v7(s)
    haskey(off, name) || error("Variable $name not found in v7 endogenous specs")
    sh = Dict(nm => sh for (nm, sh) in specs)[name]
    i0 = off[name]
    n  = prod(sh)
    reshape(x_endo[i0:i0+n-1], sh)
end
