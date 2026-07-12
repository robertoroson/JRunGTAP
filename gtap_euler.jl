# gtap_euler.jl
# ─────────────────────────────────────────────────────────────────────────────
# Level-update for multi-step Euler integration.
#
# After each linearization sub-step we update every value flow in GTAPData by
#   V_new = V_old × (1 + (dP + dQ) / 100)
# where dP and dQ are the price and quantity % changes for that flow, taken
# from the sub-step solution vector dx.  Elasticities and fixed parameters are
# left unchanged — only the shares (which are ratios of value flows) evolve.
#
# The cumulative result across steps is computed using the product rule:
#   x_level[i] = ∏_k (1 + dx_k[i] / 100)
# converted back to % change as 100 × (x_level − 1).  This is more accurate
# than simple summation for large shocks.
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_data_euler(d, dx, s) → GTAPData

Scale all value flows in `d` by their sub-step price+quantity % changes in `dx`.
Elasticities, CDE parameters, VKB, VDEP, and SLUG are held constant.
"""
function update_data_euler(d::GTAPData, dx::Vector{Float64}, s::GTAPSets)

    nT = length(s.TRAD_COMM)
    nE = length(s.ENDW_COMM)
    nR = length(s.REG)
    nP = length(s.PROD_COMM)
    nM = length(s.MARG_COMM)

    # ── Extract % change variables from sub-step solution ──────────────────
    ps_all = get_variable(dx, :ps,     s)   # (nNS, nR)  supply prices
    pfob_  = get_variable(dx, :pfob,   s)   # (nT, nR, nR)
    pcif_  = get_variable(dx, :pcif,   s)   # (nT, nR, nR)
    pms_   = get_variable(dx, :pms,    s)   # (nT, nR, nR)
    qxs_   = get_variable(dx, :qxs,    s)   # (nT, nR, nR)
    pfe_   = get_variable(dx, :pfe,    s)   # (nE, nP, nR)
    qfe_   = get_variable(dx, :qfe,    s)   # (nE, nP, nR)
    pfd_   = get_variable(dx, :pfd,    s)   # (nT, nP, nR)
    qfd_   = get_variable(dx, :qfd,    s)   # (nT, nP, nR)
    pfm_   = get_variable(dx, :pfm,    s)   # (nT, nP, nR)
    qfm_   = get_variable(dx, :qfm,    s)   # (nT, nP, nR)
    ppd_   = get_variable(dx, :ppd,    s)   # (nT, nR)
    qpd_   = get_variable(dx, :qpd,    s)   # (nT, nR)
    ppm_   = get_variable(dx, :ppm,    s)   # (nT, nR)
    qpm_   = get_variable(dx, :qpm,    s)   # (nT, nR)
    pgd_   = get_variable(dx, :pgd,    s)   # (nT, nR)
    qgd_   = get_variable(dx, :qgd,    s)   # (nT, nR)
    pgm_   = get_variable(dx, :pgm,    s)   # (nT, nR)
    qgm_   = get_variable(dx, :qgm,    s)   # (nT, nR)
    psave_ = get_variable(dx, :psave,  s)   # (nR,)
    qsave_ = get_variable(dx, :qsave,  s)   # (nR,)
    pt_    = get_variable(dx, :pt,     s)   # (nM,)
    qst_   = get_variable(dx, :qst,    s)   # (nM, nR)
    ptrans_= get_variable(dx, :ptrans, s)   # (nT, nR, nR)
    qtmfsd_= get_variable(dx, :qtmfsd, s)  # (nM, nT, nR, nR)

    # Domestic supply price for traded goods: ps_all rows nE+1 … nE+nT
    pds = ps_all[nE+1:nE+nT, :]   # (nT, nR)

    # Shorthand: scale array V by (1 + (dP + dQ)/100) element-wise
    upd(V, dP, dQ) = V .* (1 .+ (dP .+ dQ) ./ 100)

    # ── Trade flows ─────────────────────────────────────────────────────────
    # Physical quantity = qxs[i,r,s]; priced differently per flow type
    new_VXMD = [d.VXMD[i,r,ss] * (1 + (pds[i,r]       + qxs_[i,r,ss]) / 100)
                for i in 1:nT, r in 1:nR, ss in 1:nR]
    new_VXWD = [d.VXWD[i,r,ss] * (1 + (pfob_[i,r,ss]  + qxs_[i,r,ss]) / 100)
                for i in 1:nT, r in 1:nR, ss in 1:nR]
    new_VIWS = [d.VIWS[i,r,ss] * (1 + (pcif_[i,r,ss]  + qxs_[i,r,ss]) / 100)
                for i in 1:nT, r in 1:nR, ss in 1:nR]
    new_VIMS = [d.VIMS[i,r,ss] * (1 + (pms_[i,r,ss]   + qxs_[i,r,ss]) / 100)
                for i in 1:nT, r in 1:nR, ss in 1:nR]

    # ── Factor payments ──────────────────────────────────────────────────────
    new_VFM  = upd(d.VFM,  pfe_, qfe_)   # (nE, nP, nR) at market prices
    new_EVFA = upd(d.EVFA, pfe_, qfe_)   # (nE, nP, nR) at agent prices
    # EVOA[e,r] = total endowment income = Σ_j VFM[e,j,r]
    new_EVOA = dropdims(sum(new_VFM, dims=2), dims=2)   # (nE, nR)

    # ── Intermediate inputs ──────────────────────────────────────────────────
    new_VDFM = upd(d.VDFM, pfd_, qfd_)   # (nT, nP, nR)
    new_VDFA = upd(d.VDFA, pfd_, qfd_)
    new_VIFM = upd(d.VIFM, pfm_, qfm_)
    new_VIFA = upd(d.VIFA, pfm_, qfm_)

    # ── Final demand ─────────────────────────────────────────────────────────
    new_VDPM = upd(d.VDPM, ppd_, qpd_)   # (nT, nR)
    new_VDPA = upd(d.VDPA, ppd_, qpd_)
    new_VIPM = upd(d.VIPM, ppm_, qpm_)
    new_VIPA = upd(d.VIPA, ppm_, qpm_)
    new_VDGM = upd(d.VDGM, pgd_, qgd_)
    new_VDGA = upd(d.VDGA, pgd_, qgd_)
    new_VIGM = upd(d.VIGM, pgm_, qgm_)
    new_VIGA = upd(d.VIGA, pgm_, qgm_)

    # ── Savings ──────────────────────────────────────────────────────────────
    new_SAVE = d.SAVE .* (1 .+ (psave_ .+ qsave_) ./ 100)

    # ── Transport margins ────────────────────────────────────────────────────
    new_VST    = upd(d.VST, pt_,  qst_)   # (nM, nR)
    new_VTMFSD = [d.VTMFSD[m,i,r,ss] * (1 + (ptrans_[i,r,ss] + qtmfsd_[m,i,r,ss]) / 100)
                  for m in 1:nM, i in 1:nT, r in 1:nR, ss in 1:nR]

    GTAPData(
        new_SAVE,
        new_VDGA, new_VDGM, new_VIGA, new_VIGM,
        new_VDPA, new_VDPM, new_VIPA, new_VIPM,
        new_EVOA,
        new_EVFA,
        new_VDFA, new_VDFM, new_VIFA, new_VIFM,
        new_VFM,
        new_VXMD, new_VXWD, new_VIWS, new_VIMS,
        new_VST,
        new_VTMFSD,
        d.VKB,          # capital stock: fixed in comparative statics
        d.VDEP,         # depreciation: fixed
        d.DPARSUM,      # CDE distribution params: fixed
        d.ESUBD, d.ESUBT, d.ESUBVA, d.ESUBM,
        d.ETRAE,
        d.INCPAR, d.SUBPAR,
        d.RORFLEX,
        d.RORDELTA,
        d.SLUG,
    )
end
