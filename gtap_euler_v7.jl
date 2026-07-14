# =============================================================================
# gtap_euler_v7.jl
# Level-data update for multi-step Euler / Gragg integration of GTAPv7.
#
# After each sub-step the % changes are applied to all value flows so that the
# Jacobian rebuilt at the next step reflects the updated economy.
# =============================================================================

"""
    update_data_euler_v7(d, dx, s) → GTAPDataV7

Apply the sub-step solution `dx` (NamedTuple indexed by variable name) to the
level-value arrays in `d` and return an updated GTAPDataV7.

All value flows V are updated as  V_new = V_old × (1 + (dP + dQ) / 100)
where dP and dQ are the relevant price and quantity % changes.
"""
function update_data_euler_v7(d::GTAPDataV7, dx, s::GTAPSetsV7)
    nC = length(s.COMM); nA = length(s.ACTS); nR = length(s.REG)
    nE = length(s.ENDW); nM = length(s.MARG)
    nEM= length(s.ENDWM); nES= length(s.ENDWS)
    nEMS= length(s.ENDWMS); nEC = length(s.ENDWC)

    iE(x)  = findfirst(==(x), s.ENDW)::Int
    iEMS(x)= findfirst(==(x), s.ENDWMS)::Int

    scl(dP, dQ) = (1.0 + (dP + dQ) / 100.0)

    # ── Make matrix ────────────────────────────────────────────────────────
    MAKES_new = copy(d.MAKES)
    MAKEB_new = copy(d.MAKEB)
    for c in 1:nC, a in 1:nA, r in 1:nR
        MAKES_new[c,a,r] *= scl(dx.ps[c,a,r],  dx.qca[c,a,r])
        MAKEB_new[c,a,r] *= scl(dx.pca[c,a,r], dx.qca[c,a,r])
    end

    # ── Endowment flows ────────────────────────────────────────────────────
    EVOS_new = copy(d.EVOS)
    EVFB_new = copy(d.EVFB)
    EVFP_new = copy(d.EVFP)
    for ei in 1:nE, a in 1:nA, r in 1:nR
        eiMS  = findfirst(==(s.ENDW[ei]), s.ENDWMS)
        dPsup = eiMS !== nothing ? dx.pes[ei,a,r] : dx.pes[ei,a,r]  # supply price
        EVOS_new[ei,a,r] *= scl(dPsup,       dx.qes[ei,a,r])
        EVFB_new[ei,a,r] *= scl(dx.peb[ei,a,r], dx.qfe[ei,a,r])
        EVFP_new[ei,a,r] *= scl(dx.pfe[ei,a,r], dx.qfe[ei,a,r])
    end

    # ── Intermediate demand ────────────────────────────────────────────────
    VDFB_new = copy(d.VDFB)
    VMFB_new = copy(d.VMFB)
    VMFP_new = copy(d.VMFP)
    for c in 1:nC, a in 1:nA, r in 1:nR
        VDFB_new[c,a,r] *= scl(dx.pfd[c,a,r], dx.qfd[c,a,r])
        VMFB_new[c,a,r] *= scl(dx.pfm[c,a,r], dx.qfm[c,a,r])
        VMFP_new[c,a,r] *= scl(dx.pfm[c,a,r], dx.qfm[c,a,r])
    end

    # ── Household consumption ──────────────────────────────────────────────
    VDPB_new = d.VDPB .* scl.(dx.ppd, dx.qpd)
    VMPB_new = d.VMPB .* scl.(dx.ppm, dx.qpm)
    VDPP_new = d.VDPP .* scl.(dx.ppd, dx.qpd)
    VMPP_new = d.VMPP .* scl.(dx.ppm, dx.qpm)

    # ── Government consumption ─────────────────────────────────────────────
    VDGB_new = d.VDGB .* scl.(dx.pgd, dx.qgd)
    VMGB_new = d.VMGB .* scl.(dx.pgm, dx.qgm)
    VDGP_new = d.VDGP .* scl.(dx.pgd, dx.qgd)
    VMGP_new = d.VMGP .* scl.(dx.pgm, dx.qgm)

    # ── Investment ─────────────────────────────────────────────────────────
    VDIB_new = d.VDIB .* scl.(dx.pid, dx.qid)
    VMIB_new = d.VMIB .* scl.(dx.pim, dx.qim)
    VDIP_new = d.VDIP .* scl.(dx.pid, dx.qid)
    VMIP_new = d.VMIP .* scl.(dx.pim, dx.qim)

    # ── Trade flows ────────────────────────────────────────────────────────
    VXSB_new = copy(d.VXSB)
    VFOB_new = copy(d.VFOB)
    VCIF_new = copy(d.VCIF)
    VMSB_new = copy(d.VMSB)
    for c in 1:nC, r in 1:nR, dd in 1:nR
        VXSB_new[c,r,dd] *= scl(dx.pds[c,r],   dx.qxs[c,r,dd])
        VFOB_new[c,r,dd] *= scl(dx.pfob[c,r,dd], dx.qxs[c,r,dd])
        VCIF_new[c,r,dd] *= scl(dx.pcif[c,r,dd], dx.qxs[c,r,dd])
        VMSB_new[c,r,dd] *= scl(dx.pmds[c,r,dd], dx.qxs[c,r,dd])
    end

    # ── International transport ────────────────────────────────────────────
    VST_new    = copy(d.VST)
    VTMFSD_new = copy(d.VTMFSD)
    for mi in 1:nM, r in 1:nR
        VST_new[mi,r] *= scl(dx.pds[findfirst(==(s.MARG[mi]), s.COMM),r], dx.qst[mi,r])
    end
    for mi in 1:nM, c in 1:nC, r in 1:nR, dd in 1:nR
        VTMFSD_new[mi,c,r,dd] *= scl(dx.pt[mi], dx.qtmfsd[mi,c,r,dd])
    end

    # ── Saving and capital ────────────────────────────────────────────────
    SAVE_new  = d.SAVE  .* scl.(dx.psave, dx.qsave)
    VKB_new   = d.VKB   .* (1.0 .+ dx.kb   ./ 100.0)
    VDEP_new  = d.VDEP  .* (1.0 .+ dx.pinv ./ 100.0)

    return GTAPDataV7(
        SAVE_new, VDPB_new, VMPB_new, VDPP_new, VMPP_new,
        VDGB_new, VMGB_new, VDGP_new, VMGP_new,
        VDIB_new, VMIB_new, VDIP_new, VMIP_new,
        MAKES_new, MAKEB_new,
        EVOS_new, EVFB_new, EVFP_new,
        VDFB_new, VMFB_new, VMFP_new,
        VXSB_new, VFOB_new, VCIF_new, VMSB_new,
        VST_new, VTMFSD_new,
        VKB_new, VDEP_new, d.DPARSUM,
        d.ESUBT, d.ESUBC, d.ESUBVA, d.ETRAQ, d.ESUBQ,
        d.ESUBD, d.ESUBM, d.ESUBG, d.ESUBS, d.ETRAE,
        d.INCPAR, d.SUBPAR, d.RORFLEX, d.RORDELTA, d.ENDOWFLAG,
    )
end
