# =============================================================================
# gtap_names_v7.jl
# GTAPv7 overloads for var_domains, _model_sets, and resolve_config.
# Included after gtap_v7.jl so GTAPSetsV7 is already defined.
# =============================================================================

function var_domains(s::GTAPSetsV7; lowercase::Bool = true)
    _t(v) = lowercase ? Base.lowercase.(v) : v
    C   = _t(s.COMM)
    A   = _t(s.ACTS)
    R   = _t(s.REG)
    E   = _t(s.ENDW)
    EF  = _t(s.ENDWF)
    EMS = _t(s.ENDWMS)
    EC  = _t(s.ENDWC)
    M   = _t(s.MARG)

    Dict{Symbol, Vector{Vector{String}}}(
        # ── Policy instruments (exogenous in standard closure) ─────────────────
        :tm          => [C, R],
        :tms         => [C, R, R],
        :tx          => [C, R],
        :txs         => [C, R, R],
        :to          => [C, A, R],
        :tfd         => [C, A, R],
        :tfm         => [C, A, R],
        :tinc        => [E, A, R],
        :tfe         => [E, A, R],
        :tf          => [E, A, R],
        :tgd         => [C, R],
        :tgm         => [C, R],
        :tid         => [C, R],
        :tim         => [C, R],
        :tp          => [R],
        :tpreg       => [R],
        # ── Shifters / exogenous demand ────────────────────────────────────────
        :ao          => [A, R],
        :aint        => [A, R],
        :ava         => [A, R],
        :af          => [C, A, R],
        :afa         => [C, A, R],
        :afe         => [E, A, R],
        :ams         => [C, R, R],
        :atmfsd      => [M, C, R, R],
        :atm         => [M],
        :atf         => [C],
        :ats         => [R],
        :atd         => [R],
        :pop         => [R],
        :pfactwld    => [String[]],
        :qe          => [EMS, R],
        :qesf        => [EF, A, R],
        :dpgov       => [R],
        :dpsave      => [R],
        :dppriv      => [R],
        # ── Firm-level quantities and prices ───────────────────────────────────
        :qo          => [A, R],
        :po          => [A, R],
        :qint        => [A, R],
        :pint        => [A, R],
        :qva         => [A, R],
        :pva         => [A, R],
        :pb          => [A, R],
        :qfa         => [C, A, R],
        :pfa         => [C, A, R],
        :qfe         => [E, A, R],
        :pfe         => [E, A, R],
        :qfd         => [C, A, R],
        :pfd         => [C, A, R],
        :qfm         => [C, A, R],
        :pfm         => [C, A, R],
        # ── Commodity supply ──────────────────────────────────────────────────
        :qca         => [C, A, R],
        :pca         => [C, A, R],
        :ps          => [C, A, R],
        :qc          => [C, R],
        :pds         => [C, R],
        # ── Income and welfare ────────────────────────────────────────────────
        :y           => [R],
        :yp          => [R],
        :yg          => [R],
        :del_indtaxr => [R],
        :u           => [R],
        :up          => [R],
        :ug          => [R],
        :uelas       => [R],
        :uepriv      => [R],
        :p           => [R],
        :ppriv       => [R],
        :pgov        => [R],
        :dpav        => [R],
        :dpsum       => [R],
        :pfactor     => [R],
        # ── Final demand: private ──────────────────────────────────────────────
        :qpa         => [C, R],
        :ppa         => [C, R],
        :qpd         => [C, R],
        :ppd         => [C, R],
        :qpm         => [C, R],
        :ppm         => [C, R],
        :tpd         => [C, R],
        :tpm         => [C, R],
        # ── Final demand: government ───────────────────────────────────────────
        :qga         => [C, R],
        :pga         => [C, R],
        :qgd         => [C, R],
        :pgd         => [C, R],
        :qgm         => [C, R],
        :pgm         => [C, R],
        # ── Final demand: investment ───────────────────────────────────────────
        :qia         => [C, R],
        :pia         => [C, R],
        :qid         => [C, R],
        :pid         => [C, R],
        :qim         => [C, R],
        :pim         => [C, R],
        :pinv        => [R],
        :qinv        => [R],
        :qsave       => [R],
        :psave       => [R],
        # ── Trade ─────────────────────────────────────────────────────────────
        :qms         => [C, R],
        :pms         => [C, R],
        :qxs         => [C, R, R],
        :pfob        => [C, R, R],
        :pcif        => [C, R, R],
        :pmds        => [C, R, R],
        :ptrans      => [C, R, R],
        :pr          => [C, R],
        :qds         => [C, R],
        :qtmfsd      => [M, C, R, R],
        :qtm         => [M],
        :pt          => [M],
        :qst         => [M, R],
        # ── Endowments ────────────────────────────────────────────────────────
        :pe          => [EMS, R],
        :pes         => [E, A, R],
        :peb         => [E, A, R],
        :qes         => [E, A, R],
        :rental      => [R],
        # ── Capital dynamics ──────────────────────────────────────────────────
        :ke          => [R],
        :kb          => [R],
        :rorc        => [R],
        :rore        => [R],
        :expand      => [EC, R],
        # ── Swap/closure helpers ──────────────────────────────────────────────
        :qo_slack    => [C, R],
        :to_slack    => [C, A, R],
        :psave_slack => [R],
        # ── Derived / reporting aggregates (computed by gtap_derived_v7) ──────
        :pxw         => [C, R],
        :qxw         => [C, R],
        :vxwfob      => [C, R],
        :pxwreg      => [R],
        :qxwreg      => [R],
        :vxwreg      => [R],
        :pxwcom      => [C],
        :qxwcom      => [C],
        :vxwcom      => [C],
        :pmw         => [C, R],
        :qmw         => [C, R],
        :vmwcif      => [C, R],
        :pmwreg      => [R],
        :qmwreg      => [R],
        :vmwreg      => [R],
        :pmwcom      => [C],
        :qmwcom      => [C],
        :vmwcom      => [C],
        :pw          => [C],
        :qow         => [C],
        :valuew      => [C],
        :del_tbal    => [R],
        :del_tbalc   => [C, R],
        :del_tbalry  => [R],
        :psw         => [R],
        :pdw         => [R],
        :tot         => [R],
        :vgdp        => [R],
        :pgdp        => [R],
        :qgdp        => [R],
        :pfactreal   => [E, A, R],
        :pebfactreal => [EMS, R],
        :compvalad   => [A, R],
        :EV          => [R],
        :ueprivev    => [R],
        :yev         => [R],
    )
end

function _model_sets(s::GTAPSetsV7)
    Dict{String,Vector{String}}(
        "comm"      => lowercase.(s.COMM),
        "acts"      => lowercase.(s.ACTS),
        "reg"       => lowercase.(s.REG),
        "endw"      => lowercase.(s.ENDW),
        "endwm"     => lowercase.(s.ENDWM),
        "endws"     => lowercase.(s.ENDWS),
        "endwf"     => lowercase.(s.ENDWF),
        "endwms"    => lowercase.(s.ENDWMS),
        "endwc"     => lowercase.(s.ENDWC),
        "marg"      => lowercase.(s.MARG),
        "nmrg"      => lowercase.(s.NMRG),
        "demd"      => lowercase.(s.DEMD),
        "trad_comm" => lowercase.(s.COMM),
        "prod_comm" => lowercase.(s.ACTS),
        "marg_comm" => lowercase.(s.MARG),
        "endw_comm" => lowercase.(s.ENDW),
    )
end

var_domains_orig(s::GTAPSetsV7) = var_domains(s; lowercase = false)

function resolve_config(raw_shocks,
                        raw_swaps,
                        user_sets ::Dict{String,Vector{String}},
                        s         ::GTAPSetsV7)
    domains     = var_domains(s)
    normed_user = Dict{String,Vector{String}}(
        lowercase(k) => lowercase.(v) for (k, v) in user_sets)
    known_sets  = merge(_model_sets(s), normed_user)

    shocks = Dict{String,Float64}()
    for (spec, val) in raw_shocks
        for expanded in expand_varspec(spec, domains, known_sets)
            shocks[expanded] = val
        end
    end

    swaps = Swap[]
    for (out_spec, in_spec, fix_at) in raw_swaps
        outs = expand_varspec(out_spec, domains, known_sets)
        ins  = expand_varspec(in_spec,  domains, known_sets)
        length(outs) == length(ins) ||
            error("Swap '$out_spec' <-> '$in_spec' expands to different counts: " *
                  "$(length(outs)) endo vs $(length(ins)) exog elements")
        for (o, i) in zip(outs, ins)
            push!(swaps, Swap(o, i, fix_at))
        end
    end

    return shocks, swaps
end
