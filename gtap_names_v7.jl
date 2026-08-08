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
    M   = _t(s.MARG)

    Dict{Symbol, Vector{Vector{String}}}(
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
        :tpd         => [C, R],
        :tpm         => [C, R],
        :tgd         => [C, R],
        :tgm         => [C, R],
        :tp          => [R],
        :ao          => [A, R],
        :aint        => [A, R],
        :ava         => [A, R],
        :af          => [C, A, R],
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
        :qo_slack    => [C, R],
        :to_slack    => [C, A, R],
        :psave_slack => [R],
        # ── Endogenous variables that can appear in swap endo_out specs ────────
        :qxs         => [C, R, R],   # bilateral trade volumes (can be quota-fixed)
        :qo          => [A, R],       # activity output
        :psave       => [R],          # regional savings price
        :rore        => [R],          # expected rate of return
        :ke          => [R],          # expected capital stock growth
        :pe          => [EMS, R],     # economy-wide mobile/sluggish factor return
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
