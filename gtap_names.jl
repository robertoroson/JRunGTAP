# gtap_names.jl
# ─────────────────────────────────────────────────────────────────────────────
# Name-to-index resolution for GTAP variables.
#
# Allows shocks and swaps to use commodity/region names instead of numbers, and
# to span entire sets (model sets or user-defined inline sets), e.g.:
#
#   shock  tm[pdr, aut]       = 10.0   # single named element
#   shock  tm[agri, aut]      = 5.0    # user set 'agri' × 'aut'
#   shock  tm[pdr, all]       = 10.0   # pdr into every region
#   shock  tm[TRAD_COMM, aut] = 3.0    # all commodities into Austria
#   swap   rental[aut] <-> qo_endw[Land, aut]
#
# Index tokens are resolved in this priority order:
#   1. "all" or "*"  → every element of this dimension's set
#   2. Model-set name (e.g. "TRAD_COMM", "REG") → all elements of that set
#   3. User-defined set (from 'set name = [...]' in the config file)
#   4. Element name (case-insensitive match in the dimension's set)
#   5. Integer literal (1-based, for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

"""
    var_domains(s; lowercase=true) → Dict{Symbol, Vector{Vector{String}}}

Maps each variable name to the ordered list of set-element vectors that index its
dimensions.

- `lowercase=true`  (default): names are lowercased for case-insensitive matching.
- `lowercase=false`: names retain original case from the HAR data, for display use.

Use `var_domains_orig(s)` as a convenience alias for the display form.
"""
function var_domains(s::GTAPSets; lowercase::Bool = true)
    _t(v) = lowercase ? Base.lowercase.(v) : v
    T  = _t(s.TRAD_COMM)
    E  = _t(s.ENDW_COMM)
    ES = _t(s.ENDWS_COMM)
    R  = _t(s.REG)
    P  = _t(s.PROD_COMM)
    M  = _t(s.MARG_COMM)
    NS = _t(s.NSAV_COMM)

    Dict{Symbol, Vector{Vector{String}}}(
        # ── Exogenous variables (commonly shocked) ──────────────────────────
        :tm       => [T, R],            # import tariff
        :tms      => [T, R, R],         # source-specific tariff  (origin, dest)
        :tx       => [T, R],            # export tax
        :txs      => [T, R, R],         # source-specific export tax
        :tgd      => [T, R],            # govt dom tariff
        :tgm      => [T, R],            # govt imp tariff
        :tpd      => [T, R],            # priv dom tariff
        :tpm      => [T, R],            # priv imp tariff
        :tp       => [R],               # private consumption tax
        :to       => [NS, R],           # output tax (NSAV_COMM × REG)
        :tfd      => [T, P, R],         # firm dom. intermediate tax
        :tfm      => [T, P, R],         # firm imp. intermediate tax
        :tf       => [E, P, R],         # factor tax
        :ao       => [P, R],            # output-augmenting tech (PROD_COMM)
        :ava      => [P, R],            # VA-augmenting tech
        :af       => [T, P, R],         # intermediate input-augmenting tech
        :afe      => [E, P, R],         # factor-augmenting tech
        :afecom   => [E],
        :afesec   => [P],
        :afereg   => [R],
        :afeall   => [E, P, R],
        :afcom    => [T],
        :afsec    => [P],
        :afreg    => [R],
        :afall    => [T, P, R],
        :aosec    => [P],
        :aoreg    => [R],
        :aoall    => [P, R],
        :avasec   => [P],
        :avareg   => [R],
        :avaall   => [P, R],
        :ams      => [T, R, R],         # import-augmenting tech (Armington)
        :atmfsd   => [M, T, R, R],      # transport tech (mode,comm,origin,dest)
        :atm      => [M],               # transport tech by mode
        :atf      => [T],               # transport tech by commodity
        :ats      => [R],               # transport tech by source
        :atd      => [R],               # transport tech by destination
        :atall    => [M, T, R, R],
        :pop      => [R],               # population growth
        :dppriv   => [R],               # private demand shift
        :dpgov    => [R],               # govt demand shift
        :dpsave   => [R],               # savings shift
        :au       => [R],               # utility shift
        :qo_endw  => [E, R],            # endowment supply
        # ── Endogenous variables (commonly swapped) ─────────────────────────
        :pgov     => [R],
        :pgd      => [T, R],
        :pgm      => [T, R],
        :pg       => [T, R],
        :ug       => [R],
        :yg       => [R],
        :qg       => [T, R],
        :qgd      => [T, R],
        :qgm      => [T, R],
        :ppriv    => [R],
        :ppd      => [T, R],
        :ppm      => [T, R],
        :pp       => [T, R],
        :up       => [R],
        :yp       => [R],
        :uepriv   => [R],
        :qp       => [T, R],
        :qpd      => [T, R],
        :qpm      => [T, R],
        :pim      => [T, R],
        :qim      => [T, R],
        :atpd     => [T, R],
        :atpm     => [T, R],
        :pva      => [P, R],
        :qva      => [P, R],
        :pf       => [T, P, R],
        :qf       => [T, P, R],
        :pfd      => [T, P, R],
        :qfd      => [T, P, R],
        :pfm      => [T, P, R],
        :qfm      => [T, P, R],
        :pfe      => [E, P, R],
        :qfe      => [E, P, R],
        :qo_prod_cgds => [P, R],
        :qds      => [T, R],
        :pmes_slug => [ES, P, R],
        :qoes_slug => [ES, P, R],
        :ps       => [NS, R],
        :rental   => [R],
        :kb       => [R],
        :ke       => [R],
        :rorc     => [R],
        :rore     => [R],
        :ksvces   => [R],
        :qcgds    => [R],
        :pcgds    => [R],
        :psave    => [R],
        :qsave    => [R],
        :pfob     => [T, R, R],
        :pcif     => [T, R, R],
        :pms      => [T, R, R],
        :pr       => [T, R],
        :qxs      => [T, R, R],
        :qtmfsd   => [M, T, R, R],
        :qtm      => [M],
        :pt       => [M],
        :ptrans   => [T, R, R],
        :qst      => [M, R],
        :pm       => [NS, R],
        :fincome  => [R],
        :dpav     => [R],
        :uelas    => [R],
        :dpsum    => [R],
        :y        => [R],
        :p        => [R],
        :u        => [R],
    )
end

"""
    var_domains_orig(s) → Dict{Symbol, Vector{Vector{String}}}

Like `var_domains(s)` but retains the original capitalisation of names from the
HAR data.  Use this for labelling CSV output.
"""
var_domains_orig(s::GTAPSets) = var_domains(s; lowercase = false)

# ── Name-to-index resolution ──────────────────────────────────────────────────

"""
    _model_sets(s) → Dict{String, Vector{String}}

Return the built-in model sets keyed by lowercase name, for use as wildcard
resolution in config files (e.g. `shock tm[TRAD_COMM, aut] = 5.0`).
"""
function _model_sets(s::GTAPSets)
    Dict{String,Vector{String}}(
        "trad_comm"  => lowercase.(s.TRAD_COMM),
        "reg"        => lowercase.(s.REG),
        "endw_comm"  => lowercase.(s.ENDW_COMM),
        "endws_comm" => lowercase.(s.ENDWS_COMM),
        "endwm_comm" => lowercase.(s.ENDWM_COMM),
        "endwc_comm" => lowercase.(s.ENDWC_COMM),
        "prod_comm"  => lowercase.(s.PROD_COMM),
        "marg_comm"  => lowercase.(s.MARG_COMM),
        "nsav_comm"  => lowercase.(s.NSAV_COMM),
        "cgds_comm"  => lowercase.(s.CGDS_COMM),
        "demd_comm"  => lowercase.(s.DEMD_COMM),
    )
end

"""
    resolve_token(token, dim_set, known_sets) → Vector{Int}

Resolve a single index token to a list of 1-based integer indices within
`dim_set` (the set for this variable dimension).

`known_sets` is a merged dict of model sets + user-defined sets.

Tokens are resolved in priority order:
  1. "all" or "*"    → every index in dim_set
  2. Known-set name  → all matching elements (intersection with dim_set)
  3. Element name    → exact match in dim_set (case-insensitive)
  4. Integer literal → returned as-is
"""
function resolve_token(token::String, dim_set::Vector{String},
                       known_sets::Dict{String,Vector{String}})
    tok = lowercase(strip(token, ['\'', '"', ' ']))

    # Wildcard
    (tok == "all" || tok == "*") && return collect(1:length(dim_set))

    # Named set (model or user-defined)
    if haskey(known_sets, tok)
        idxs = Int[]
        for name in known_sets[tok]
            i = findfirst(==(name), dim_set)
            i !== nothing ? push!(idxs, i) :
                @warn "Set element '$name' not found in dimension set; skipping"
        end
        isempty(idxs) && @warn "Set '$tok' has no elements in common with the dimension set"
        return idxs
    end

    # Element name (exact, case-insensitive)
    i = findfirst(==(tok), dim_set)
    i !== nothing && return [i]

    # Integer literal (backward compatibility)
    n = tryparse(Int, tok)
    n !== nothing && return [n]

    error("Cannot resolve index '$token': not 'all'/'*', not a known set name, " *
          "not an element of the dimension set, and not an integer.\n" *
          "  Dimension set: $(join(dim_set[1:min(10,end)], ", "))" *
          (length(dim_set) > 10 ? ", …" : ""))
end

"""
    expand_varspec(varspec, domains, known_sets) → Vector{String}

Expand a variable specification like `"tm[pdr,all]"` or `"tm[agri,aut]"` into
a list of fully-numeric specs like `["tm[1,1]", "tm[1,2]", …]`.

Returns `[varspec]` unchanged (with a warning) if the variable is not in
the domain table — numeric specs still work in that case.
"""
function expand_varspec(varspec::String,
                        domains   ::Dict{Symbol,Vector{Vector{String}}},
                        known_sets::Dict{String,Vector{String}})
    m = match(r"^(\w+)\[([^\]]+)\]$", strip(varspec))
    m === nothing && error("Cannot parse variable spec '$varspec' — expected 'name[i,j,…]'")

    varname = Symbol(m[1])
    tokens  = [String(strip(t)) for t in split(m[2], ',')]

    if !haskey(domains, varname)
        known = sort(string.(keys(domains)))
        error("Unknown variable '$varname' in spec '$varspec'.\n" *
              "  Known variables: $(join(known[1:min(15,end)], ", "))" *
              (length(known) > 15 ? ", …" : ""))
    end

    sets = domains[varname]
    length(tokens) == length(sets) ||
        error("$varname has $(length(sets)) dimensions but $(length(tokens)) " *
              "indices were given in '$varspec'")

    idx_lists = [resolve_token(tokens[d], sets[d], known_sets)
                 for d in eachindex(tokens)]

    ["$(varname)[$(join(combo, ','))]"
     for combo in Iterators.product(idx_lists...)]
end

# ── High-level expanders ──────────────────────────────────────────────────────

"""
    resolve_config(raw_shocks, raw_swaps, user_sets, s)
        → (shocks::Dict{String,Float64}, swaps::Vector{Swap})

Expand named/set-referenced shocks and swaps into fully-numeric form, using the
data sets in `s` for name lookup.

`raw_shocks` is a `Vector{Pair{String,Float64}}` (ordered, allows duplicates).
`raw_swaps`  is a `Vector{Tuple{String,String,Float64}}` (endo_out, exog_in, fix_at).
`user_sets`  is a `Dict{String,Vector{String}}` from `parse_config`.
"""
function resolve_config(raw_shocks,
                        raw_swaps,
                        user_sets  ::Dict{String,Vector{String}},
                        s          ::GTAPSets)
    domains    = var_domains(s)
    # Normalize everything to lowercase so element matching is case-insensitive.
    # model sets are already lowercase; user sets may have mixed case.
    normed_user = Dict{String,Vector{String}}(
        lowercase(k) => lowercase.(v) for (k, v) in user_sets)
    known_sets = merge(_model_sets(s), normed_user)   # user sets override model sets

    # ── Shocks ────────────────────────────────────────────────────────────────
    shocks = Dict{String,Float64}()
    for (spec, val) in raw_shocks
        for expanded in expand_varspec(spec, domains, known_sets)
            shocks[expanded] = val      # later entries override earlier ones
        end
    end

    # ── Swaps ─────────────────────────────────────────────────────────────────
    swaps = Swap[]
    for (out_spec, in_spec, fix_at) in raw_swaps
        outs = expand_varspec(out_spec, domains, known_sets)
        ins  = expand_varspec(in_spec,  domains, known_sets)
        length(outs) == length(ins) ||
            error("Swap '$out_spec' <-> '$in_spec' expands to different counts: " *
                  "$(length(outs)) endo elements vs $(length(ins)) exog elements")
        for (o, i) in zip(outs, ins)
            push!(swaps, Swap(o, i, fix_at))
        end
    end

    return shocks, swaps
end
