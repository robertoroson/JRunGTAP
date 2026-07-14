# =============================================================================
# gtap_analytical_v7.jl
# Exact sparse Jacobian for the GTAPv7 linearised system.
#
# Because the model is fully linear in the % changes, the Jacobian is constant
# for a given data vintage (it changes only when level variables are updated
# in Euler/Gragg steps).  We exploit linearity to compute each column exactly
# by evaluating F_core_v7(e_j, exog0, d, s, C) where e_j is the j-th unit
# vector — no automatic differentiation package required.
# =============================================================================

using SparseArrays

"""
    build_A_v7(d, s, C) → SparseMatrixCSC

Build the full Jacobian A of the GTAPv7 core residual vector with respect to
the endogenous variable vector, evaluated at x_endo = 0 (which is exact for a
linear model in % changes).
"""
function build_A_v7(d::GTAPDataV7, s::GTAPSetsV7, C)
    n_col = n_endog_v7(s)       # number of endogenous variables (columns)
    exog  = make_exog_zero_v7(s)
    # F_core_v7 returns n_eqs rows; compute on first column to get row count
    e_j  = zeros(n_col)
    e_j[1] = 1.0
    col0 = F_core_v7(e_j, exog, d, s, C)
    e_j[1] = 0.0
    n_row = length(col0)        # = n_eqs (may differ from n_col by Walras DOF)

    Is   = Int[]
    Js   = Int[]
    Vs   = Float64[]
    sizehint!(Is, n_col * 3)
    sizehint!(Js, n_col * 3)
    sizehint!(Vs, n_col * 3)

    # Store first column result
    for i in eachindex(col0)
        if col0[i] != 0.0; push!(Is, i); push!(Js, 1); push!(Vs, col0[i]); end
    end

    for j in 2:n_col
        e_j[j] = 1.0
        col    = F_core_v7(e_j, exog, d, s, C)
        e_j[j] = 0.0
        for i in eachindex(col)
            if col[i] != 0.0
                push!(Is, i); push!(Js, j); push!(Vs, col[i])
            end
        end
    end
    sparse(Is, Js, Vs, n_row, n_col)
end

"""
    eq_row_offsets_v7(s) → Dict{Symbol, Int}

Return the row offset (1-based) of each equation block inside the residual
vector produced by pack_residuals_core_v7.
"""
function eq_row_offsets_v7(s::GTAPSetsV7)
    sz  = core_eq_sizes_v7(s)
    off = Dict{Symbol,Int}()
    cursor = 1
    for nm in CORE_EQ_NAMES_V7
        off[nm] = cursor
        cursor  += prod(sz[nm])
    end
    off
end
