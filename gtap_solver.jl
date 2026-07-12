# gtap_solver.jl
# Result accessors for the linearised GTAP solver.
#
# The Jacobian is built analytically in gtap_analytical.jl (build_A_analytical).
# This file provides get_variable for extracting named variables from x_endo.

using SparseArrays, LinearAlgebra

include("gtap_pack.jl")   # pack / unpack / F_core

# ─────────────────────────────────────────────────────────────────────────────
# RESULT ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_variable(x_endo, name::Symbol, s) → Array

Extract a single endogenous variable from the solution vector.

Example:
    pm = get_variable(x_sol, :pm, s)   # [nNS, nR] matrix of % changes
"""
function get_variable(x_endo::AbstractVector, name::Symbol, s::GTAPSets)
    _, offsets, _ = endo_offsets(s)
    specs = Dict(_endo_specs(s))
    haskey(specs, name)   || error("Unknown endogenous variable: $name")
    haskey(offsets, name) || error("Variable $name has no offset (check _endo_specs)")
    shape = specs[name]
    off = offsets[name]
    return reshape(x_endo[off : off + prod(shape) - 1], shape)
end
