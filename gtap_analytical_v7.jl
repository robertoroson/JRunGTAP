# =============================================================================
# gtap_analytical_v7.jl
# Sparse Jacobian for the GTAPv7 linearised system.
#
# The GTAP linearised model is exactly linear in % changes, so the Jacobian
# is a constant coefficient matrix (for given benchmark data).  We exploit
# this in two ways:
#
#  1. First call: build the full pattern column-by-column (slow, O(n) residual
#     evaluations) and cache the sparsity structure + greedy column colouring.
#
#  2. Subsequent calls (Euler/Gragg sub-steps): use compressed finite
#     differences.  Columns that share no common row are "compatible" and can
#     be probed simultaneously.  With ~20–50 colours for a typical GTAP
#     dataset this is 2000–5000× faster than the column-by-column approach.
#
# The linearity guarantee means F(sum_j e_j) = sum_j A e_j exactly, so the
# compressed approach is numerically exact (not an approximation).
# =============================================================================

using SparseArrays
using Serialization

# ── Module-level caches ────────────────────────────────────────────────────────
# Sparsity pattern (Bool sparse matrix) and column groups from greedy colouring.
# Bump this string whenever the model equations change (forces Jacobian rebuild).
const JAC_MODEL_VERSION_V7 = "v7.6-pfactor-full-endowments"

const _JAC_PATTERN_V7    = Ref{Union{Nothing, SparseMatrixCSC{Bool,Int}}}(nothing)
const _JAC_GROUPS_V7     = Ref{Union{Nothing, Vector{Vector{Int}}}}(nothing)
# Path for the on-disk Jacobian cache (set by load_data_v7 from the zip path).
const _JAC_CACHE_PATH_V7 = Ref{Union{Nothing, String}}(nothing)
# Path of the zip file whose data the cached Jacobian was built from.
const _JAC_SOURCE_ZIP_V7 = Ref{Union{Nothing, String}}(nothing)

"""
    set_jac_cache_path_v7(zip_path)

Derive and store the cache file path from the data zip path.
The cache is saved as `<stem>_jacobian_v7.jac` next to the zip file.
Called automatically by `load_data_v7`.
"""
function set_jac_cache_path_v7(zip_path::String)
    abs_zip = abspath(zip_path)
    stem = splitext(abs_zip)[1]
    _JAC_CACHE_PATH_V7[] = stem * "_jacobian_v7.jac"
    _JAC_SOURCE_ZIP_V7[] = abs_zip
end

"""
Greedy column colouring of a sparse matrix.

Two columns conflict if they share at least one nonzero row.  Returns a
partition of 1:n_col into groups of mutually compatible columns; evaluating
F on the sum of a group's unit vectors simultaneously recovers all columns
in the group in a single residual call.
"""
function _greedy_column_coloring(A::SparseMatrixCSC)
    n  = size(A, 2)
    nr = size(A, 1)
    # Invert: for each row, list its nonzero columns
    row_cols = [Int[] for _ in 1:nr]
    @inbounds for j in 1:n
        for k in A.colptr[j]:A.colptr[j+1]-1
            push!(row_cols[A.rowval[k]], j)
        end
    end
    colors    = zeros(Int, n)
    max_color = 0
    used      = Set{Int}()
    @inbounds for j in 1:n
        empty!(used)
        # Collect colors of conflicting (already-colored) columns
        for k in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[k]
            for k2 in row_cols[i]
                k2 < j && colors[k2] > 0 && push!(used, colors[k2])
            end
        end
        c = 1; while c in used; c += 1; end
        colors[j] = c
        max_color = max(max_color, c)
    end
    [findall(==(c), colors) for c in 1:max_color]
end

"""
    build_A_v7(d, s, C) → SparseMatrixCSC

Build the full Jacobian A of the GTAPv7 core residual vector.

On the first call the sparsity pattern is discovered column-by-column and
cached together with the greedy column colouring.  All subsequent calls use
compressed finite differences (one residual evaluation per colour group),
which is 2000–5000× faster for typical GTAP data sizes.
"""
function build_A_v7(d::GTAPDataV7, s::GTAPSetsV7, C)
    n_col = n_endog_v7(s)
    exog  = make_exog_zero_v7(s)

    if _JAC_PATTERN_V7[] === nothing
        # ── Try loading from disk cache first ─────────────────────────────
        cache_path = _JAC_CACHE_PATH_V7[]
        zip_path   = _JAC_SOURCE_ZIP_V7[]
        if cache_path !== nothing && isfile(cache_path)
            cached = deserialize(cache_path)
            if length(cached) == 5
                model_ver, A_cached, pat, groups, saved_mtime = cached
                current_mtime = mtime(zip_path)
                if model_ver == JAC_MODEL_VERSION_V7 && saved_mtime == current_mtime
                    @info "Loading Jacobian from cache: $cache_path  ($(nnz(pat)) nonzeros, $(length(groups)) colour groups)"
                    _JAC_PATTERN_V7[] = pat
                    _JAC_GROUPS_V7[]  = groups
                    return A_cached
                elseif model_ver != JAC_MODEL_VERSION_V7
                    @info "Cache stale (model version changed $model_ver → $JAC_MODEL_VERSION_V7): rebuilding Jacobian"
                else
                    @info "Cache stale (zip modified): rebuilding Jacobian"
                end
            else
                @info "Cache format outdated: rebuilding Jacobian"
            end
        end

        # ── First call: full column-by-column pattern discovery ────────────
        e_j  = zeros(n_col)
        e_j[1] = 1.0
        col0 = F_core_v7(e_j, exog, d, s, C)
        e_j[1] = 0.0
        n_row = length(col0)

        Is = Int[]; Js = Int[]; Vs = Float64[]
        sizehint!(Is, n_col * 3); sizehint!(Js, n_col * 3); sizehint!(Vs, n_col * 3)
        for i in eachindex(col0)
            col0[i] != 0.0 && (push!(Is,i); push!(Js,1); push!(Vs,col0[i]))
        end
        for j in 2:n_col
            e_j[j] = 1.0
            col = F_core_v7(e_j, exog, d, s, C)
            e_j[j] = 0.0
            for i in eachindex(col)
                col[i] != 0.0 && (push!(Is,i); push!(Js,j); push!(Vs,col[i]))
            end
        end
        A = sparse(Is, Js, Vs, n_row, n_col)
        # Cache boolean pattern and compute colouring
        pat = SparseMatrixCSC{Bool,Int}(n_row, n_col, copy(A.colptr),
                                         copy(A.rowval), ones(Bool, length(A.nzval)))
        _JAC_PATTERN_V7[] = pat
        _JAC_GROUPS_V7[]  = _greedy_column_coloring(pat)
        ncolors = length(_JAC_GROUPS_V7[])
        @info "Jacobian pattern cached: $(nnz(pat)) nonzeros, $ncolors colour groups (future builds will use compressed FD)"
        # ── Save to disk cache ────────────────────────────────────────────
        cache_path = _JAC_CACHE_PATH_V7[]
        zip_path   = _JAC_SOURCE_ZIP_V7[]
        if cache_path !== nothing && zip_path !== nothing
            serialize(cache_path, (JAC_MODEL_VERSION_V7, A, _JAC_PATTERN_V7[], _JAC_GROUPS_V7[], mtime(zip_path)))
            @info "Jacobian saved to: $cache_path"
        end
        return A
    end

    # ── Subsequent calls: compressed finite differences ─────────────────────
    pat    = _JAC_PATTERN_V7[]
    groups = _JAC_GROUPS_V7[]
    n_row  = size(pat, 1)

    Is = Int[]; Js = Int[]; Vs = Float64[]
    sizehint!(Is, nnz(pat)); sizehint!(Js, nnz(pat)); sizehint!(Vs, nnz(pat))

    x = zeros(n_col)
    for grp in groups
        # Probe the sum of all unit vectors in this group
        for j in grp; x[j] = 1.0; end
        y = F_core_v7(x, exog, d, s, C)
        for j in grp; x[j] = 0.0; end
        # Scatter: column j owns row i exclusively within this group
        for j in grp
            for k in pat.colptr[j]:pat.colptr[j+1]-1
                i = pat.rowval[k]
                i <= length(y) || continue
                y[i] != 0.0 && (push!(Is,i); push!(Js,j); push!(Vs,y[i]))
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
