include("jrungtap.jl")

zippath = "Taiwan.zip"   # on other PC: raw"C:\runGTAP375\under2\gtapv7.zip"

# Load data to get sets (needed to resolve names to indices)
s, d0, C0 = load_data_v7(zippath)

taiwan_idx = findfirst(==("Taiwan"), s.REG)
comp_idx   = findfirst(==("Computer"), s.ACTS)
println("Computer idx=$comp_idx, Taiwan idx=$taiwan_idx")

# Build swap with integer-indexed specs (as resolve_config would produce)
sw  = Swap("qxw[$comp_idx,$taiwan_idx]", "ao[$comp_idx,$taiwan_idx]", -5.0)
exp = gtap_experiment("TwC1";
    shocks = Dict{String,Float64}(),
    swaps  = [sw],
    method = :gragg,
    steps  = 6)

sol = run_gtap_v7(exp, zippath)

qxw   = get_result_v7(sol, :qxw)
aoall = get_result_v7(sol, :aoall)

println("\nqxw[Computer,Taiwan]   = ", round(qxw[comp_idx, taiwan_idx],   digits=4),
        "  (expected ≈ -5.0)")
println("aoall[Computer,Taiwan] = ", round(aoall[comp_idx, taiwan_idx], digits=4),
        "  (endogenous response)")
