include("jrungtap.jl")

zippath = "gtapv7.zip"   # adjust if needed
cfgpath = raw"C:\Users\roson\Downloads\TwC1.cfg"

method, steps, raw_shocks, raw_swaps, user_sets, model = parse_config(cfgpath)
println("Method: $method  Steps: $steps  Model: $model")

s, d0, C0 = load_data_v7(zippath)
shocks, swaps = resolve_config(raw_shocks, raw_swaps, user_sets, s)

exp = gtap_experiment("TwC1"; shocks=shocks, swaps=swaps, method=method, steps=steps)
sol = run_gtap_v7(exp, zippath)

taiwan_idx = findfirst(==("Taiwan"), s.REG)
comp_idx   = findfirst(==("Computer"), s.ACTS)

qxw   = get_result_v7(sol, :qxw)
aoall = get_result_v7(sol, :aoall)

println("\nqxw[Computer, Taiwan]   = ", qxw[comp_idx, taiwan_idx],
        "  (expected ≈ -5.0 if fix_scale works)")
println("aoall[Computer, Taiwan] = ", aoall[comp_idx, taiwan_idx])
