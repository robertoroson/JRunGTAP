include("jrungtap.jl")

zippath    = "GTAP12_15x15x17.zip"
s, d0, C0  = load_data_v7(zippath)
taiwan_idx = findfirst(==("Taiwan"),   s.REG)
comp_idx   = findfirst(==("Computer"), s.ACTS)
println("Computer idx=$comp_idx, Taiwan idx=$taiwan_idx")
println("ACTS = ", s.ACTS)
println("REG  = ", s.REG)

sw  = Swap("qxw[$comp_idx,$taiwan_idx]", "ao[$comp_idx,$taiwan_idx]", -5.0)
exp = gtap_experiment("TwC1"; shocks=Dict{String,Float64}(), swaps=[sw], method=:gragg, steps=6)
sol = run_gtap_v7(exp, zippath)

qo = get_result_v7(sol, :qo)
nA, nR = length(s.ACTS), length(s.REG)

println("\nqo (% change) — JRunGTAP Gragg/6 — GTAP12_15x15x17:")
@printf "%-14s" "sector"
for r in s.REG; @printf "%8s" r[1:min(7,end)]; end; println()
for ai in 1:nA
    @printf "%-14s" s.ACTS[ai][1:min(14,end)]
    for ri in 1:nR; @printf "%8.2f" qo[ai,ri]; end
    println()
end

println("\nRunGTAP reference qo[Computer,Taiwan] ≈ -4.84")
println("JRunGTAP       qo[Computer,Taiwan]   = ", round(qo[comp_idx, taiwan_idx], digits=4))
