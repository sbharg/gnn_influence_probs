import MatrixMarket
import SparseArrays
using StatsBase
using ArgParse
using BenchmarkTools

include("src/structures.jl")
include("src/cascade_tools.jl")
include("src/network_tools.jl")
include("src/lagrange_dmp_method.jl")
include("src/gradient_dmp_method.jl")

function load_graph(path::String)
    M = MatrixMarket.mmread(path * "/graph.mtx")

    # Construct edge list from matrix
    data = SparseArrays.findnz(M)
    src = data[1]
    dst = data[2]

    edges = hcat(src, dst)
    weights = data[3]

    m = size(edges)[1]
    n = maximum(edges)

    edgelist = edgelist_from_array(edges, weights)
    out_neighbors, in_neighbors = dir_neighbors_from_edges(edgelist, n)
    g = DirGraph(n, m, edgelist, out_neighbors, in_neighbors)
    return g, edges
end

function generate_cascades(path::String, n::Int64, M::Int64=250, T::Int64=12)
    cascades = zeros(Int64, n, M)

    # Iterate through cascades
    for i in 1:M
        c = path * "/diffusions/timestamps/$(i-1).txt"
        tvec = Vector{Int}[]
        c_data = open(c) do file
            for line in eachline(file)
                push!(tvec, [parse(Int, x)+1 for x in split(line)])
            end
        end

        p0 = zeros(Int64, n)
        for i in eachindex(tvec)
            for j in tvec[i]
                p0[j] = i
            end
        end
        for i in 1:n
            if p0[i] == 0
                p0[i] = T
            end
        end
        cascades[1:n, i] = p0
    end
    return cascades
end

function run_benchmark(g, edges, cascades, M, T)
    threshold = 1e-6
    max_iter = 1000
    iter_threshold = 400

    d = 0
    unobserved = StatsBase.sample(1:g.n, d, replace=false)
    observed = filter(!in(unobserved), 1:g.n)

    cascades_classes = preprocess_cascades(cascades)
    remove_unobserved!(cascades_classes, unobserved)
    g_temp = DirGraph(g.n, g.m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])), g.out_neighbors, g.in_neighbors)

    ratio = 1.0
    multiplier = g.n / M / T / 80.0
    iter = 0
    while (abs(ratio) > threshold) & (iter < max_iter)
        D, objective_old = get_lagrange_gradient(cascades_classes, g_temp, T)

        for (e, v) in g_temp.edgelist
            step = D[e] * multiplier
            while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                step /= 2.0
            end
            g_temp.edgelist[e] = v - step
        end
        objective_new = get_full_objective(cascades_classes, g_temp, T)
        ratio = (objective_new - objective_old) / abs(objective_old)

        iter += 1
        if iter > iter_threshold
            multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
        end
    end
    
    # Comparison with true values
    average_error_on_alphas = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / g.m
    return average_error_on_alphas
end

@eval BenchmarkTools macro btimed(args...)
    _, params = prunekwargs(args...)
    bench, trial, result = gensym(), gensym(), gensym()
    trialmin, trialallocs = gensym(), gensym()
    tune_phase = hasevals(params) ? :() : :($BenchmarkTools.tune!($bench))
    return esc(quote
        local $bench = $BenchmarkTools.@benchmarkable $(args...)
        $BenchmarkTools.warmup($bench)
        $tune_phase
        local $trial, $result = $BenchmarkTools.run_result($bench)
        local $trialmin = $BenchmarkTools.minimum($trial)
        $result, $BenchmarkTools.time($trialmin)
    end)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "filepath"
            help = "Path to folder containing graph and cascades"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    g, edges = load_graph(parsed_args["filepath"])
    
    T = 12
    M = 250
    cascades = generate_cascades(parsed_args["filepath"], g.n, M, T)
    
    println("M \tMAE \t\t\tTime")
    for i in [50, 100, 150, 200, 250]
        c = cascades[1:g.n, 1:i]
        a, t = BenchmarkTools.@btimed run_benchmark(g, edges, cascades, M, T) setup=(g=$g; edges=$edges; cascades=$c; M=$i; T=$T) samples=3
        #println("For M = ", i, ", MAE = ", a, ", time = ", t*1e-9, " s")
        println(i, " \t", a, " \t", t*1e-9, " s")
    end
end

main()