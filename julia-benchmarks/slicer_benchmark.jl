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
    T = 0
    for i in 1:M
        c = path * "/diffusions/timestamps/$(i-1).txt"
        time_len = 0
        c_data = open(c) do file
            for line in eachline(file)
                time_len += 1
            end
        end
        T = max(T, time_len)
    end

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
    return cascades, T
end

function run_benchmark(g, edges, cascades, M, T, threshold=1e-7, max_iter=10000)
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
    return average_error_on_alphas, g_temp.edgelist
end

function write_to_graph(edgelist, n, gname, k)
    src = []
    dst = []
    probs = []
    for (e, p) in edgelist
        #println(e)
        push!(src, e[1])
        push!(dst, e[2])
        p_new = convert(Float64, p)
        push!(probs, p_new)
    end

    probs = convert(Array{Float64,1}, probs)
    M = SparseArrays.sparse(src, dst, probs)
    name = "results/slicer/$(gname)_$(k).mtx"
    MatrixMarket.mmwrite(name, M)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "filepath"
            help = "Path to folder containing graph and cascades"
            arg_type = String
            required = true
        "--kmin", "-i"
            help = "Minimum number of training cascades"
            arg_type = Int64
            default = 50
        "--kmax", "-a"
            help = "Maximum number of training cascades"
            arg_type = Int64
            default = 250
        "--delta", "-d"
            help = "Increment in number of training cascades to test on. Runs algorithm with [kmin, kmin+delta, kmin+2*delta, ..., kmax] training cascades"
            arg_type = Int64
            default = 50
        "--threshold", "-t"
            help = "Threshold for convergence"
            arg_type = Float64
            default = 1e-6
        "--maxiter", "-m"
            help = "Maximum number of iterations"
            arg_type = Int64
            default = 1000
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    g, edges = load_graph(parsed_args["filepath"])
    kmin = parsed_args["kmin"]
    kmax = parsed_args["kmax"]
    delta = parsed_args["delta"]
    cascades, T = generate_cascades(parsed_args["filepath"], g.n, kmax)
    
    io = open("results/slicer/slicer.txt", "a")
    print("M \tMAE \t\t\tTime\n")
    gname = split(parsed_args["filepath"], "/")[end-1]
    write(io, "\nGraph: $gname\n")
    write(io, "M\t\tMAE\t\t\t\t\t\tTime\n")    
    for i in range(start=kmin, step=delta, stop=kmax)
        start = time()
        c = cascades[1:g.n, 1:i]
        a, probs = run_benchmark(g, edges, c, i, T, parsed_args["threshold"], parsed_args["maxiter"])
        t = time() - start

        res = "$i \t$a \t$t\n"
        print(res)
        res_write = "$i\t\t$a\t\t$t\n"
        write(io, res_write)  
        write_to_graph(probs, g.n, gname, i)
    end
    close(io);
end

main()