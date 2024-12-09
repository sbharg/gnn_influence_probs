import MatrixMarket
import SparseArrays
using ArgParse

include("src/structures.jl")
include("src/cascade_tools.jl")
include("src/network_tools.jl")
include("src/maximum_likelihood_method.jl")

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
    return g
end

function generate_cascades(path::String, n::Int64, M::Int64=250)
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
    name = "results/mle/$(gname)_$(k).mtx"
    MatrixMarket.mmwrite(name, M)
end

function run_benchmark(g, cascades, T, threshold=1e-6, max_iter=1000)
    likelihood_edgelist = max_likelihood_params(g, cascades, T, threshold, max_iter)

    # Comparison with true values
    average_error_on_alphas = sum(abs.(values(merge(-, g.edgelist, likelihood_edgelist)))) / g.m
    return average_error_on_alphas, likelihood_edgelist
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

    g = load_graph(parsed_args["filepath"])
    kmin = parsed_args["kmin"]
    kmax = parsed_args["kmax"]
    delta = parsed_args["delta"]
    cascades, T = generate_cascades(parsed_args["filepath"], g.n, kmax)

    io = open("results/mle/mle.txt", "a")
    print("M \tMAE \t\t\tTime\n")
    gname = split(parsed_args["filepath"], "/")[end-1]
    write(io, "\nGraph: $gname\n")
    write(io, "M\t\tMAE\t\t\t\t\t\tTime\n")    
    for i in range(start=kmin, step=delta, stop=kmax)
        start = time()
        c = cascades[1:g.n, 1:i]
        a, probs = run_benchmark(g, c, T, parsed_args["threshold"], parsed_args["maxiter"])
        t = time() - start

        res = "$i \t$a \t$t\n"
        print(res)
        res_write = "$i\t\t$a\t\t$t\n"
        write(io, res_write);    
        write_to_graph(probs, g.n, gname, i)
    end
    close(io);
end

main()