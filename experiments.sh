#!/bin/sh

#python3 gnn.py datasets/synthetic/er_100_02/

julia julia-benchmarks/slicer_benchmark.jl datasets/real/ego-facebook/ 31
#julia julia-benchmarks/mle_benchmark.jl datasets/real/ego-facebook/ 31