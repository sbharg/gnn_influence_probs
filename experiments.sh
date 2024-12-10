#!/bin/sh

dataset=datasets/synthetic/er_50_005/ 

julia julia-benchmarks/slicer_benchmark.jl $dataset
julia julia-benchmarks/mle_benchmark.jl $dataset 
uv run gnn.py $dataset 
