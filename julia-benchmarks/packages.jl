using Pkg

dependencies = [
    "MatrixMarket",
    "StatsBase",
    "Ipopt",
    "JuMP",
    "ArgParse",
]

Pkg.add(dependencies)