# Learning Diffusion Probabilities from Cascades

Source code for my ECE381K project, which aims to learn the parameters of an Independent Cascade (IC)
model on a network using only cascades that are passively observed on the network. 
Contains scripts to
- Generate training cascades on a network with a ground truth IC model defined on it. 
- Run the GNN-based method developed for the project. 
- Run two baseline algorithms (SLICER and MLE) described in prior work by Wilinski and Lokhov [1]. 

We also test the quality of the learned IC model parameters by using them in a 
simple downstream application of the Influence Maximization problem. In particular, we 
run the IMM algorithm [2] to find $k$ influential nodes in the graph with the learned parameters, and 
compare them against the $k$ influential nodes with the ground truth parameters.  

## Requirements

The GNN method is written in Python and was tested with version `python3.12.7`. 
We use the [uv](https://docs.astral.sh/uv/) package manager to handle 
creating a virtual env, adding dependencies, and running scripts. Once uv is installed, 
you can run `uv venv` and then `uv sync` to setup a virtual environment and install 
all the necessary packages to run the scripts. 
Alternatively, you can set up a python virtual environment using 
`python3 -m venv`, active it using `source .venv/bin/activate`, and then run `pip3 install -r requirements.txt`. 

The two baseline algorithms are written in [Julia](https://julialang.org/downloads/) and 
were tested with version `v1.11.21`. The source code was slightly modified from
[the repository](https://github.com/mateuszwilinski/dynamic-message-passing) linked in the 
paper by Wilinski and Lokhov [1]. Once Julia is installed, you can 
run `julia julia-benchmarks/packages.jl` to install all the required dependencies to run the algorithms. 

The IMM algorithm is written in C++11 and requires version `4.8.1` or above of the g++ compiler to compile. 
The original source code was downloaded in [the repository](https://sourceforge.net/projects/im-imm/) linked in the 
IMM paper by Tang, Shi, and Xiao [2]. 

## Graph Format

We use the [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html]) coordinate format to 
represent and store all the graphs used. We consider only directed graphs. The weight of an edge represents 
the IC model parameter associated with it. All graphs are stored within the `datasets/` folder in the format of 
`[name]/graph.mtx` (ex. `datasets/real/ego-facebook/graph.mtx)`. 

## Usage (Python Scripts)

### Cascade Generator

Generates training cascades according to the ground truth IC model defined on a network. For each cascade generated, 
both the timestamp and edgelist version is logged. The timestamp version the nodes activated at each timestamp, 
where the nodes listed on the i'th line indicate the nodes activated during the i'th timestamp. The edgelist version 
list all the edges that were successfully activated and passed information from their source node to their destination node. 

Usage: 
```
uv run cascade_generator.py [-h] [-n NCASCADES] [-s SEEDSIZE] filepath
```

required arguments:
- `filepath`: Path to folder containing a graph.mtx file with ground truth IC probabilities (eg. datasets/real/ego-facebook/)

optional arguments:
- `-n, --ncascades <NCASCADES>`
    - Number of cascades to generate (default: 250)
- `-s, --seedsize <SEEDSIZE>`
    - Size of seed set to start cascades from (default: 1)
- `-h, --help`           
    - show this help message and exit

### Cascade GNN

Runs the GNN based method. 
Prints out and logs the average $L_1$ error and time taken for the method 
for each number of training cascades in the range specified in the `results/gnn/gnn.txt` file. 
Also saves the estimated probabilities on each edge as an `.mtx` file in the `results/gnn` folder. 

Usage:
```
uv run cascade_gnn.py [-h] [-i KMIN] [-a KMAX] [-d DELTA] [-e EPOCHS] [-l LEARNINGRATE] [-n NUMLAYERS] [-b EMBEDDINGDIM] filepath
```
required arguments:
- `filepath`: Path to folder containing graph and cascades (ex. datasets/real/ego-facebook/)

optional arguments:
- `-i, --kmin <KMIN>`  
    - Minimum number of training cascades (default: 50)
- `-a, --kmax <KMAX>`  
    - Maximum number of training cascades (default: 250)
- `-d, --delta <DELTA>`
    - Runs method with [kmin, kmin+delta, kmin+2*delta, ..., kmax] training cascades (default: 50)
- `-e, --epochs <EPOCHS>`
    - Number of epochs to train for (default: 35)
- `-l, --learningrate <LEARNINGRATE>`
    - Learning rate for training (default: 0.01)
- `-n, --numlayers <NUMLAYERS>`
    - Number of GAT layers in the GNN (default: 2)
- `-b, --embeddingdim <EMBEDDINGDIM>`
    - The dimension of the edge embeddings (default: 16)
- `-h, --help `           
    - show this help message and exit

## Usage (Julia Scripts)

### MLE Benchmark

Runs the Convex Programming based Maximum Likelihood Estimation (MLE) method. 
Prints out and logs the average $L_1$ error and time taken for the algorithm 
for each number of training cascades in the range specified in the `results/mle/mle.txt` file. 
Also saves the estimated probabilities on each edge as an `.mtx` file in the `results/mle` folder. 

Usage: 
```
julia julia_benchmarks/mle_benchmark.jl [-i KMIN] [-a KMAX] [-d DELTA] [-t THRESHOLD] [-m MAXITER] [-h] filepath
```

required arguments:
- `filepath`: Path to folder containing graph and cascades (ex. datasets/real/ego-facebook/)

optional arguments:
- `-i, --kmin <KMIN>`
    - Minimum number of training cascades (type: Int64, default: 50)
- `-a, --kmax <KMAX>`
    - Maximum number of training cascades (type: Int64, default: 250)
- `-d, --delta <DELTA>`
    - Runs algorithm with {kmin, kmin+delta, kmin+2*delta, ..., kmax} training cascades (type: Int64, default: 50)
- `-t, --threshold <THRESHOLD>`
    - Threshold for convergence (type: Float64, default: 1.0e-6)
- `-m, --maxiter <MAXITER>`
    - Maximum number of iterations (type: Int64, default: 1000)
- `-h, --help`
    - show this help message and exit

### SLICER Benchmark

Runs the Dynamic Message Passing (SLICER) based method. 
Prints out and logs the average $L_1$ error and time taken for the algorithm 
for each number of training cascades in the range specified in the `results/slicer/slicer.txt` file. 
Also saves the estimated probabilities on each edge as an `.mtx` file in the `results/slicer` folder. 

Usage: 
```
julia julia_benchmarks/slicer_benchmark.jl [-i KMIN] [-a KMAX] [-d DELTA] [-t THRESHOLD] [-m MAXITER] [-h] filepath
```

required arguments:
- `filepath`: Path to folder containing graph and cascades (ex. datasets/real/ego-facebook/)

optional arguments:
- `-i, --kmin <KMIN>`
    - Minimum number of training cascades (type: Int64, default: 50)
- `-a, --kmax <KMAX>`
    - Maximum number of training cascades (type: Int64, default: 250)
- `-d, --delta <DELTA>`
    - Runs algorithm with {kmin, kmin+delta, kmin+2*delta, ..., kmax} training cascades (type: Int64, default: 50)
- `-t, --threshold <THRESHOLD>`
    - Threshold for convergence (type: Float64, default: 1.0e-6)
- `-m, --maxiter <MAXITER>`
    - Maximum number of iterations (type: Int64, default: 1000)
- `-h, --help`
    - show this help message and exit

## Usage (InfMax Experiment)

To recreate the InfMax Experiment on the ego-facebook graph, run the first three cells in the `imm_experiments.ipynb` file to 
generate the graphs in the format required for the IMM code. Then run the `IMM/imm_experiments.sh` file to generate the 
seed sets for each method. After manually importing the seed sets generated, 
running the remaining cells of the `imm_experiments.ipynb` file will calculate the 
expected spread of the seed sets of each method and list them. 

## References

[1] M. Wilinski and A. Lokhov. Prediction-centric learning of independent cascade dynamics from partial observations. In Proceedings of ICML 2021,
pages 11182–11192, 2021. [arxiv:2007.06557](https://arxiv.org/abs/2007.06557). 

[2] Y. Tang, Y. Shi, and X. Xiao. Influence maximization in near-linear time: A martingale approach. In Proceedings of SIGMOD 2015, pages 1539–
1554, 2015. [doi:10.1145/2723372.2723734](https://doi.org/10.1145/2723372.2723734).