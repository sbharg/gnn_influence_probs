# Learning Diffusion Probabilities from Cascades

Source code for my ECE381K project, which aims to learn the parameters of an Independent Cascade (IC)
model on a network using only cascades that are passively observed on the network. 
Contains scripts to
- Generate training cascades on a network with a ground truth IC model defined on it. 
- Run the GNN-based method developed for the project. 
- Run two baseline algorithms (SLICER and MLE) described in prior work by Wilinski and Lokhov [1]. 

## Requirements

The GNN method is written in Python and was tested with version `python3.12.7`. 
We use the [uv](https://docs.astral.sh/uv/) package manager to handle 
creating a virtual env, adding dependencies, and running scripts. Once uv is installed, 
you can run `uv venv` and then `uv sync` to setup a virtual environment and install 
all the necessary packages to run the scripts. 
Alternatively, you can set up a python virtual environment using 
`python3 -m venv`, active it using `source /.venv/bin/activate`, and then run `pip3 install -r requirements.txt`. 

The two baseline algorithms are written in [Julia](https://julialang.org/downloads/) and 
were tested with version `v1.11.21`. The source code was taken and slightly modified from
[the repository](https://github.com/mateuszwilinski/dynamic-message-passing) linked in the 
paper by Wilinski and Lokhov [1]. Once Julia is installed, you can 
run `julia julia-benchmarks/packages.jl` to install all the required dependencies to run the algorithms. 

## Usage (Python Scripts)

## Usage (Julia Scripts)

## References

[1] M. Wilinski and A. Lokhov. Prediction-centric learning of independent cascade dynamics from partial observations. In Proceedings of ICML 2021,
pages 11182–11192, 2021. [arxiv:2007.06557](https://arxiv.org/abs/2007.06557). 