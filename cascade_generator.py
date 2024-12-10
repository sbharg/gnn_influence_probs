import random
import io
import networkx as nx
import scipy as sp
import numpy as np
rng = np.random.default_rng()
from collections import deque
from pathlib import Path
import argparse

def monte_carlo_diffusion_nodes(G: nx.DiGraph, seed_nodes: list[int]):
  activated_nodes = set(seed_nodes)
  queue = deque(seed_nodes)
  diffusion_edges = []
  diffusion_timestamp_map = {}
  for node in seed_nodes:
    diffusion_timestamp_map[node] = 0
  while queue:
    node = queue.popleft()
    for neighbor in G.neighbors(node):
      if neighbor not in activated_nodes:
        if rng.random() < G[node][neighbor]['weight']:
          activated_nodes.add(neighbor)
          queue.append(neighbor)
          diffusion_edges.append((node, neighbor))
          diffusion_timestamp_map[neighbor] = diffusion_timestamp_map[node] + 1
  
  n = G.number_of_nodes()
  diffusion_timestamp_temp = [[] for _ in range(n)]
  diffusion_timestamp = []
  for k, v in diffusion_timestamp_map.items():
    diffusion_timestamp_temp[v].append(k)

  for t in range(len(diffusion_timestamp_temp)):
    if len(diffusion_timestamp_temp[t]) > 0:
      diffusion_timestamp.append(diffusion_timestamp_temp[t])

  return diffusion_edges, diffusion_timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to folder containing a graph.mtx file with ground truth IC probabilities")
    parser.add_argument("-n", "--ncascades", type=int, 
        help="Number of cascades to generate (default: %(default)s)", default=250)
    parser.add_argument("-s", "--seedsize", type=int, 
        help="Size of seed set to start cascades from (default: %(default)s)", default=1)
    args = parser.parse_args()

    path = Path(args.filepath)
    gname = args.filepath.split('/')[-2]

    p_ts = path / "diffusions/timestamps/"
    p_edges = path / "diffusions/edges/"

    p_ts.mkdir(parents=True, exist_ok=True)
    p_edges.mkdir(parents=True, exist_ok=True)

    with open(path / f"graph.mtx", "rb") as fh:
        G = nx.from_scipy_sparse_array(sp.io.mmread(fh), create_using=nx.DiGraph)

    for i in range(args.ncascades):
        seed_nodes = [i.item() for i in rng.choice(list(G.nodes()), args.seedsize, replace=False)]

        d_edges, d_timestamp = monte_carlo_diffusion_nodes(G, seed_nodes)
        f_ts = p_ts / f"{i}.txt"
        f_edges = p_edges / f"{i}.edgelist"
        with f_ts.open("w") as fh:
            for ts in d_timestamp:
                fh.write(" ".join(map(str, ts)) + "\n")
        with f_edges.open("w") as fh:
            fh.write(f"#Source Target\n")
            for e in d_edges:
                fh.write(" ".join(map(str, e)) + "\n")