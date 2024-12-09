import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear, SAGEConv, GATv2Conv, GATConv
import torch.nn.functional as F
import time
import argparse

import networkx as nx
import numpy as np
from pathlib import Path
import scipy as sp
rng = np.random.default_rng()

class CascadeGNN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_dim=64, hidden_dim=16, num_layers=2):
        super(CascadeGNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial node embedding layer
        self.node_embedding = nn.Embedding(self.num_nodes, self.node_dim)
        #self.edge_emb = nn.Parameter(torch.Tensor(num_nodes, num_nodes, hidden_dim))
        self.edge_embedding = nn.Embedding(self.num_edges, self.hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList([
            GATConv(self.node_dim if i == 0 else hidden_dim, hidden_dim) 
            for i in range(num_layers)
        ])
            
        # Edge probability prediction layer
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, edge_index):
        # Get initial node embeddings
        x = self.node_embedding(torch.arange(self.num_nodes))
        #src, dst = edge_index
        #edge_arr = self.edge_emb[src, dst]
        edge_emb = self.edge_embedding(torch.arange(self.num_edges))

        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = F.gelu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
        # Compute edge probabilities for all edges
        edge_probabilities = {}
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            #edge_repr = torch.cat([torch.add(x[source], x[target]), edge_emb[i]], dim=0)
            edge_repr = torch.cat([x[source], x[target], edge_emb[i]], dim=0)
            prob = self.edge_mlp(edge_repr)
            edge_probabilities[(source.item(), target.item())] = prob
        
        return edge_probabilities

def compute_cascade_likelihood(edge_probs, adj_list, cascade, eps=1e-6):
    """
    Compute the negative log likelihood of observing a cascade given edge probabilities
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_probs: Dictionary mapping (source, target) tuples to probabilities
        cascade: List of lists, where cascade[i] contains nodes activated at time i
        eps: Small value to prevent log(0)
    
    Returns:
        Negative log likelihood of the cascade
    """
    log_likelihood = 0.0
    activated_nodes = set()
    
    # Process each time step
    for t in range(len(cascade)):
        prev_activated = cascade[t-1] if t-1 >= 0 else []
        curr_activated = cascade[t]
        next_activated = cascade[t+1] if t+1 < len(cascade) else []
        activated_nodes.update(curr_activated)

        #print(t)
        #print(prev_activated)
        #print(curr_activated)
        #print(next_activated)

        for v in curr_activated:
            # Probability of activation from parents
            if prev_activated:
                #parents = set([u for u in range(num_nodes) if (u, v) in edge_probs and u in prev_activated])
                parents = adj_list[v][0]
                activated_parents = parents.intersection(set(prev_activated))
                prob = [1 - edge_probs[(u, v)] for u in activated_parents]
                prob = torch.cat(prob)
                prob_not_activated = torch.prod(prob)
                log_likelihood += torch.log(1 - prob_not_activated + eps)
            if next_activated:
                children = adj_list[v][1]
                non_activated_children = children.difference(activated_nodes).difference(set(next_activated))
                #children = set([w for w in range(num_nodes) if (v, w) in edge_probs and w not in activated_nodes and w not in set(next_activated)])
                if not non_activated_children:
                    continue
                prob = [1 - edge_probs[(v, w)] for w in non_activated_children]
                prob = torch.cat(prob)
                #print(prob)
                prob_not_activated = torch.prod(prob)
                log_likelihood += torch.log(prob_not_activated + eps)
    
    return log_likelihood

def compute_loss(edge_probs, adj_list, cascades, eps=1e-6):
  """
  Compute the negative log-likelihood loss for multiple cascades.
  
  Args:
    num_nodes: Number of nodes in the
    edge_probs: Tensor of predicted edge probabilities
    edge_index: Tensor of shape [2, num_edges] containing edge indices
    cascades: List of cascades, where each cascade is a list of lists of activated nodes
  
  Returns:
    loss: Negative log-likelihood loss
  """
  total_log_likelihood = 0.0
  for cascade in cascades:
    total_log_likelihood += compute_cascade_likelihood(edge_probs, adj_list, cascade, eps=eps)
  
  # Return negative log-likelihood as the loss
  #print(-total_log_likelihood)
  return -total_log_likelihood

def train_cascade_gnn(model, edge_index, cascades, adj_list, num_epochs=100, batch_size = 50, lr=0.001, verbose=True):
    """
    Train the GNN model using the observed cascades
    
    Args:
        model: CascadeGNN model
        edge_index: Tensor of shape [2, num_edges] containing edge indices
        cascades: List of cascades, where each cascade is a list of lists
        adj_list: Adjaceny List of the graph
        num_epochs: Number of training epochs (default = 100)
        batch_size: Size of batch used to train (default = 50)
        lr: Learning rate (default = 0.001)
        verbose: Print loss function info over epochs (default = True)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    for epoch in range(num_epochs):
        loss = 0.0
        rng.shuffle(cascades)
        batches = [cascades[i:i+batch_size] for i in range(0, len(cascades), batch_size)]
        optimizer.zero_grad()

        for batch in batches:
            optimizer.zero_grad()
            # Get edge probabilities
            edge_probs = model.forward(edge_index)

            batch_loss = compute_loss(edge_probs, adj_list, batch)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

def create_dataset(G: nx.DiGraph):
  edge_index = torch.tensor(list(G.edges)).t().contiguous()
  adj_list = {v : (set(), set()) for v in G.nodes}
  for e in G.edges:
    u, v = e
    adj_list[u][1].add(v)
    adj_list[v][0].add(u)
  return edge_index, adj_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to folder containing graph and cascades")
    parser.add_argument("-i", "--kmin", type=int, 
        help="Minimum number of training cascades (default: %(default)s)", default=50)
    parser.add_argument("-a", "--kmax", type=int, 
        help="Maximum number of training cascades (default: %(default)s)", default=250)
    parser.add_argument("-d", "--delta", type=int, 
        help="Runs method with [kmin, kmin+delta, kmin+2*delta, ..., kmax] training cascades (default: %(default)s)", 
        default=50)
    parser.add_argument("-e", "--epochs", type=int, 
        help="Number of epochs to train for (default: %(default)s)", 
        default=35)
    parser.add_argument("-l", "--learningrate", type=int, 
        help="Learning rate for training (default: %(default)s)", 
        default=0.01)
    parser.add_argument("-n", "--numlayers", type=int, 
        help="Number of GAT layers in the GNN (default: %(default)s)", 
        default=2)
    parser.add_argument("-b", "--embeddingdim", type=int, 
        help="The dimension of the edge embeddings (default: %(default)s)", 
        default=16)
    args = parser.parse_args()

    path = Path(args.filepath)

    with open(path / f"graph.mtx", "rb") as fh:
        G = nx.from_scipy_sparse_array(sp.io.mmread(fh), create_using=nx.DiGraph)

    cascades = []
    for i in range(args.kmax):
        with open(path / f"diffusions/timestamps/{i}.txt", "r") as fh:
            cascade = []
            for line in fh:
                cascade.append(list(map(int, line.strip().split())))
                cascades.append(cascade)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    edge_index, adj_list = create_dataset(G)

    l1_errors = []
    times = []
    cascade_sizes = range(args.kmin, args.kmax+1, args.delta)

    torch.manual_seed(28)
    results_path = Path(f"results/gnn/")
    gname = args.filepath.split('/')[-2]
    print(f"Graph: {gname}")
    print("M\tMAE\t\t\tTime")
    with open(results_path / "gnn.txt", "a") as fh:
        fh.write(f"\nGraph: {gname}\n")
        fh.write("M\t\tMAE\t\t\t\t\t\tTime\n")

    for k in cascade_sizes:
        start = time.time()
        model = CascadeGNN(n, m, hidden_dim=args.embeddingdim, num_layers=args.numlayers)
        #trained_model = 
        train_cascade_gnn(model, edge_index, cascades[:k], adj_list, num_epochs=args.epochs, lr=args.learningrate, verbose=False)
        end = time.time()
        times.append(end-start)

        model.eval()
        edge_probs = model(edge_index)
        residuals = []
        H = nx.DiGraph()
        for i, e in enumerate(G.edges()):
            u, v = e
            p = G[u][v]['weight']
            residuals.append(abs(p - edge_probs[e].item()))
            H.add_edge(u, v, weight=edge_probs[e].item())

        l1_err = sum(residuals) / len(residuals)
        l1_errors.append(l1_err)
        print(f"{k}\t{l1_err}\t{end-start}")
        with open(results_path / "gnn.txt", "a") as fh:
            fh.write(f"{k}\t\t{l1_err}\t\t{end-start}\n")

        graph_path = results_path / f"{gname}_{k}.mtx"
        with graph_path.open('wb') as fh:
            mat = nx.to_scipy_sparse_array(H)
            sp.io.mmwrite(fh, mat, precision=5)