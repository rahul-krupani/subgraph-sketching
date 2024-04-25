from ogb.linkproppred import LinkPropPredDataset
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import gc
import time
import torch
from GraphRicciCurvature.FormanRicci import FormanRicci

import torch
import networkx as nx
from torch_geometric.datasets import Planetoid

start = time.time()
# Load a Planetoid dataset, for example, the "Cora" dataset
dataset = Planetoid(root='/home/resl/csci566/subgraph-sketching/dataset/Cora', name='Cora')

# Get the edge index from the data
edge_index = dataset[0].edge_index

# Initialize an empty NetworkX graph
G = nx.Graph()

# Add edges to the NetworkX graph
for i in range(edge_index.shape[1]):
    source = edge_index[0, i].item()
    target = edge_index[1, i].item()
    G.add_edge(source, target)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
print("Time taken: ",  time.time() - start)
print("Karate Club Graph: The  Ollivier-Ricci curvature of edge (0,1) i ", orc.G)

'''
# Load an OGBL dataset, for example, ogbl-collab
dataset = LinkPropPredDataset(name='ogbl-vessel')

# Get the split edge
split_edge = dataset.get_edge_split()

# Create a NetworkX graph from the edge list
G = nx.Graph()
train_edges = split_edge['train']['edge']
print("Converting Graph")
G.add_edges_from(train_edges[:500000])
print("Finished Converting 1")
G.add_edges_from(train_edges[500000:1000000])
print("Finished Converting 2")
G.add_edges_from(train_edges[1000000:1500000])
print("Finished Converting 3")
G.add_edges_from(train_edges[1500000:2000000])
print("Finished Converting 4")
G.add_edges_from(train_edges[2000000:2500000])
print("Finished Converting 5")
G.add_edges_from(train_edges[2500000:3000000])
print("Finished Converting 6")
G.add_edges_from(train_edges[3000000:3500000])
print("Finished Converting 7")
G.add_edges_from(train_edges[3500000:4000000])
print("Finished Converting 8")
G.add_edges_from(train_edges[4000000:])
print("Finished Converting 9")
# Optionally, add node features
# This part depends on the dataset and your specific needs
# node_features = dataset[0]['node_feat']
# for i, feat in enumerate(node_features):
#     G.nodes[i]['feature'] = feat

#print("Number of nodes:", G.number_of_nodes())
#print("Number of edges:", G.number_of_edges())
gc.collect()

# print("\n- Import an example NetworkX karate club graph")
# G = nx.karate_club_graph()
print("\n===== Compute the Ollivier-Ricci curvature of the given graph G =====")
# compute the Ollivier-Ricci curvature of the given graph G
# orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc = OllivierRicci(G, proc=4, alpha=0.5, verbose="INFO", shortest_path="pairwise")
# orc.compute_ricci_?curvature()
start = time.time()
out = orc.compute_ricci_curvature_edges([[378, 255038]])
print("TIme taken: ",  time.time() - start)
print("Karate Club Graph: The  Ollivier-Ricci curvature of edge (0,1) i ", out)
'''

# print("\n===== Compute the Forman-Ricci curvature of the given graph G =====")
# frc = FormanRicci(G)
# frc.compute_ricci_curvature()
# print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % frc.G[0][1]["formanCurvature"])