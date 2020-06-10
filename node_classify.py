import torch
from torch_geometric.datasets import Planetoid

core_dataset = Planetoid(root='node_classify/cora', name='cora')
print(core_dataset)