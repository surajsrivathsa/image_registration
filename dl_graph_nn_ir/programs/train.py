import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid


class MyConv(MessagePassing):
    def __init__(self):
        super(MyConv, self).__init__(aggr='mean', node_dim=0)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


dataset = Planetoid('/tmp/Planetoid', name='Cora')
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]
x = torch.randn(data.num_nodes, 8, 8, 8)

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Edge features dimensionality: {data.num_edge_features}')
print(f'Node features dimensionality: {data.num_node_features}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print("x shape: {}".format(x.shape))

model = MyConv()
out = model(x, data.edge_index)
print("out shape: {}".format(out.shape))