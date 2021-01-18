import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import degree

print(torch_geometric.__version__)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='mean', node_dim=-4)  # "Add" aggregation (Step 5).
        #self.lin = torch.nn.Linear(in_channels, out_channels)

        self.conv =  nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True), nn.BatchNorm3d(out_channels),nn.LeakyReLU())

    def forward(self, x, edge_index):

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)
        x = self.conv(x)
        print("x shape: {}".format(x.shape))

        # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(1), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j has shape [E, out_channels]
        # print(norm)
        # print(norm.shape)
        # print(x_j.shape)
        # new_x_j = torch.reshape(x_j, shape=(128,64))
        # msg = norm.view(-1, 1) * new_x_j
        # Step 4: Normalize node features.
        return x_j


#mygcnconv = GCNConv(8, 16)
mygcnconv = GCNConv(8, 4)

#mygcnconvop = mygcnconv(torch.randn(size=(8, 8)), torch.randint(0, 7, (2, 128), dtype=torch.long))
mygcnconvop = mygcnconv(torch.randn(size=(1, 8, 2,2,2)), torch.randint(0, 7, (2, 128), dtype=torch.long))
print(mygcnconvop.shape)

"""
x shape: torch.Size([8, 16])
size, dim, data_shape: [None, None], 0, torch.Size([8, 16])
args: {'x_j', 'norm'}
node_dim, index: -2, tensor([6, 6, 2, 5, 4, 6, 6, 5, 4, 3, 1, 5, 4, 4, 3, 3, 1, 5, 6, 6, 0, 6, 0, 2,
        4, 5, 4, 1, 1, 6, 3, 1, 6, 1, 3, 2, 2, 3, 6, 5, 5, 3, 4, 4, 0, 0, 2, 0,
        2, 1, 2, 1, 1, 4, 6, 1, 1, 2, 5, 2, 2, 4, 2, 2, 1, 2, 3, 5, 6, 3, 1, 1,
        4, 0, 1, 2, 0, 1, 1, 6, 0, 3, 1, 1, 3, 5, 6, 2, 3, 3, 6, 4, 0, 5, 6, 2,
        5, 2, 0, 4, 4, 5, 0, 5, 5, 4, 0, 1, 4, 3, 1, 4, 3, 0, 2, 5, 4, 3, 5, 3,
        2, 2, 2, 4, 5, 5, 2, 4])
torch.Size([128, 16])
torch.Size([128])
torch.Size([128, 16])

size, dim, data_shape: [None, None], 0, torch.Size([1, 64, 4, 4, 4])
size, dim, data_shape: [None, None], 0, torch.Size([64, 16])
args: {'norm', 'x_j'}


======================

Dataset: Cora():
======================
Number of graphs: 1
Number of features: 1433
Number of classes: 7
Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
==============================================================
Number of nodes: 2708
Number of edges: 10556
Average node degree: 3.90
Number of training nodes: 140
Training node label rate: 0.05
Contains isolated nodes: False
Contains self-loops: False
Is undirected: True
torch.Size([2708, 8, 8, 8])
size, dim, data_shape: [None, None], 0, torch.Size([2708, 8, 8, 8])
args: {'x_j'}
node_dim, index: 0, tensor([   0,    0,    0,  ..., 2707, 2707, 2707])
torch.Size([10556, 8, 8, 8])
torch.Size([2708, 8, 8, 8])

"""




