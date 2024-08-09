from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.common_blocks import batch_norm
from torch_geometric.utils import dropout_adj
from Loss.calculate import *


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.dropout = args.dropout
        self.cached = True
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups

        self.DDCD = args.DDCD

        self.loss_DDCD = 0

        self.temperature = args.temperature
        self.alpha = args.alpha

        # build up the convolutional layers
        if self.num_layers == 1:
            self.layers_GCN.append(GCNConv(self.num_feats, self.num_classes, cached=self.cached, bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        else:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            for _ in range(self.num_layers - 2):
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))

        # build up the normalization layers
        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))

        self.group_func = torch.nn.Linear(self.dim_hidden, self.num_classes, bias=True)

    def forward(self, x, edge_index, y, train_mask=None, report_metrics=False):
        edge_index_t = edge_index
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index_t)
            x = self.layers_bn[i](x)

            if self.training:
                if i != self.num_layers - 1:
                    if self.DDCD:
                        self.loss_DDCD = self.loss_DDCD + calculate_loss(x[train_mask], y[train_mask], self.temperature, self.alpha)

            if i == self.num_layers - 2:
                if report_metrics:
                    self.inner_sim, self.intra_sim, self.r_class = class_sim(x, y)

            if i != self.num_layers - 1:
                x = F.relu(x)

        return x
