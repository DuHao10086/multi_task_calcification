import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GENConv, DeepGCNLayer, GATConv
from utils import Self_Attn

import vgg

class genconv_morph(nn.Module):
    def __init__(self, num_features, hidden_size, node_num_classes, graph_num_classes, n_layers,
                 graph_transform=None, patch_size=None, device=None):
        super().__init__()
        self.graph_transform = graph_transform
        self.patch_size = patch_size
        self.device = device

        self.patch_cnn = vgg.vgg8(classify=False)

        self.node_encoder = nn.Linear(num_features, hidden_size)
        self.edge_encoder = nn.Linear(2, hidden_size)

        self.layers = nn.ModuleList()
        for i in range(1, n_layers + 1):
            conv = GENConv(hidden_size, hidden_size, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=False)
            self.layers.append(layer)

        self.node_att =Self_Attn(in_dim=1, activation=nn.ReLU())
        self.linear_node = nn.Linear(hidden_size, node_num_classes)

    def forward(self, patch_tensor, coord_tensor):
        patch_tensor = patch_tensor.view(-1, 1, self.patch_size, self.patch_size)
        patch_feature = self.patch_cnn(patch_tensor)
        patch_feature = patch_feature.view(patch_feature.size(0), -1)

        node_feature = torch.cat([patch_feature, coord_tensor], dim=1)
        data = Data(x=node_feature, pos=coord_tensor)

        if self.graph_transform is not None:
            data = self.graph_transform(data)

        batch_data = Batch()
        batch_data = batch_data.from_data_list([data]).to(self.device)

        x, edge_index, batch, edge_attr = batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.edge_attr

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        node_att_out = self.node_att(x)
        node_pred = self.linear_node(node_att_out)
        node_pred = F.log_softmax(node_pred, dim=1)
        return node_pred