import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GENConv, DeepGCNLayer, GATConv, GlobalAttention, global_max_pool
from utils import Self_Attn

import vgg

class genconv_mg(nn.Module):
    def __init__(self, num_features, hidden_size, node_num_classes, graph_num_classes, n_layers,
                 graph_transforms=None, patch_size=None, device=None):
        super().__init__()
        self.graph_transforms = graph_transforms
        self.patch_size = patch_size
        self.device = device

        self.patch_cnn = vgg.vgg8(classify=False)
        # self.pos_encoder = nn.Linear(2, hidden_size//8)

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

        self.node_att = Self_Attn(in_dim=len(graph_transforms), activation=nn.ReLU())
        self.linear_node1 = nn.Linear(hidden_size * len(graph_transforms), hidden_size * len(graph_transforms))
        self.linear_node2 = nn.Linear(hidden_size*len(graph_transforms), node_num_classes)

        self.graph_att = Self_Attn(in_dim=len(graph_transforms), activation=nn.ReLU())
        # self.att_pool = GlobalAttention(nn.Linear(hidden_size * len(graph_transforms), 1))
        self.linear_graph1 = nn.Linear(hidden_size * len(graph_transforms), hidden_size * len(graph_transforms))
        self.linear_graph2 = nn.Linear(hidden_size * len(graph_transforms), graph_num_classes)

    def forward(self, patch_tensor, coord_tensor):
        patch_tensor = patch_tensor.view(-1, 1, self.patch_size, self.patch_size)
        patch_feature = self.patch_cnn(patch_tensor)
        patch_feature = patch_feature.view(patch_feature.size(0), -1)

        # pos_embedding = self.pos_encoder(coord_tensor)

        # node_feature = torch.cat([patch_feature, pos_embedding], dim=1)
        node_feature = torch.cat([patch_feature, coord_tensor], dim=1)
        data = Data(x=node_feature, pos=coord_tensor)

        embeddings = []

        for graph_transform in self.graph_transforms:

            data = graph_transform(data)
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

            embeddings.append(x)

        embeddings = torch.stack(embeddings).permute(1, 0, 2)

        node_pred = self.node_att(embeddings).view(patch_tensor.size(0), -1)
        node_pred = self.layers[0].act((self.linear_node1(node_pred)))
        node_pred = self.linear_node2(node_pred)
        node_pred = F.log_softmax(node_pred, dim=1)

        graph_pred = self.graph_att(embeddings).view(patch_tensor.size(0), -1)
        graph_pred = global_mean_pool(graph_pred, batch)
        # graph_x2 = global_
        # graph_pred = self.att_pool(embeddings.reshape(patch_tensor.size(0), -1), batch)
        graph_pred = self.layers[0].act(self.linear_graph1(graph_pred))
        graph_pred = self.linear_graph2(graph_pred)
        graph_pred = F.log_softmax(graph_pred, dim=1)

        return node_pred, graph_pred