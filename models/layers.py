import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class GCNConv(MessagePassing):

    def __init__(self, in_features_dim, out_features_dim, edge_features_dim, aggr="add", flow="target_to_source"):
        super().__init__(aggr=aggr, flow=flow)
        self.alpha = torch.nn.Parameter(torch.Tensor(edge_features_dim))
        self.W = torch.nn.Linear(in_features_dim, out_features_dim)
        self.reset_parameters()

    def forward(self, batch_graph):
        h = batch_graph.h
        Wh = self.W(h)
        feats = batch_graph.feats
        edge_index = batch_graph.edge_index
        # for idx, i in enumerate(range(edge_index.size(1))):
        #     print(idx, ':', edge_index[0, i], edge_index[1, i])
        # print(edge_index.size(1))
        # edge_copy = edge_index.clone().detach().cpu()
        # edge_copy = edge_copy.numpy().transpose()
        # strs = []
        # for item in edge_copy:
        #     strs.append('{}_{}'.format(item[0], item[1]))
        # for item in strs:
        #     a, b = item.split('_')
        #     item2 = '{}_{}'.format(b, a)
        #     if item2 not in strs:
        #         print(item, ', but {} not in'.format(item2))
        #     if item2 in strs:
        #         print(item, ', and {} in'.format(item2))
        # for item in edge_copy:
        #     if item[::-1] in edge_copy:
        #         # print('true', item, item[::-1])
        #         continue
        #     else:
        #         print('false', item)
        # print(edge_copy, type(edge_copy))
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index)
        return self.propagate(edge_index=edge_index, feats=feats, h=Wh, num_nodes=h.size(0))

    def message(self, feats_i, feats_j, h_j):
        # edge attention
        edge_features = feats_i * feats_j
        alpha_ij = torch.sum(edge_features * self.alpha, dim=1, keepdim=True)
        return alpha_ij * h_j

    def reset_parameters(self):
        torch.nn.init.uniform_(self.a)
