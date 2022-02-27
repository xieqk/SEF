import torch

from .layers import GCNConv


class SEF(torch.nn.Module):

    def __init__(self, dropout=0.0, body_feats_dim=2048, face_feats_dim=512):
        super().__init__()

        print('\n SEF ... \n')
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 32), 
            torch.nn.BatchNorm1d(32),
            torch.nn.PReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.w_0 = torch.nn.Linear(32, 1)

        self.gcn_body = GCNConv(32, 32, body_feats_dim)
        self.gcn_face = GCNConv(32, 32, face_feats_dim)
        # self.conv_dropout = torch.nn.Dropout(p=dropout)
        self.w_b = torch.nn.Linear(32, 1)
        self.w_f = torch.nn.Linear(32, 1)

    def forward(self, graph_body, graph_face):
        embeddings = self.input_mlp(graph_body.x)
        graph_body.x = embeddings
        graph_face.x = embeddings

        graph_body.x = self.gcn_body(graph_body)
        graph_face.x = self.gcn_face(graph_face)

        linear_scores = self.w_0(embeddings).squeeze()
        gcn_body_scores = self.w_b(graph_body.x).squeeze()
        gcn_face_scores = self.w_f(graph_face.x).squeeze()

        return linear_scores + gcn_body_scores + gcn_face_scores
