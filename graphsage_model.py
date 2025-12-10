import torch
import numpy as np
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import kneighbors_graph

def build_knn_graph(X, k=5):
    knn = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.vstack(knn.nonzero()), dtype=torch.long)
    return edge_index

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=32, out_channels=16):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def generate_graphsage_embeddings(X_train, epochs=50):
    if hasattr(X_train, "values"):
        X_train = X_train.values
    X_tensor = torch.tensor(X_train.astype(np.float32))
    edge_index = build_knn_graph(X_train, k=5)

    model = GraphSAGE(in_channels=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(X_tensor, edge_index)
        loss = embeddings.norm(2).mean()
        loss.backward()
        optimizer.step()

    return embeddings.detach().numpy()
