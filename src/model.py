"""
Graph Neural Network model & training with early stopping.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_communities, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_communities)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_model_with_early_stopping(model, data: Data, patience: int, num_epochs: int):
    """
    Train `model` on `data`, stopping if validation loss doesn’t improve
    for `patience` epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        model.eval()
        val_out = model(data.x, data.edge_index)
        val_loss = F.cross_entropy(val_out, data.y).item()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model
