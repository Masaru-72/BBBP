import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import networkx as nx

# Load dataset
data = pd.read_excel('bagging_previous_code.xlsx')
data.rename(columns={'Likelihood of belonging to the positive class': 'Likelihood'}, inplace=True)
data['Label'] = np.where(data['Likelihood'] > 0.5, 1, 0)

# Convert SMILES to graphs
def smiles_to_graph(smiles_list, label_list):
    graph_list = []

    for smiles, label in zip(smiles_list, label_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
            continue  # Skip invalid molecules

        # Create graph
        G = nx.Graph()
        atom_features = []

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
            atom_features.append(atom.GetAtomicNum())  # Atomic number as feature

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            G.add_edge(i, j)
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected graph
            edge_attr.append(bond.GetBondTypeAsDouble())  # Bond type feature
            edge_attr.append(bond.GetBondTypeAsDouble())

        # Convert to PyTorch tensors
        x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        # Create PyTorch Geometric Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float))
        graph_list.append(graph)

    return graph_list

# Convert SMILES to graphs
graph_list = smiles_to_graph(data['SMILES'].tolist(), data['Label'].astype(int).tolist())

# Ensure dataset is not empty
if len(graph_list) == 0:
    raise ValueError("Error: No valid molecules found in dataset.")

# Split dataset
train_ratio = 0.80
dataset_size = len(graph_list)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator1)

# Ensure DataLoader receives non-empty dataset
if len(train_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("Error: Train or test dataset is empty.")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=1, hidden_channels=128, out_channels=1).to(device)  # in_channels=1 due to atomic number feature
optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5, weight_decay=10**-5)

def train():
    total_loss = correct_predictions = 0
    total_samples = 0
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        out = out.view(-1, 1)
        data.y = data.y.view(-1, 1)

        loss = F.binary_cross_entropy_with_logits(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
        total_samples += data.num_graphs
        preds = (torch.sigmoid(out) > 0.5).float()  # Apply sigmoid for predictions
        correct_predictions += (preds == data.y).sum().item()

    return total_loss / total_samples, correct_predictions / total_samples

@torch.no_grad()
def test(loader):
    total_loss = correct_predictions = 0
    total_samples = 0
    model.eval()

    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out.view(-1, 1)
        data.y = data.y.view(-1, 1)

        loss = F.binary_cross_entropy_with_logits(out, data.y, reduction='sum')
        total_loss += float(loss)
        total_samples += data.num_graphs
        preds = (torch.sigmoid(out) > 0.5).float()  # Apply sigmoid for predictions
        correct_predictions += (preds == data.y).sum().item()

    return total_loss / total_samples, correct_predictions / total_samples

# Train model
score_train_loss, score_train_acc = [], []
score_test_loss, score_test_acc = [], []
epochs = 75

for epoch in range(epochs):
    train_loss, train_acc = train()
    test_loss, test_acc = test(test_loader)
    score_train_loss.append(train_loss)
    score_train_acc.append(train_acc)
    score_test_loss.append(test_loss)
    score_test_acc.append(test_acc)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), score_train_loss, c='goldenrod')
plt.plot(range(epochs), score_test_loss, c='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Test Loss'])

plt.subplot(1, 2, 2)
plt.plot(range(epochs), score_train_acc, c='goldenrod')
plt.plot(range(epochs), score_test_acc, c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Test Accuracy'])

plt.show()
