import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import networkx as nx
from math import sqrt
import torch
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from torch_geometric.utils import from_smiles
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools, Descriptors

data = pd.read_excel('bagging_previous_code.xlsx')
data.rename(columns={'Likelihood of belonging to the positive class':'Likelihood'}, inplace=True)
data['Label'] = np.where(data['Likelihood'] > 0.5, 1, 0)

data.head()


graph_list = []

smiles_list = data['SMILES'].tolist()  # Convert SMILES column to a list
label_list = data['Label'].astype(int).tolist()

valid_smiles = [Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list]
print(valid_smiles)


# In[107]:


for smile, label in zip(smiles_list, label_list):

    try:
        # Convert SMILES to graph
        g = from_smiles(smile)
        g.x = g.x.float()  # Ensure node features are float
        g.y = torch.tensor([label], dtype=torch.float)  # Assign label correctly

        graph_list.append(g)
    except Exception as e:
        print(f"Error processing SMILES {smile}: {e}")


# In[108]:


from torch_geometric.utils import to_networkx

def draw_graphs_from_list(graph_list, smiles_list=None, num_to_draw=5):
    # Limit the number of graphs to draw
    graphs_to_draw = graph_list[:num_to_draw]
    smiles_subset = smiles_list[:num_to_draw] if smiles_list else [None] * num_to_draw

    # Create subplots
    fig, axes = plt.subplots(1, len(graphs_to_draw), figsize=(5 * len(graphs_to_draw), 5))

    # Ensure axes is iterable for a single graph case
    if len(graphs_to_draw) == 1:
        axes = [axes]

    for ax, g, smile in zip(axes, graphs_to_draw, smiles_subset):
        g.x = g.x.float()  # Ensure node features are float if necessary

        # Convert PyG graph to NetworkX
        nx_graph = to_networkx(g, node_attrs=['x'])

        pos = nx.spring_layout(nx_graph, seed=42)

        # Draw graph
        nx.draw(nx_graph, with_labels=True, node_color='skyblue', edge_color='gray', ax=ax)
        ax.set_title(smile if smile else "Graph")

    plt.show()


# In[109]:


draw_graphs_from_list(graph_list)


# Classifier

# In[111]:


train_ratio = 0.80  # 80% for training, 20% for testing
dataset_size = len(graph_list)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# Split the dataset into train and test subsets
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator1)


# In[112]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[113]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AttentiveFP(in_channels=9, hidden_channels=64, out_channels=1,
                    edge_dim=3, num_layers=4, num_timesteps=2,
                    dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
                             weight_decay=10**-5)


# In[117]:


def train():
    total_loss = total_samples = 0
    correct_predictions = 0
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Apply sigmoid activation for binary classification
        out = torch.sigmoid(out)

        # Reshape data.y to match out's shape
        # data.y = data.y.unsqueeze(1) #original
        data.y = data.y.view(-1, 1) #suggested change
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
        total_samples += data.num_graphs

        # Calculate number of correct predictions
        preds = (out > 0.5).float()
        correct_predictions += (preds == data.y).sum().item()

    accuracy = correct_predictions / (total_samples * data.y.size(1))  # Assuming data.y is of shape [batch_size, num_classes]
    return total_loss / total_samples, accuracy


# In[119]:


@torch.no_grad()
def test(loader):
    total_loss = total_samples = 0
    correct_predictions = 0
    model.eval()

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Apply sigmoid activation for binary classification
        out = torch.sigmoid(out)

        # Reshape data.y to have the same shape as out
        data.y = data.y.view(-1, 1)  # Reshape to [batch_size, 1]

        loss = F.binary_cross_entropy(out, data.y, reduction='sum')
        total_loss += float(loss)
        total_samples += data.num_graphs

        # Calculate number of correct predictions
        preds = (out > 0.5).float()
        correct_predictions += (preds == data.y).sum().item()

    accuracy = correct_predictions / (total_samples * data.y.size(1))  # Assuming data.y is of shape [batch_size, num_classes]
    average_loss = total_loss / total_samples
    return average_loss, accuracy


# In[ ]:


score_train_loss = []
score_train_acc = []
score_test_loss = []
score_test_acc = []

epochs = 75
model.reset_parameters()

for epoch in range(epochs):
    train_loss, train_acc = train()
    test_loss, test_acc = test(test_loader)

    score_train_loss.append(train_loss)
    score_train_acc.append(train_acc)
    score_test_loss.append(test_loss)
    score_test_acc.append(test_acc)

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

plt.figure(figsize=(12, 5))


