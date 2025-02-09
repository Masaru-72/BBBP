import numpy
import pandas
import torch
from torch_geometric.loader import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset
from model.util import *
import time
import random
from collections import defaultdict

# Device setup for CPU
device = torch.device('cpu')

start = time.time()

# Available models: AttFP, DimeNet++, SchNet
model_name = 'AttFP'

# Experiment settings.
dataset_name = 'bbbp_train'

# Number of iterations
n_bag = 100

task = 'clf'
n_folds = 5
batch_size = 64
init_lr = 0.001
l2_coeff = 5e-6
n_epochs = 500
dim_hidden = 128
n_gnn = 2
list_preds = list()
list_targets = list()
list_models = list()
list_acc = list()
list_f1 = list()
list_roc_auc = list()
dim_out, criterion = 2, torch.nn.CrossEntropyLoss()

# Load the dataset.
elem_attrs = load_elem_attrs('matscholar-embedding.json')
dataset = load_dataset(path_dataset='bbbp1_1k.xlsx'.format(dataset_name),
                       elem_attrs=elem_attrs,
                       idx_smiles=0,
                       idx_target=1,
                       task=task,
                       pu=True,
                       calc_pos=False
                       )
random.seed(0)
random.shuffle(dataset)

# Divide positive sample and unlabeled sample
pos = [dataset[i] for i in range(len(dataset)) if dataset[i].y == 1]
unk = [dataset[i] for i in range(len(dataset)) if dataset[i].y != 1]

# Data collection
coll_data = []
for bag in range(n_bag):
    random.seed(bag)

    # train : valid = 4:1
    valid_pos = random.sample(pos, int(0.2 * len(pos)))
    train_pos = [item for item in pos if item not in valid_pos]

    # Label some portion of unlabeled data as a negative sample
    start, stop = (bag * len(pos)) % len(unk), ((bag + 1) * len(pos)) % len(unk)
    neg = unk[start:stop] if start < stop else unk[start:] + unk[:stop]

    # Alternative method?, still working...
    # neg = random.sample(unk, len(pos))

    # train : valid = 4:1
    valid_neg = random.sample(neg, int(0.2 * len(neg)))
    train_neg = [item for item in neg if item not in valid_neg]

    # merge positive and (assumed) negative data
    train_tot = train_pos + train_neg
    valid_tot = valid_pos + valid_neg

    # Unlabeled data (w/o negative data) becomes test data
    unlabel = [item for item in unk if item not in neg]

    # print(len(train_tot), len(valid_tot), len(unlabel))
    coll_data.append((train_tot, valid_tot, unlabel))
    if bag % 10 == 9: print(f'Data partition: bagging {bag + 1}/{n_bag}')

def fit(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)  # Move data to the specified device
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)  # Move data to the specified device
            out = model(data)
            preds.append(out.cpu())  # Move predictions back to CPU if needed
            targets.append(data.y.cpu())  # Move targets back to CPU if needed
    return preds, targets

tot_dict_list = []
# valid_dict_list = []
for bag in range(n_bag):
    loader_train = DataLoader(coll_data[bag][0], batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(coll_data[bag][1], batch_size=batch_size)
    loader_test = DataLoader(coll_data[bag][2], batch_size=batch_size)
    model = get_model(model_name=model_name,
                      dim_node_feat=dataset[0].x.shape[1],
                      dim_edge_feat=dataset[0].edge_attr.shape[1],
                      dim_hidden=dim_hidden,
                      n_gnnlayer=n_gnn,
                      dim_out=dim_out)
    model = model.to(device)  # Move model to CPU
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)

    for epoch in range(n_epochs):
        loss_train = fit(model, loader_train, optimizer, criterion, device)

    preds_test, targets_test = test(model, loader_test, device)  # Pass device

    smiles_test = []
    labeling_dict = {}
    for i in range(len(coll_data[bag][2])):
        smiles_test.append(coll_data[bag][2][i].smls)
    if task == 'clf':
        # Stack the tensors in preds_test and convert to a NumPy array
        preds_test_array = torch.cat(preds_test, dim=0).numpy()
        max_preds_test = numpy.argmax(preds_test_array, axis=1)
        for i in range(len(smiles_test)):
            labeling_dict[smiles_test[i]] = max_preds_test[i]
    tot_dict_list.append(labeling_dict)
    if bag % 5 == 4: print(f'Training: bagging {bag + 1}/{n_bag}')

sums = defaultdict(int)
counts = defaultdict(int)
for d in tot_dict_list:
    for key, value in d.items():
        sums[key] += value
        counts[key] += 1

# Calculate the averages
averaged_dict = {key: sums[key] / counts[key] for key in sums}
df = pandas.DataFrame(list(averaged_dict.items()), columns=['SMILES', 'Likelihood of belonging to the positive class'])
df.to_excel('bagging_previous_code_new.xlsx', index=False)

end = time.time()
print(f'Time required for prediction:{(end - start) / 60:.2f} minutes')

print("Excel file created successfully.")