
import sys
import random

sys.setrecursionlimit(15000)#15000
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
import os
import matplotlib
from sklearn import metrics
matplotlib.use('Agg')
from data_processing import load_data
from MyModel import mycnn

np.random.seed(1377)
device = torch.device('cpu')
EPOCH = 50
Margin = 2
SAVE_DIR = 'cnn_model_02'
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

filename = 'data/A.thaliana.xlsx'

[X_train, y_train, X_valid, y_valid, X_test, y_test] = load_data(filename)

X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = torch.cat((y_train, y_valid), axis=0)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)


import torch.utils.data as Data

train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
batch_size = 64

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=Margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(self.margin - euclidean_distance, 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)

    for i in range(int(batch_size)-1):
        seq1, label1 = batch[i][0], batch[i][1]
        for j in range(i+1, int(batch_size)):
            seq2, label2 = batch[j][0], batch[j][1]
            label1_ls.append(label1.unsqueeze(0))
            label2_ls.append(label2.unsqueeze(0))
            label = (label1 ^ label2)  # 异或, 相同为 0 ,相异为 1
            seq1_ls.append(seq1.unsqueeze(0))
            seq2_ls.append(seq2.unsqueeze(0))
            label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)

    return seq1, seq2, label, label1, label2


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


model = mycnn()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')




def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n




net = mycnn().to(device)

lr = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
criterion = ContrastiveLoss()
criterion_model = nn.CrossEntropyLoss(reduction='mean')
best_auc = 0






for epoch in range(EPOCH):
    loss_ls = []
    loss1_ls = []
    loss2_3_ls = []
    t0 = time.time()
    net.train()
    for seq1, seq2, label, label1, label2 in train_iter_cont:

        output1 = net(seq1)
        output2 = net(seq2)
        output3 = net.trainModel(seq1)
        output4 = net.trainModel(seq2)
        loss1 = criterion(output1, output2, label)
        loss2 = criterion_model(output3, label1)
        loss3 = criterion_model(output4, label2)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
        loss1_ls.append(loss1.item())
        loss2_3_ls.append((loss2 + loss3).item())

    net.eval()
    with torch.no_grad():
        train_acc = evaluate_accuracy(train_iter, net)
        test_acc = evaluate_accuracy(test_iter, net)

    outputs = net.trainModel(X_test.to(device))
    outputs = outputs.cpu().detach().numpy()
    predict = np.vstack(outputs[:, 1])

    auc = metrics.roc_auc_score(y_test, predict)



    results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
    results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, test_auc: {colored(auc, "blue")}, time: {time.time() - t0:.2f}'
    print(results)
    if auc > best_auc:
        best_auc = auc
        print(f"best_auc: {best_auc}")

        MODEL_name = str(epoch) +"test_auc" + str(auc) + ".pl"
        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, MODEL_name)
        torch.save({"best_auc": best_auc, "model": net.state_dict()}, MODEL_SAVE_PATH)
print("down")