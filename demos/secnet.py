

import sys
import torch.nn as nn
import os
from pathlib import Path
import torch
sys.path.append(os.path.dirname(__file__))

# from modified_optimizer import customAdam
# from lightning.pytorch import seed_everything
# from ho import *
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
current_module = sys.modules[__name__]

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from metabci.brainda.algorithms.deep_learning.secnet import BaseModel
from metabci.brainda.algorithms.deep_learning.utils import *
import sys
current_module = sys.modules[__name__]

# 1. 加载数据
demo_data_dir = Path(__file__).resolve().parent / "data" / "BCIIV2a"
train_data_path = str(demo_data_dir / "A01T.mat")
test_data_path = str(demo_data_dir / "A01E.mat")

train_data = sio.loadmat(train_data_path)
test_data = sio.loadmat(test_data_path)

X_train = train_data['x_data']  #注意，这里数据维度为(样本数，通道数，时间采样点)
y_train = train_data['y_data'].reshape(-1)


print('X_train',X_train.shape)
print('y_train',y_train.shape)
X_test = test_data['x_data']
y_test = test_data['y_data'].reshape(-1)


print('X_test',X_test.shape)
print('y_test ',y_test .shape)
# 将数据转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



configs = {
        'class_num': 4,      # 分类数量
        'channelNum': 22,     # 通道数量
        'width': 300,         # 输入宽度
        'drop_att': 0.2,      # 注意力机制的 dropout 概率
        'p': 3                # 其他自定义参数
    }

model = BaseModel(configs)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()


num_epochs = 100
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"Input device: {inputs.device}, Label device: {labels.device}")
        optimizer.zero_grad()
        outputs,x1= model(inputs)
        loss = criterion(outputs, labels)
        # centloss = centerloss(labels, f)
        # loss = loss + 0.000005 * centloss
        loss.backward()
        optimizer.step()
        # optimzer4center.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,x1 = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
            all_outputs.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
