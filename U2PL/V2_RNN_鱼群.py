import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.utils.data as data
from RNN_model import Conv1DNet

# 从Excel文件中读取数据
excel_file = "./image.xlsx"
data_frame = pd.read_excel(excel_file).to_numpy()

# 划分训练集和测试集
X_train = data_frame[1:26, 1:6].astype(float)
y_train = data_frame[1:26, 6].astype(float)
y_train = y_train[:, np.newaxis]

X_test = data_frame[27:, 1:6].astype(float)
y_test = data_frame[27:, 6].astype(float)
y_test = y_test[:, np.newaxis]

# 定义超参数
input_size = 5
output_size = 1
learning_rate = 0.0001
num_epochs = 60
batch_size = len(y_train)
channels_1 = 32
channels_2 = 64

# 创建数据集和数据加载器
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train_dataset = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(X_test, y_test)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # 定义模型
# class Conv1DNet(nn.Module):
#     def __init__(self, input_size, output_size, channels_1, channels_2):
#         super(Conv1DNet, self).__init__()
#         self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=channels_1, kernel_size=2)
#         self.conv1d2 = nn.Conv1d(in_channels=channels_1, out_channels=channels_2, kernel_size=2)
#         self.fc1 = nn.Linear(channels_2 * 3, output_size)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv1d1(x.unsqueeze(1))
#         x = self.relu(x)
#         x = self.conv1d2(x)
#         x = self.relu(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         return x
#
#     def compute_loss(self, output, target):
#         loss = torch.linalg.norm(output - target, ord=1)
#         return loss
#
#     def compute_loss_t(self, output, target):
#         self.criterion_t = nn.MSELoss()
#         loss = self.criterion_t(output, target)
#         return loss

# 实例化模型
net = Conv1DNet(input_size, output_size, channels_1, channels_2)

# 初始化鱼群
num_fish = 10
fish = [np.random.randint(5, 128, size=(3,)) for _ in range(num_fish)]

# 优化循环
num_iterations = 10
step_size = 1
visual_range = 5

def get_loss(input_size, output_size, fish_x, fish_y):
    # 训练模型
    net = Conv1DNet(input_size, output_size, fish_x, fish_y)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

    # 评估模型
    net.eval()
    test_loss = 0.0

    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = net.compute_loss_t(outputs, labels)
        test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    return test_loss, fish_x, fish_y



for iteration in range(num_iterations):
    new_fish = []
    for i in range(num_fish):
        fish_x, fish_y, fish_loss = fish[i]

        test_loss, fish_x, fish_y = get_loss(input_size, output_size, fish_x, fish_y)

        # 随机选择邻近的鱼
        neighbor_index = random.randint(0, num_fish - 1)
        neighbor_x, neighbor_y, neighbor_loss = fish[neighbor_index]

        # 移动鱼
        new_x = fish_x + random.randint(-step_size, step_size)
        new_y = fish_y + random.randint(-step_size, step_size)

        test_loss_new, new_x, new_y = get_loss(input_size, output_size, new_x, new_y)

        if test_loss_new < test_loss:
            fish_x, fish_y = new_x, new_y
            test_loss = test_loss_new
        else:
            fish_x += random.randint(-visual_range, visual_range)

        new_fish.append([fish_x, fish_y, test_loss])

    fish = new_fish

    for i in range(len(fish)):
        for j in range(len(fish[i])):
            if fish[i][j] <= 0:
                fish[i][j] += 5

    for i in range(len(fish)):
        for j in range(len(fish[i])):
            if fish[i][j] >= 128:
                fish[i][j] -= 5

    print(fish)

last_column = [row[-1] for row in fish]
min_index = last_column.index(min(last_column))
min_row_indices = fish[min_index]

print(f"最优解：x = {min_row_indices[0]}, y = {min_row_indices[1]}")
print(f"最优解的适应度：{min_row_indices[2]}")

