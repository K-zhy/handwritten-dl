import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 随机种子，保证每次一样
torch.manual_seed(0)

# 数据
x = torch.rand(1, 3)
y = torch.rand(1, 3)
print('x', x)
print('y', y)

# 定义网络，继承自 nn.Module
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 3)   # 输入3维，输出3维
        self.fc2 = nn.Linear(3, 3)   # 输入3维，输出3维
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

# 实例化模型
model = SimpleNN()

# 损失函数
criterion = nn.MSELoss()

# 优化器（让PyTorch自动更新参数）
optimizer = optim.SGD(model.parameters(), lr=0.3)

# 训练
epochs = 520
loss_history = []

for epoch in range(epochs):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = criterion(y_pred, y)
    loss_history.append(loss.item())

    # 反向传播+优化
    optimizer.zero_grad()   # 梯度清零
    loss.backward()         # 反向传播
    optimizer.step()        # 自动更新参数

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print(f"Final Epoch {epoch+1}, Loss: {loss.item()}, Pred: {y_pred.detach().numpy()}")

# 画图
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
