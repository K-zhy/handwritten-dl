import numpy as np

import matplotlib.pyplot as plt

x = np.random.rand(1, 3)
print('x',x)  
y = np.random.rand(1, 3)  # 10个样本的标签，二分类问题
print('y',y)  

# hidden layer 权重矩阵 & 偏置向量
w1 = np.random.rand(3, 3)  # 输入层到隐藏层的权重矩阵
b1 = np.random.rand(1, 3)  # 隐藏层的偏置向量

# output layer 权重矩阵 & 偏置向量
w2 = np.random.rand(3,3)
b2 = np.random.rand(1,3)

# 激活函数
class sigmoid:
    # 1/(1+exp(-x)) 输出范围为0-1，适用于二分类问题
    def __call__(self,x):
        return 1/(1+np.exp(-x))
    # 求导
    def gradient(self,x):
        return  x*(1-x)

class Mse:
    def __call__(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    def gradient(self,y_true,y_pred):
        return 2*(y_pred-y_true)/y_true.size



if __name__ == '__main__':
    sigmoid = sigmoid()
    # 前向传播
    z1 = np.dot(x, w1) + b1  # 隐藏层的加权和
    print(z1)  # 10个样本，3个隐藏单元的加权和
    a1 = sigmoid(z1)  # 隐藏层的激活值
    print(a1)  # 10个样本，3个隐藏单元的激活值

    z2 = np.dot(a1, w2) + b2  # 输出层的加权和

    lr = 0.3
    epochs = 520
    mse = Mse()
    loss_hisory = []
    for epoch in range(epochs):
        # 前向传播
        h = sigmoid(np.dot(x, w1) + b1)
        y_pred = sigmoid(np.dot(h, w2) + b2)

        loss = mse(y, y_pred)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        loss_hisory.append(loss)
        delta2 = mse.gradient(y, y_pred) * sigmoid.gradient(y_pred)
        delta1 = sigmoid.gradient(h) *  np.dot(delta2, w2.T)

        w2 = w2 - lr * np.dot(h.T, delta2)
        b2 = b2 - lr * delta2

        w1 = w1 - lr * np.dot(x.T, delta1)   
        b1 = b1 - lr * delta1
    
    print(f'epoch {epoch+1}, loss={loss}, pred={y_pred}')
    plt.plot(loss_hisory)
    plt.show()