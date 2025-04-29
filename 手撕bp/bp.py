import numpy as np

import random
import matplotlib.pyplot as plt
class sigmoid:
    # 1/(1+exp(-x))
    def __call__(self,x):
        return 1/(1+np.exp(-x))
    # 求导
    def gradient(self,x):
        return  x*(1-x)

class relu:
    # max(0,x)
    def __call__(self,x):
        return np.maximum(0,x)

    def gradient(self,x):
        return np.where(x>0,1,0)
    
class tanh:
    # (exp(x)-exp(-x))/(exp(x)+exp(-x))
    def __call__(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    # 求导
    def gradient(self,x):
        return 1-x**2

class softmax:
    # exp(x)/sum(exp(x))
    def __call__(self,x):
        e_x = np.exp(x - np.max(x))  # 数值稳定处理
        return e_x / np.sum(e_x)
    
    def gradient(self,x):
        s = self.__call__(x).reshape(-1,1)  # 转为列向量
        return np.diagflat(s) - np.dot(s, s.T)  # 雅可比矩阵计算

class Mse:
    # 
    def __call__(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def gradient(self,y_true,y_pred):
        # 修改为同时支持标量和数组输入
        if isinstance(y_true, (int, float)):
            return 2*(y_pred-y_true)  # 标量情况
        return 2*(y_pred-y_true)/y_true.size  # 数组情况


class MAE:
    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def gradient(self, y_true, y_pred):
        # 修改为同时支持标量和数组输入
        if isinstance(y_true, (int, float)):
            if y_pred > y_true:
                return 1
            elif y_pred < y_true:
                return -1
            else:
                return 0  

if __name__ == '__main__':
    relu = relu()
    sigmoid = sigmoid()
    mse = Mse()  # 创建MSE实例
    mae = MAE()
    x = random.random()
    w = random.random()
    b = random.random()
    true = random.random()
    print(f'x={x}   true={true}')

    lr = 0.3
    epochs = 520
    loss_hisory = []
    for epoch in range(epochs):
        pred = sigmoid(w * x + b)
        loss = mse(true, pred)  # 使用实例调用__call__方法
        #更新参数部分保持不变
        #反向传播 
        w -= lr * x * sigmoid.gradient(pred) * mse.gradient(true, pred)  # 注意方法名是gradient不是diff
        b -= lr * sigmoid.gradient(pred) * mse.gradient(true, pred)
        if epoch % 100 == 0:
            print(f'epoch {epoch}, loss={loss}, pred={pred}')
        loss_hisory.append(loss)

    print(f'epoch {epoch+1}, loss={loss}, pred={pred}')
    plt.plot(loss_hisory)
    plt.show()

