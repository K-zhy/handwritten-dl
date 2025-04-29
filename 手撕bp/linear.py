# 损失函数
class Mse:
    def __call__(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    def gradient(self,y_true,y_pred):
        return 2*(y_pred-y_true)/y_true.size

# 激活函数
class sigmoid:
    # 1/(1+exp(-x)) 输出范围为0-1，适用于二分类问题
    def __call__(self,x):
        return 1/(1+np.exp(-x))
    # 求导
    def gradient(self,x):
        return  x*(1-x)

class linear:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size,output_size) * 0.1 # 初始化权重矩阵，大小为input_size*output_size，值在-0.01到0.01之间

        self.bias = np.random.randn(1,output_size) * 0.1 # 初始化偏置向量，大小为1*output_size，值在-0.01到0.01之间

        self.activation = activation # 激活函数

    def forward(self, x):
        self.input = x # 保存输入，用于反向传播
        self.output = np.dot(x, self.weights) + self.bias # 计算加权和并加上偏置，得到输出
        self.a_output = self.activation(self.output) # 应用激活函数并返回输出
        return self.a_output

    # 更新weights和bias
    def update(self, loss_gradient):
        activation_gradient = self.activation.gradient(self.a_output) * loss_gradient # 计算激活函数的导数
        # new_gradient =  np.dot(self.input.T, activation_gradient) # 计算权重的梯度

        self.weights -= lr * np.dot(self.input.T, activation_gradient) # 更新权重矩阵
        self.bias -= lr * activation_gradient * 1 # 更新偏置向量


class Network:
    def __init__(self, layers):
        self.layers = [] # 保存所有层的列表

        self.layers1 = linear(4, 16, sigmoid())
        self.layers2 = linear(16, 16, sigmoid())
        self.layers3 = linear(16, 1, sigmoid())
    
    # 模型计算
    def __call__(self, x) :
        x = self.layers1.forward(x)
        x = self.layers2.forward(x)
        x = self.layers3.forward(x)

        return x
    
    def backward(self, true_y, pred_y):
        loss_gradient = Mse().gradient(true_y, pred_y) # 计算损失函数的梯度
        return loss_gradient
        
