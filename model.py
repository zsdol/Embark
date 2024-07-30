import torch
from torch import nn

# 定义神经网络Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性层1，输入层和隐藏层之间的线性层
        self.layer1 = nn.Linear(784, 256)
        # 线性层2，隐藏层和输出层之间的线性层
        self.layer2 = nn.Linear(256, 10)
    # 在前向传播，forward函数中，输入为图像x
    def forward(self, x):
        x = x.view(-1, 28 * 28) # 使用view函数，将x展平
        x = self.layer1(x) # 将x输入到layer1
        x = torch.relu(x) # 使用relu激活
        return self.layer2(x) # 输入至layer2计算结果

    # 这里没有直接定义softmax层，因为后面会使用CrossEntropyLoss损失函数
    # 在这个损失函数中，会实现softmax的计算