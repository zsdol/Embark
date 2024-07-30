import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 图像的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 读入并构造数据集
    train_dataset = datasets.MNIST('./data/train',train=True,transform=transform,download=True)
    # train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    print("train_dataset length: ", len(train_dataset))

    # 小批量的数据读入
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("train_loader length: ", len(train_loader))

    # 在使用Pytorch训练模型时，需要创建三个对象：
    model = Network() # 1.模型本身，就是我们设计的神经网络
    optimizer = optim.Adam(model.parameters()) #2.优化器，优化模型中的参数
    criterion = nn.CrossEntropyLoss() #3.损失函数，分类问题，使用交叉熵损失误差

    # 进入模型的循环迭代
    # 外层循环，代表了整个训练数据集的遍历次数
    for epoch in range(10):
        # 内层循环使用train_loader, 进行小批量的数据读取
        for batch_idx, (data, label) in enumerate(train_loader):
            # 内层每循环一次，就会进行一次梯度下降算法
            # 包括了5个步骤(使用pytorch框架训练模型的定式)
            output = model(data) # 1. 计算神经网络的前向传播结果
            loss = criterion(output, label) # 2. 计算output和标签label之间的损失loss
            loss.backward() # 3. 使用backward计算梯度
            optimizer.step() # 4. 使用optimizer.step更新参数
            optimizer.zero_grad() # 5.将梯度清零

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10"
                      f"| Batch {batch_idx}/{len(train_loader)}"
                      f"| Loss: {loss.item():.4f}"
                      )
    torch.save(model.state_dict(), 'mnist.pth')