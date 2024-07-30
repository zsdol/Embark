from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 初学只要知道大致的数据处理流程即可
if __name__ == '__main__':
    # 实现图像的预处理pipeline
    transform = transforms.Compose([
        # 转换成单通道灰度图
        transforms.Grayscale(num_output_channels=1),
        # 转换为张量
        transforms.ToTensor()
    ])
    # 构建数据集dataset
    train_dataset = datasets.MNIST(root='./data/mnist/train', train=True,
                                   transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/mnist/test', train=False,
                                  transform=transform, download=True)

    # 打印他们的长度
    print("train_dataset length: ", len(train_dataset))
    print("test_dataset length: ", len(test_dataset))

    # 使用train_loader, 实现小批量的数据读取
    # 这里设置小批量的大小，batch_size=64. 也就是每个批次，包括64个数据
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 打印train_loader的长度
    print("train_loader length: ", len(train_loader))
    # 6000个训练数据，如果每个小批量，读入64个样本，那么60000个数据会被分成938组
    # 938*64=60032，说明最后一组不够64个数据

    # 循环遍历train_loader
    # 每一次循环，都会取出64个图像数据，作为一个小批量batch
    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx == 3:
            break
        print("batch_idx: ", batch_idx)
        print("data.shape: ", data.shape) # 数据的尺寸
        print("label: ", label.shape) # 图像中的数字
        print(label)