'''
通过卷积神经网络（Convolutional Neural Networks, CNN）做图片分类
本例用于演示如何通过 ResNet50 做图片分类的学习（对 cifar10 数据集做训练和测试），保存训练后的模型，加载训练后的模型并评估指定的图片


损失率（Loss）用于描述模型预测与实际情况之间的差异
  比如说有两种分类 0 和 1，实际分类为 1，预测为 0.8 的可能性为分类 1，那么损失率为 20%
准确率（Accuracy）用于描述模型在整个数据集上正确预测的样本比例
  比如说有两种分类 0 和 1，实际分类为 1，预测为 0.8 的可能性为分类 1，所以预测的结果是分类 1，那么准确率为 100%

学习率（learning rate）是比较重要的一个超参数（超参数是在模型训练之前设置的参数，非超参数是在模型训练过程中通过学习而自动得到的参数）
学习率决定了模型参数在每一轮迭代中的更新步长，其选择对模型的训练和性能有着重要的影响


.pth 文件通常是 PyTorch 用来保存模型权重（parameters）的文件格式，训练一个神经网络模型后，可以将训练得到的权重保存为 .pth 文件
.pt 文件通常用于存储 PyTorch 模型的状态字典（state_dict）、模型结构、模型权重等相关信息。训练一个神经网络模型后，如果想要保存其状态、模型的结构和参数等，则可以保存文 .pt 文件
checkpoints 文件夹通常用于保存训练过程中的模型检查点。训练模型需要经过多个 epoch 的训练，定期保存模型的参数（权重）、优化器的状态以及其他相关信息是很重要的
'''

import os 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision
# PIL（Python Imaging Library）
from PIL import Image


# 从 cifar10 数据集获取训练数据和测试数据
def sample1():
    # 定义数据变换
    transform = transforms.Compose([
        # 将图片数据转换为张量，并将原来的 height, widht, channel(rgb) 改为 channel(rgb), height, widht 的结构
        # 将数据通过除以 255 的方式归一化，变为 0 到 1 之间的数据
        transforms.ToTensor(), 
        # 对数据做标椎化处理，即将数据变为“均值为0，标准差为1”的数据（注：这个是由原始数据转换过来的，其不一定是正态分布，是否是正态分布是由原始数据决定的）
        # 注：在深度学习的图像处理中，经过标准化处理之后的数据可以更好的响应激活函数，减少梯度爆炸和梯度消失的出现
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # 训练集（cifar10 训练集提供 10 个类别的共 50000 张 32*32 的 rgb 通道的图片）
    # root 代表数据集的路径，train=True 代表训练集，train=False 代表测试集，download 代表是否需要从网络下载数据集
    train_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=True, download=False, transform=transform)
    # 测试集（cifar10 测试集提供 10 个类别的共 10000 张 32*32 的 rgb 通道的图片）
    test_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=False, download=False, transform=transform)

    # 获取数据集的数据数量
    print(f'train_dataset:{len(train_dataset)}') # train_dataset:50000
    print(f'test_dataset:{len(test_dataset)}') # test_dataset:10000

    # cifar10 数据集的 10 个类别
    image_category_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 获取数据集的第 1 条数据
    # 返回的 tuple 的第 1 个数据是图片的数据，其按照 channel(rgb), height, widht 的结构保存
    # 返回的 tuple 的第 2 个数据是图片的类别的索引
    image_data, image_category_index = train_dataset.__getitem__(0)
    print(image_data.size()) # torch.Size([3, 32, 32])
    print(image_category_list[image_category_index]) # frog
 
    # 定义画布
    figure = plt.figure()
    for i in range(8):
        image_data, image_category_index = train_dataset.__getitem__(i)
        # 将 channel(rgb), height, widht 变为 height, widht, channel(rgb)
        image = image_data.numpy().transpose(1, 2, 0)
        # 将画布分为 2 行 4 列，当前在第 i + 1 单元格绘制（注：这个值从 1 开始，而不是从 0 开始）
        ax = figure.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        # 设置当前单元格的标题
        plt.xlabel(image_category_list[image_category_index])
        # 调整单元格的布局
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # 在当前单元格绘制图片
        # 这么绘制图片会有失真，并伴随一个警告 Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
        # 因为之前通过 transforms.Normalize() 对数据做标椎化处理了，导致数据变为“均值为0，标准差为1”的数据了
        plt.imshow(image)
    # 显示图片并阻塞，直到关闭图片
    plt.show()

    # 如果想要不失真的绘制数据集的图片，则只要不对数据做标准化处理就行了
    train_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=True, download=False)
    figure = plt.figure()
    for i in range(8):
        image_data, image_category_index = train_dataset.__getitem__(i)
        ax = figure.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        plt.xlabel(image_category_list[image_category_index])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.imshow(image_data)
    plt.show()
    

# 通过 ResNet50 做图片分类的学习（对 cifar10 数据集做训练和测试），并保存训练后的模型
def sample2():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=False, download=False, transform=transform)

    # 每个批次的样本数（在深度学习中，数据通常被分割成小的批次进行训练）
    batch_size = 32
    # 用于数据加载的进程数
    num_workers = 0
    # 通过 DataLoader 读取数据集中的数据
    # shuffle - 是否需要打乱样本顺序（可以尽量避免数据的顺序对模型的训练的影响）
    # drop_last - 是否舍弃最后的不足批次大小的数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # ResNet50（Residual Network-50）是深度学习领域中一种经典的卷积神经网络结构，其适用于训练大规模的图像数据集，通常用于图像分类、目标检测和图像分割等计算机视觉任务
    # 从网络上下载 ResNet50 模型（包括模型结构和模型权重，模型权重保存在模型的状态字典中）并缓存到本地
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    # 实例化 ResNet50 模型结构（不加载模型权重）
    model = torchvision.models.resnet50(weights=None)
    # 加载模型的状态字典，即模型的权重
    state_dict = torch.load('checkpoints/original_resnet50.pth')
    model.load_state_dict(state_dict)
    # 需要分类的类别的数量
    model.fc.out_features = 10

    # cpu 加速就是 'cpu'
    # 第 1 块 gpu 加速就是 'cuda:0'，第 2 块 gpu 加速就是 'cuda:1'，以此类推...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数
    # torch.nn.CrossEntropyLoss 通常用于多类别分类任务，它是一个用于计算交叉熵损失的损失函数，它将 Softmax 激活函数和负对数似然损失结合在一起
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器的学习率（learning rate）
    lr = 1e-4
    # 定义优化器
    # torch.optim.Adam 是 Adam 优化算法的实现
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 整个数据集的训练轮次
    epoch = 10
    
    for i in range(epoch):
        # 将模型设置为训练模式
        model.train()
        # 总的训练数
        train_total_num = 0
        # 总的训练损失率
        train_total_loss = 0
        # 总的训练正确数
        train_total_correct = 0

        # 每次迭代出的数据量就是 batch_size 指定的大小
        for iter, (images, labels) in enumerate(train_loader):
            # 当前迭代出的批次的图片数据的集合
            images = images.to(device)
            # 当前迭代出的批次的分类数据的集合
            labels = labels.to(device)
            
            # 通过模型预测数据
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_total_num += labels.shape[0]
            train_total_correct += (outputs.argmax(1) == labels).sum().item()
            train_total_loss += loss.item()
            
            # 反向传播计算梯度
            loss.backward()
            # 优化模型参数（根据学习率和梯度做参数更新）
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
            
            # 打印当前批次的训练结果
            print("epoch({}/{}), iter({}/{}), train_loss:{:4f}".format(
                i+1, epoch, 
                iter+1, len(train_loader), 
                loss.item()/labels.shape[0]))

        
        # 将模型设置为测试模式，即评估模式
        model.eval()
        # 总的测试数
        test_total_num = 0
        # 总的测试损失率
        test_total_loss = 0
        # 总的测试正确数
        test_total_correct = 0
        
        for iter,(images,labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 通过模型预测数据
            outputs = model(images)
            loss = criterion(outputs,labels)
            test_total_num += labels.shape[0]
            test_total_loss += loss.item()
            test_total_correct += (outputs.argmax(1) == labels).sum().item()
            
        # 打印测试结果
        print("epoch({}/{}), train_loss:{:.4f}, train_acc:{:.4f}%, test_loss:{:.4f}, test_acc:{:.4f}%".format(
            i+1, epoch, 
            train_total_loss / train_total_num, 
            train_total_correct / train_total_num * 100, 
            test_total_loss / test_total_num, 
            test_total_correct / test_total_num * 100))


    # 保存整个模型（包括模型结构和模型权重）。之后可以重新加载并继续训练或评估指定的图片
    # torch.save(model, 'checkpoints/my_whole_resnet50.pth')
    # 只保存模型的状态字典，即模型的权重。之后可以重新加载并继续训练或评估指定的图片
    torch.save(model.state_dict(), 'checkpoints/my_resnet50.pth')


# 加载训练后的模型，并评估指定的图片
def sample3():
    # 加载整个模型（包括模型结构和模型权重）。之后可以继续训练或评估指定的图片
    # model = torch.load('checkpoints/my_whole_resnet50.pth')
    # 实例化 ResNet50 模型结构（不加载模型权重）
    model = torchvision.models.resnet50(weights=None)
    # 加载模型的状态字典，即模型的权重。之后可以继续训练或评估指定的图片
    state_dict = torch.load('checkpoints/my_resnet50.pth')
    model.load_state_dict(state_dict)

    # 开启 gpu 加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 将模型设置为评估模式
    model.eval()

    # 获取需要评估的图片数据
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open('assets/ship.jpg').convert('RGB')
    input_data = transform(image)
    # 添加批次维度
    input_data = input_data.unsqueeze(0)
    input_data = input_data.to(device)

    image_category_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 获取评估结果
    with torch.no_grad():
        output_data = model(input_data)
    # 获取每个类别的概率
    probs = torch.nn.functional.softmax(output_data[0], dim=0)
    # 获取前 3 条数据（topk 就是 top -k 的意思，即获取前 k 条数据）
    top3_probability, top3_category = torch.topk(probs, 3)
    for i in range(top3_probability.size(0)):
        print(f'category:{image_category_list[top3_category[i].item()]}, probability:{top3_probability[i].item()}')
    '''
category:ship, probability:0.9999663829803467
category:plane, probability:2.452900844218675e-05
category:horse, probability:2.469446826580679e-06
    '''


if __name__ == '__main__':
    # 从 cifar10 数据集获取图片分类的训练数据和测试数据
    sample1()
    # 通过 ResNet50 做图片分类的学习（对 cifar10 数据集做训练和测试），并保存训练后的模型
    sample2()
    # 加载训练后的模型，并评估指定的图片
    sample3()

