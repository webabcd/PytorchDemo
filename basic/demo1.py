# 本例用于演示 torch 张量的基础操作，其操作与 numpy 基本相同（注：看本例之前先要把 numpy 弄懂）
# torch 可以支持在 gpu 上运算，而 numpy 只能在 cpu 上运算

import numpy as np
import torch


# python 列表，numpy 数组，torch 张量相互转换
def sample1(): 
    # python 列表   
    a = [0,1,2]
    print(a) # [0, 1, 2]

    # python 列表转 numpy 数组
    b = np.array(a) 
    print(b) # [0 1 2]

    # python 列表转 torch 张量
    c = torch.tensor(a) 
    print(c) # tensor([0, 1, 2])

    # numpy 数组转 torch 张量
    d = b.tolist() 
    print(d) # [0, 1, 2]

    # numpy 数组转 torch 张量
    e = torch.from_numpy(b)
    print(e) # tensor([0, 1, 2], dtype=torch.int32)

    # torch 张量转 python 列表
    f = c.tolist()
    print(f) # [0, 1, 2]

    # torch 张量转 numpy 数组
    g = c.numpy()
    print(g) # [0 1 2]


# torch 张量的创建
def sample2():
    # 创建 torch 张量
    a = torch.Tensor([1,2,3])
    print(a)
    '''
tensor([1., 2., 3.])
    '''

    # 创建 torch 张量，并指定元素的数据类型
    b = torch.tensor([1,2,3], dtype=torch.int32)
    print(b)
    '''
tensor([1, 2, 3], dtype=torch.int32)
    '''

    # 关于 zeros(), zeros_like(), ones(), ones_like(), arange(), linspace() 之类的函数用法与 numpy 一致（参见 /numpy/demo1.py 中的说明）
    c = torch.zeros((2, 3), dtype=torch.int32)
    print(c)
    '''
tensor([[0, 0, 0],
        [0, 0, 0]], dtype=torch.int32)
    '''

    # 创建一个 2 * 3 的随机数组，每个元素的值在 0 - 1 之间（左闭右开）
    d = torch.rand(2, 3)
    print(d)
    '''
tensor([[0.9596, 0.0913, 0.8888],
        [0.9932, 0.5563, 0.9348]])
    '''

    # 创建一个 2 * 3 的随机数组，生成的数据满足正态分布
    e = torch.randn(2, 3)
    print(e)
    '''
tensor([[-0.7344, -0.7011, -0.3015],
        [ 0.6692, -0.7376, -0.2654]])
    '''

    # 创建一个 2 * 5 的随机数组，生成的数据满足正态分布，且正态分布的均值为 0，标准差为 1
    f = torch.normal(0, 1, size=(2, 5))
    print(f)
    '''
tensor([[ 0.5377,  0.4246, -0.3362,  1.1923, -1.4358],
        [ 0.0945, -1.1919, -0.7694,  0.8314,  0.7839]])
    '''

    # 生成 0 - 10 之间（左闭右开）的数组，且顺序是随机的
    g = torch.randperm(10)
    print(g)
    '''
tensor([9, 0, 6, 7, 3, 2, 5, 4, 8, 1])
    '''

    x = np.array(bytes("abcdefg", "utf-8"))
    # 将二进制数据转换为 torch 张量
    x1 = torch.frombuffer(x, dtype=torch.uint8)
    print(x1) # tensor([ 97,  98,  99, 100, 101, 102, 103], dtype=torch.uint8)
    # 将二进制数据转换为 torch 张量，offset 为数据读取的起始位置，count 为读取的数据量
    x2 = torch.frombuffer(x, dtype=torch.uint8, offset=1, count=2)
    print(x2) # tensor([98, 99], dtype=torch.uint8)


# torch 张量的属性，切片，索引
# 基本和 numpy 差不多，参见 /numpy/demo2.py 中的说明
def sample3():
    a = torch.arange(24).reshape(2, 4, 3)
    print(a)
    '''
tensor([[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11]],

        [[12, 13, 14],
         [15, 16, 17],
         [18, 19, 20],
         [21, 22, 23]]])
    '''

    # ndim 张量的维度的个数，维度称之为 axis，维度的个数称之为 rank
    print(a.ndim) # 3

    # shape 张量的形状
    print(a.shape) # torch.Size([2, 4, 3])

    # dtype 张量元素的数据类型
    print(a.dtype) # torch.int64

    # 获取第 0 轴的位置 1 的数据
    print(a[1])
    '''
tensor([[12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23]])
    '''

    # 获取第 1 轴的位置 2 的数据
    print(a[...,2])
    '''
tensor([[ 2,  5,  8, 11],
        [14, 17, 20, 23]])
    '''


# torch 张量的运算，连接，分割
# 基本和 numpy 差不多，参见 /numpy/demo3.py 中的说明
def sample4():
    a = torch.tensor([1, 2, 3])
    print(a)
    '''
tensor([1, 2, 3])
    '''
    b = torch.tensor([[2], [4], [6]])
    print(b)
    '''
tensor([[2],
        [4],
        [6]])
    '''
    # 不同形状的张量在进行运算时，会通过广播机制（broadcasting）扩展为合适形状的张量
    print(a + b)
    '''
tensor([[3, 4, 5],
        [5, 6, 7],
        [7, 8, 9]])
    '''

    c = torch.tensor([[1,2],[3,4]])
    d = torch.tensor([[5,6],[7,8]])
    # 沿着轴 0 连接两个数组，轴（axis）就是指的维度
    # torch.cat(), torch.concat(), torch.concatenate() 是一样的
    e = torch.cat((c,d), axis=0)
    print(e)
    '''
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
    '''


# torch 张量的形状变化（升维，降维，轴变换）
def sample5():
    a = torch.arange(8)
    print(a)
    '''
tensor([0, 1, 2, 3, 4, 5, 6, 7])
    '''

    # 将一维张量的形状修改为 4 * 2
    b = a.reshape(4,2)
    print(b)
    '''
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])
    '''

    # 将二维张量降维成一维张量
    c = b.flatten()
    print(c)
    '''
tensor([0, 1, 2, 3, 4, 5, 6, 7])
    '''

     # torch.ravel() 将多维张量降级为一维张量，新张量中的元素通过指针引用原张量中的元素
    d = torch.ravel(b)
    print(d)
    '''
tensor([0, 1, 2, 3, 4, 5, 6, 7])
    '''
    # 因为 torch.ravel() 做的是数据指针的复制，而不是数据的复制，所以新张量中的元素的变化会影响原张量，原张量中的元素的变化也会影响新张量
    d[0] = 100
    print(d)
    '''
tensor([100,   1,   2,   3,   4,   5,   6,   7])
    '''
    print(b)
    '''
tensor([[100,   1],
        [  2,   3],
        [  4,   5],
        [  6,   7]])
    '''

    # 交换 0 轴和 1 轴
    e = torch.swapaxes(b, 0, 1)
    print(e)
    '''
tensor([[100,   2,   4,   6],
        [  1,   3,   5,   7]])
    '''

    # 升维，在指定的位置上添加轴
    f = torch.unsqueeze(e, axis=2)
    print(f)
    '''
tensor([[[100],
         [  2],
         [  4],
         [  6]],

        [[  1],
         [  3],
         [  5],
         [  7]]])
    '''

    # 降维，删除指定位置的轴
    g = torch.squeeze(f, axis=2)
    print(g)
    '''
tensor([[100,   2,   4,   6],
        [  1,   3,   5,   7]])
    '''


# torch 张量的常用函数
# torch.sin(), torch.cos(), torch.tan(), torch.floor(), torch.ceil(), torch.min(), torch.max() 之类的用法和 numpy 差不多，参见 /numpy/demo5.py 中的说明
def sample6():
    a = torch.tensor([[1,2,3],[2,3,4]])
    print(a)
    '''
tensor([[1, 2, 3],
        [2, 3, 4]])
    '''

    # 张量 a 中，如果元素的值 > 0.5，则元素值变为原值 + 100，否则元素值变为原值 + 10
    b = torch.where(a > 2, a + 100, a + 10)
    print(b)
    '''
tensor([[ 11,  12, 103],
        [ 12, 103, 104]])
    '''

    # 去重并降至一维
    c = torch.unique(a)
    print(c)
    '''
tensor([1, 2, 3, 4])
    '''


if __name__ == '__main__':
    # python 列表，numpy 数组，torch 张量相互转换
    sample1()
    # torch 张量的创建
    sample2()
    # torch 张量的属性，切片，索引
    sample3()
    # torch 张量的运算，连接，分割
    sample4()
    # torch 张量的形状变化（升维，降维，轴变换）
    sample5()
    # torch 张量的常用函数
    sample6()
