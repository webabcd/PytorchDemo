import numpy as np


# 数组的常用属性
def sample1():
    # 数组 a 是一个一维数组
    a = np.arange(24)
    # 数组 b 是一个三维数组
    b = a.reshape(2,4,3) 

    # ndim 数组的维度的个数，维度称之为 axis，维度的个数称之为 rank
    print(a.ndim) # 1
    print(b.ndim) # 3

    # shape 数组的形状，返回一个整数元组，用于描述每个维度的大小
    print(a.shape) # (24,)
    print(b.shape) # (2, 4, 3)

    # size 数组元素的总数
    print(a.size) # 24
    print(b.size) # 24

    # dtype 数组元素的数据类型
    print(a.dtype) # int32
    print(b.dtype) # int32

    # itemsize 数组中每个元素占用的字节大小
    print(a.itemsize) # 4
    print(b.itemsize) # 4


# 数组的切片和索引（切片就是取多个数据，索引就是取一个数据）
def sample2():
    a = np.arange(10)
    print(a) # [0 1 2 3 4 5 6 7 8 9]
    # 取索引位置的 1 - 7 之间的元素（左闭右开），且步长为 2
    print(a[slice(1, 7, 2)]) # [1 3 5]
    print(a[1:7:2]) # [1 3 5]
    # 反向排列
    print(a[-1::-1]) # [9 8 7 6 5 4 3 2 1 0]
    # 取第 0 个元素
    print(a[0]) # 0
    # 取倒数第 1 个元素
    print(a[-1]) # 9
    # 取第 1 个元素到最后一个元素
    print(a[1:]) # [1 2 3 4 5 6 7 8 9]
    # 取第 0 个元素到倒数第 2 个元素（左闭右开）
    print(a[:-2]) # [0 1 2 3 4 5 6 7]
    # 取第 0 个元素到第 2 个元素（左闭右开）
    print(a[0:2]) # [0 1]

    b = np.array([[1,2,3],[3,4,5],[4,5,6]])
    print(b)
    '''
[[1 2 3]
 [3 4 5]
 [4 5 6]]
    '''
    # 第 1 列元素
    print (b[...,1]) # [2 4 5]
    # 第 1 行元素
    print (b[1,...]) # [3 4 5]
    # 第 1 列以及之后所有列的数据
    print (b[...,1:]) 
    '''
[[2 3]
 [4 5]
 [5 6]]
    '''

if __name__ == '__main__':
    # 数组的常用属性
    sample1()
    # 数组的切片和索引（切片就是取多个数据，索引就是取一个数据）
    sample2()
