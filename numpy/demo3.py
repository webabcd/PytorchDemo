import numpy as np


# 数组之间的运算
def sample1():
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4, 6, 8])
    # 加法
    print(a + b) # [ 3  6  9 12]
    # 减法
    print(a - b) # [-1 -2 -3 -4]
    # 乘法
    print(a * b) # [ 2  8 18 32]
    # 除法
    print(a / b) # [0.5 0.5 0.5 0.5]
    # 幂运算
    print(a ** b) # [1 16 729 65536]


# 数组之间的连接
def sample2():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print(a)
    '''
[[1 2]
 [3 4]]
    '''
    print(b)
    '''
[[5 6]
 [7 8]]
    '''

    # 沿着轴 1 连接两个数组
    # 轴（axis）就是指的维度
    c = np.concatenate((a,b), axis=0)
    print(c)
    '''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
    '''
    # 沿着轴 1 连接两个数组
    d = np.concatenate((a,b), axis=1)
    print(d)
    '''
[[1 2 5 6]
 [3 4 7 8]]
    '''

    # 沿着轴 0 组合两个数组，会封装成为一个三维数组
    e = np.stack((a,b), axis=0)
    print(e)
    '''
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
    '''
    # 沿着轴 1 组合两个数组，会封装成为一个三维数组
    f = np.stack((a,b), axis=1)
    print(f)
    '''
[[[1 2]
  [5 6]]

 [[3 4]
  [7 8]]]
    '''

    # 垂直方向连接两个数组
    g = np.vstack((a,b))
    print(g)
    '''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
    '''
    # 水平方向连接两个数组
    h = np.hstack((a,b))
    print(h)
    '''
[[1 2 5 6]
 [3 4 7 8]]
    '''


# 数组之间的分割
def sample3():
    a = np.arange(9)
    print (a) # [0 1 2 3 4 5 6 7 8]
    # 将数组分割为 3 个大小相等的数组
    b = np.split(a,3)
    print(b) # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
    # 从位置 4, 6, 7 分割数组
    c = np.split(a,[4,6,7])
    print(c) # [array([0, 1, 2, 3]), array([4, 5]), array([6]), array([7, 8])]

    d = np.array([[1,2,3,4],[5,6,7,8]])
    print(d)
    '''
[[1 2 3 4]
 [5 6 7 8]]
    '''
    # 垂直分割为 2 个数组
    e = np.vsplit(d, 2)
    print(e)
    '''
[array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]])]
    '''
    # 水平分割为 2 个数组
    f = np.hsplit(d, 2)
    print(f)
    '''
[array([[1, 2],
       [5, 6]]), array([[3, 4],
       [7, 8]])]
    '''


if __name__ == '__main__':
    # 数组之间的运算
    sample1()
    # 数组之间的连接
    sample2()
    # 数组之间的分割
    sample3()
