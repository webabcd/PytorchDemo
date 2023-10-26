import numpy as np


# 改变数组的形状（升维，降维）
def sample1():
    a = np.arange(8)
    print(a) # [0 1 2 3 4 5 6 7]

    # np.resize() 修改数组的形状
    print(np.resize(a, (4,2)))
    '''
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
    '''

    # reshape() 修改数组的形状，按行排序
    b = a.reshape(4,2, order='C')
    print(b)
    '''
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
    '''
    # reshape() 修改数组的形状，按列排序
    c = a.reshape(4,2, order='F')
    print(c)
    '''
[[0 4]
 [1 5]
 [2 6]
 [3 7]]
    '''

    # flat 可以将多维数组转换为元素的迭代器，其可以迭代出每一个元素
    d = [v for v in c.flat]
    print(d) # [0, 4, 1, 5, 2, 6, 3, 7]

    # flatten() 将多维数组降级为一维数组，按行排序
    e = c.flatten(order='C')
    print(e) # [0 4 1 5 2 6 3 7]

    # flatten() 将多维数组降级为一维数组，按列排序
    f = c.flatten(order='F')
    print(f) # [0 1 2 3 4 5 6 7]

    g = np.array([[1, 2], [3, 4]])
    print(g)
    '''
[[1 2]
 [3 4]]
    '''
    # np.ravel() 将多维数组降级为一维数组，新数组中的元素通过指针引用原数组中的元素
    h = np.ravel(g)
    print(h)
    '''
[1 2 3 4]
    '''
    # 因为 np.ravel() 做的是数据指针的复制，而不是数据的复制，所以新数组中的元素的变化会影响原数组，原数组中的元素的变化也会影响新数组
    h[0] = 100
    print(h)
    '''
[100   2   3   4]
    '''
    print(g)
    '''
[[100   2]
 [  3   4]]
    '''

    # 升维，在指定的位置上添加轴
    i = np.expand_dims(g, axis=2)
    print(i)
    '''
[[[100]
  [  2]]

 [[  3]
  [  4]]]
    '''

    # 降维，删除指定位置的轴
    j = np.squeeze(i, axis=2)
    print(j)
    '''
[[100   2]
 [  3   4]]
    '''


# 改变数组的形状（轴变换）
def sample2():
    a = np.arange(6).reshape(2, 3)
    print(a)
    '''
[[0 1 2]
 [3 4 5]]
    '''

    # 轴对换，即行变列，列变行
    b = np.transpose(a)
    print(b)
    '''
[[0 3]
 [1 4]
 [2 5]]
    '''

    # 交换 0 轴和 1 轴
    c = np.swapaxes(a, 0, 1)
    print(c)
    '''
[[0 3]
 [1 4]
 [2 5]]
    '''


if __name__ == '__main__':
    # 改变数组的形状（升维，降维）
    sample1()
    # 改变数组的形状（轴变换）
    sample2()
