import numpy as np


# 数组的添加和删除
def sample1():
    a = np.array([[1,2,3],[4,5,6]])
    print(a)
    '''
[[1 2 3]
 [4 5 6]]
    '''

    # 在指定的轴上，追加指定的数据
    b = np.append(a, [[7,8,9]], axis = 0)
    print(b)
    '''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
    '''

    # 在指定的轴的指定的位置上，添加指定的数据
    c = np.insert(b, 1, [[7,8,9]], axis = 0)
    print(c)
    '''
[[1 2 3]
 [7 8 9]
 [4 5 6]
 [7 8 9]]]
    '''

    # 在指定的轴的指定的位置上，删除数据
    d = np.delete(c, 2, axis=0)
    print(d)
    '''
[[1 2 3]
 [7 8 9]
 [7 8 9]]
    '''


# 数组的常用函数
def sample2():
    a = np.array([0,20,40,80,90])
    print(a)

    # 三角函数 np.sin(), np.cos(), np.tan() 等
    b = np.sin(a*np.pi/180)
    print(b) # [0. 0.34202014 0.64278761 0.98480775 1. ]

    # 取小于等于的整数
    c = np.floor(b)
    print(c) # [0. 0. 0. 0. 1.]

    # 取大于等于的整数
    d = np.ceil(b)
    print(d) # [0. 1. 1. 1. 1.]

    # 取四舍五入后的的整数（around 和 round 是一样的）
    e = np.around(b)
    print(e) # [0. 0. 1. 1. 1.]
    # 指定保留的小数位数，并四舍五入（around 和 round 是一样的）
    e = np.around(b, 3)
    print(e) # [0. 0.342 0.643 0.985 1.]
    # 对于正好 .5 来说，其会被舍入到最接近的偶数
    print(np.around(0.5), np.around(1.5)) # 0.0 2.0

    # 取最小数
    f = np.min(b)
    print(f) # 0.0

    # 取最大数
    g = np.max(b)
    print(g) # 1.0

    # 取平均数
    h = np.average(b)
    print(h) # 0.5939231012048831

    # 数组 b 中，如果元素的值 > 0.5，则元素值变为原值 + 100，否则元素值变为原值 + 10
    i = np.where(b > 0.5, b + 100, b + 10)
    print(i) # [10. 10.34202014 100.64278761 100.98480775 101. ]

    x = np.array([[1,2],[2,3],[3,4]])
    print(x)
    '''
[[1 2]
 [2 3]
 [3 4]]
    '''
    # 去重并降至一维，结果对应 x1
    # return_index=True 返回的数据对应 x2，其代表结果数据相对于原始数据的索引位置
    # return_inverse=True 返回的数据对应 x3，其代表原始数据相对于结果数据的索引位置
    # return_counts=True 返回的数据对应 x4，其代表结果数据相对于原始数据的重复次数
    x1, x2, x3, x4 = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
    print(x1, x2, x3, x4)
    '''
[1 2 3 4] [0 1 3 5] [0 1 1 2 2 3] [1 2 2 1]
    '''


if __name__ == '__main__':
    # 数组的添加和删除
    sample1()
    # 数组的常用函数
    sample2()

