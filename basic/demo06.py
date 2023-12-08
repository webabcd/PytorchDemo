'''
K-means 聚类算法（自定义实现，对一个 x,y 数据做分类）
本例中可以把 x,y 数据理解为二维坐标上的一个点

K-means 聚类算法是一种把数据分成 k 个组的聚类算法
它先随机选出 k 个数据点作为初始的簇中心，然后计算每个数据点到每个簇中心的距离，把每个数据点分配给距离它最近的那个簇中心，然后根据已有的数据点重新计算簇中心
这个过程会重复进行，直到满足某个条件，例如没有数据点需要重新分配或没有簇中心再变化，或者误差最小


.pt 文件通常用于存储 PyTorch 模型的状态字典（state_dict）、模型结构、模型权重等相关信息。训练一个神经网络模型后，如果想要保存其状态、模型的结构和参数等，则可以保存文 .pt 文件
'''

import torch
import matplotlib.pyplot as plt
import random
import numpy as np

def sample1():
    # 生成随机二维数据
    # 100 个数据点，每个数据点有 2 个特征（也就是 x,y 数据）
    # 本例中可以把 x,y 数据理解为二维坐标上的一个点
    data = torch.randn(100, 2) # 100,2

    # 指定聚类的数量（即簇的数量）
    k = 5
    # 初始化 k 个中心点（随机选取前 k 个数据点作为初始中心点）
    centroids = data[:k, :].clone()
    print("初始中心点：", centroids) # shape: 5,2

    # 定义最大迭代次数
    max_iters = 100
    # 定义容差，用于判断算法是否收敛到最优解
    tolerance = 1e-5

    for iter in range(max_iters):
        # 计算数据点到中心点的距离
        # torch.cdist() - 用于计算两组张量中每对向量之间的距离，并返回包含这些距离的张量
        # 本例中，得到的就是 data 中的 100 个点分别与 centroids 中的 5 个点的分别的距离
        distances = torch.cdist(data, centroids) # shape: 100,5

        # 分配数据点到最近的中心点
        # torch.argmin() - 用于找到张量中沿指定维度的最小值所在的索引位置
        # 本例中，得到的就是 data 中的 100 个点的所属的中心点的索引位置
        cluster_assignment = torch.argmin(distances, dim=1) # shape: 100

        # 用于保存更新后的中心点位置
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            # 获取属于第 i 个簇的所有数据点
            cluster_i = data[cluster_assignment == i]
            if len(cluster_i) > 0:
                # 计算当前簇的数据点的均值，并作为新的中心点
                new_centroids[i] = cluster_i.mean(dim=0)
            else:
                # 若当前簇没有数据点，则保持原中心点不变
                new_centroids[i] = centroids[i]

        # 判断算法是否收敛到最优解
        if torch.all(torch.abs(new_centroids - centroids) < tolerance):
            print(f"算法收敛于第 {iter + 1} 次迭代")
            break

        # 更新中心点位置
        centroids = new_centroids.clone()

    # 保存簇中心到文件
    torch.save(centroids, 'checkpoints/my_kmeans_centroids_demo06.pt')

    # 用已有的簇中心结果对指定的数据做分类
    test()

    # 可视化聚类结果
    for i in range(k):
        cluster_i = data[cluster_assignment == i]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=generate_random_color(), label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', s=100, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()

def generate_random_color():  
    red, green, blue = (random.randint(0, 255) for _ in range(3))  
    color = "#{:02x}{:02x}{:02x}".format(red, green, blue)  
    return color


def test():
    # 从文件加载簇中心
    centroids = torch.load('checkpoints/my_kmeans_centroids_demo06.pt') # shape: 5,2

    # 定义一个需要分类的测试数据（一个 x,y 数据）
    test_data = torch.tensor([0, 0]) # shape: 2

    # 计算数据点与每个簇中心点之间的距离
    distances = np.linalg.norm(centroids - test_data, axis=1) # shape: 5
    # 找到张量中最小值所在的索引位置，在本例中就是找到测试数据点所属簇的索引位置（即分类的结果）
    cluster_assignment = np.argmin(distances)

    # 打印测试数据和对其分类的结果
    print(f"test_data:{test_data}, cluster_assignment:{cluster_assignment}")


if __name__ == '__main__':
    sample1()