'''
K-means 聚类算法（sklearn.cluster 的 KMeans 实现，对一个包含 10 个特征的数据做分类）

K-means 聚类算法是一种把数据分成 k 个组的聚类算法
它先随机选出 k 个数据点作为初始的簇中心，然后计算每个数据点到每个簇中心的距离，把每个数据点分配给距离它最近的那个簇中心，然后根据已有的数据点重新计算簇中心
这个过程会重复进行，直到满足某个条件，例如没有数据点需要重新分配或没有簇中心再变化，或者误差最小


.pt 文件通常用于存储 PyTorch 模型的状态字典（state_dict）、模型结构、模型权重等相关信息。训练一个神经网络模型后，如果想要保存其状态、模型的结构和参数等，则可以保存文 .pt 文件
'''

import torch
from sklearn.cluster import KMeans  # 用于实现 K-means 聚类
import numpy as np

def sample1():
    # 指定聚类的数量（即簇的数量）
    n_clusters = 5
    # 定义容差，用于判断算法是否收敛到最优解
    tolerance = 1e-5

    # 测试数据（100 个数据点，每个数据点有 10 个特征）
    data_tensor = torch.randn(100, 10) 
    data_np = data_tensor.numpy()

    # 实例化 KMeans 对象
    kmeans = KMeans(n_clusters=n_clusters, tol=tolerance)
    # 对指定的数据做聚类分析
    kmeans.fit(data_np)

    # 获取簇中心点的位置
    centroids = torch.tensor(kmeans.cluster_centers_) # shape: 5,10
    # 获取测试数据中，每个数据点所属的簇索引
    cluster_assignment = torch.tensor(kmeans.labels_) # shape: 100

    print("centroids: ", centroids)
    print("cluster_assignment: ", cluster_assignment)

    # 保存簇中心到文件
    torch.save(centroids, 'checkpoints/my_kmeans_centroids_demo07.pt')

    # 用已有的簇中心结果对指定的数据做分类
    test()

def test():
    # 从文件加载簇中心
    centroids = torch.load('checkpoints/my_kmeans_centroids_demo07.pt') # shape: 5,10

    # 定义一个需要分类的测试数据
    test_data = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # 计算数据点与每个簇中心点之间的距离
    distances = np.linalg.norm(centroids - test_data, axis=1) # shape: 5
    # 找到张量中最小值所在的索引位置，在本例中就是找到测试数据点所属簇的索引位置（即分类的结果）
    cluster_assignment = np.argmin(distances)

    # 打印测试数据和对其分类的结果
    print(f"test_data:{test_data}, cluster_assignment:{cluster_assignment}")


if __name__ == '__main__':
    sample1()