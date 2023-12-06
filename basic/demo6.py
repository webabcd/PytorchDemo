'''
K-means 聚类算法

K-means 聚类算法是一种把数据分成 k 个组的聚类算法
它先随机选出 k 个数据点作为初始的组中心，然后计算每个数据点到每个组中心的距离，把每个数据点分配给距离它最近的那个组中心，然后根据已有的数据点重新计算组中心
这个过程会重复进行，直到满足某个条件，例如没有数据点需要重新分配或没有组中心再变化，或者误差最小
'''

import torch
import matplotlib.pyplot as plt
import random
import numpy as np

def sample1():
    # 生成随机二维数据（100 个数据点，每个数据点有 2 个特征）
    data = torch.randn(100, 2) 

    # 指定聚类的数量
    k = 5
    # 初始化 k 个中心点（随机选取前 k 个数据点作为初始中心点）
    centroids = data[:k, :].clone()
    print("初始中心点：", centroids)

    # 定义迭代次数和阈值
    max_iters = 100
    threshold = 1e-5

    for iter in range(max_iters):
        # 计算数据点到中心点的欧氏距离
        distances = torch.cdist(data, centroids)
        # 分配数据点到最近的中心点
        cluster_assignment = torch.argmin(distances, dim=1)

        # 更新中心点位置
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            # 属于第 i 个簇的数据点
            cluster_i = data[cluster_assignment == i]
            if len(cluster_i) > 0:
                # 计算每个簇的均值作为新的中心点
                new_centroids[i] = cluster_i.mean(dim=0)
            else:
                # 若某个簇没有数据点，则保持原中心点不变
                new_centroids[i] = centroids[i]

        # 判断是否收敛
        if torch.all(torch.abs(new_centroids - centroids) < threshold):
            print(f"算法收敛于第 {iter + 1} 次迭代")
            break

        # 更新中心点位置
        centroids = new_centroids.clone()

    
    # 可视化聚类结果
    for i in range(k):
        cluster_i = data[cluster_assignment == i]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=generate_random_color(), label=f'Cluster {i + 1}')
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


if __name__ == '__main__':
    sample1()