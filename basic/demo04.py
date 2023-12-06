'''
通过长短时记忆网络（Long Short-Term Memory, LSTM）做时间序列预测

注：LSTM 是循环神经网络（Recurrent Neural Network, RNN）的一种
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error

# 是否仅用于预测未来
# True - 训练整个数据，并预测未来数据
# False - 整个数据被分割为训练数据和测试数据，并检测训练效果和测试效果
only_predict_future = True
# 测试数据占全部数据的百分比
test_data_percent = 0.1
# 需要回溯的数据的条数（为了预测下一个时间点的数据，所需的之前的数据的条数）
lookback = 100
# 需要预测未来的天数
days_future = 20
# 整个数据集的训练轮次
epoch = 50
# 优化器的学习率（learning rate）
lr = 1e-2

def sample1():
    # 生成一个包含 x, y 的 DataFrame 数据，模拟一个正弦曲线
    x = np.linspace(0, 64 * np.pi, 5000)  
    y = np.sin(x)
    df = pd.DataFrame({  
        'x': x,  
        'y': y  
    })  

    # 绘制 x, y 曲线图
    if not only_predict_future:
        plot_xy("x,y", df)


    # 输入数据的维度
    input_dim = 1
    # 隐藏层的维度
    hidden_dim = 32
    # 输出数据的维度
    output_dim = 1
    # LSTM 模型的层数
    num_layers = 2
    # 实例化自定义的 LSTM 模型
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    # 定义损失函数
    # torch.nn.MSELoss 用于计算均方误差，通常用于回归问题
    # reduction='mean' 意味着将对所有样本的损失值求平均，得到最终的损失值
    # MSE - 均方误差（Mean Squared Error）是将预测值与实际值之间的差值平方后求平均
    # RMSE - 均方根误差（Root Mean Squared Error）就是将 MSE 开方
    criterion = torch.nn.MSELoss(reduction='mean')
    # 定义优化器
    # torch.optim.Adam 是 Adam 优化算法的实现
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # df['y'].values 用于获取 y 列的全部数据
    # reshape(-1, 1) 用于把一维数组变为二维数组，因为机器学习中某些库要求输入的数据必须是二维的
    y_data = df['y'].values.reshape(-1, 1)
    # 数据归一（-1 到 1 之间）
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_data = scaler.fit_transform(y_data)
    # 获取训练数据和测试数据
    x_train, y_train, x_test, y_test = get_train_test(y_data, lookback)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 将模型设置为训练模式
    model.train()
    # 用于保存每个训练轮次的损失率
    loss_rate = np.zeros(epoch)
    start_time = time.time()
    for i in range(epoch):
        # 通过模型预测数据，并得到 prediction
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print(f"epoch({i+1}/{epoch}), MSE:{loss.item()}")
        loss_rate[i] = loss.item()        

        # 反向传播计算梯度
        loss.backward()
        # 优化模型参数（根据学习率和梯度做参数更新）
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
    print("training time: {}".format(time.time() - start_time))

    # 绘制所有轮次的损失率的曲线图
    if not only_predict_future:
        plot_loss("Training Loss", loss_rate)

    # 把训练数据和预测数据，通过 detach() 去掉梯度信息，然后转为 numpy 数据，然后通过 scaler.inverse_transform() 取消数据的归一
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    # 计算 RMSE（将 MSE 开方）以便衡量模型的预测误差
    train_rmse = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('train rsme: %.2f' % (train_rmse))
    # 绘制训练数据中的，真实值的曲线和预测值的曲线
    if not only_predict_future:
        plot_stock_pred("Training", pd.DataFrame(y_train), pd.DataFrame(y_train_pred))


    # 将模型设置为测试模式，即评估模式
    model.eval()
    if not only_predict_future:
    # 通过模型预测数据，并得到 prediction
        y_test_pred = model(x_test)
        # 把测试数据和预测数据，通过 detach() 去掉梯度信息，然后转为 numpy 数据，然后通过 scaler.inverse_transform() 取消数据的归一
        y_test = scaler.inverse_transform(y_test.detach().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
        # 计算 RMSE（将 MSE 开方）以便衡量模型的预测误差
        test_rmse = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
        print('test rsme: %.2f' % (test_rmse))
        # 绘制测试数据中的，真实值的曲线和预测值的曲线
        plot_stock_pred("Test", pd.DataFrame(y_test), pd.DataFrame(y_test_pred))


    # 预测未来的值
    if only_predict_future:
        x_future = np.expand_dims(y_data[-lookback:], axis=0)
        x_future = torch.from_numpy(x_future).type(torch.Tensor)
        with torch.no_grad():
            preds = []
            for i in range(days_future):
                y_future_pred = model(x_future)
                preds.append(y_future_pred.item())
                x_future_temp = np.append(x_future.numpy(), np.expand_dims(y_future_pred.numpy(), axis=0), axis=1)
                x_future = torch.from_numpy(x_future_temp[:, -lookback:])
        result = scaler.inverse_transform(np.array(preds).reshape(-1,1))
        # 打印数据集中的最后一条数据和预测结果
        print(df[-1:], "\n", result)
        # 绘制最近的真实的曲线图，以及预测的未来的曲线图
        real = pd.DataFrame(scaler.inverse_transform(y_data[-lookback:]))
        real_and_pred = pd.concat([real, pd.DataFrame(result)]).reset_index()
        plot_stock_pred("Future", real, real_and_pred)


# 获取训练数据和测试数据
# 比如 y_data 有 100 条数据，lookback 是 20，现在把他们全部用于训练数据，则一共可以构造出 79 组数据
#   每组数据中，x 有 20 条数据，y 有 1 条数据（这条数据就是 x 的那 20 条数据的后面的那一条）
#   后续要通过 x 预测 y，即通过前 20 条数据预测第 21 条数据的值
def get_train_test(y_data, lookback):
    data = []
    
    for index in range(len(y_data) - lookback): 
        data.append(y_data[index: index + (lookback + 1)])
    
    data = np.array(data)

    test_data_size = 0
    if not only_predict_future:
        test_data_size = int(np.round(test_data_percent * data.shape[0]))
    train_data_size = data.shape[0] - (test_data_size)
    
    x_train = data[:train_data_size, :-1, :]
    y_train = data[:train_data_size, -1, :]
    
    x_test = data[train_data_size:, :-1]
    y_test = data[train_data_size:, -1, :]
    
    # 这里保存的是训练数据的输入值，其有 n 个数据，每个数据里有 lookback 条数据
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    # 这里保存的是每个训练数据对应的真实值
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return x_train, y_train, x_test, y_test

# 自定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 定义一个 LSTM 层
        #   input_dim: 输入数据的维度。比如每个时间点都有 n 个特征，那么这里就设置为 n
        #   hidden_dim: 隐藏层的维度。隐藏层的维度会影响到模型能够捕捉到的复杂性的级别（参考 RNN 的说明）
        #   num_layers: 堆叠的层数，也就是有多少个 LSTM 单元（增加层数可以让模型学习更复杂的模式）
        #   batch_first: 表示输入数据是否以批量的形式首先进入 LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 定义一个线性层，其常常被用作神经网络的最后一层，负责将隐藏层的结果转化为具体的输出
        # 注：fc 的意思是全连接层（fully connected layer）
        #   hidden_dim：输入数据的维度，其来自于模型前面的隐藏层。
        #   output_dim：输出数据的维度，其对应于模型的预测目标。比如你要通过输入数据预测 n 个值，那么这里就设置为 n
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 定义 LSTM 模型的前向传播函数
    def forward(self, x):
        # 定义一个隐藏状态张量 h0，形状为 self.num_layers, x.size(0), self.hidden_dim，并且在反向转播时需要计算梯度
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 和上面类似，这个张量 c0 用于存储 LSTM 的单元状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 通过 LSTM 层将输入 x 和隐藏状态 h0 和单元状态 c0 进行前向传播
        #   out 是 LSTM 模型的输出结果，(hn, cn) 是一个包含隐藏状态和单元状态的元组
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将 LSTM 模型的输出传递给一个全连接层
        #   out[:, -1, :] 的意思是选择 LSTM 输出的最后一个时间步的隐藏状态
        #   最后的输出结果 out 是全连接层的输出，这个输出可以直接用于预测或进一步处理
        out = self.fc(out[:, -1, :]) 
        return out

# 绘制 x, y 曲线图
def plot_xy(title, df):
    # 横坐标上刻度线的数量
    num = 10
    # 横坐标上刻度线之间的数据量
    step = len(df) // num
    # 设置横坐标刻度线的位置，标签文本，标签文件的旋转角度
    plt.xticks(ticks=df['x'][::step].index, labels=df['x'][::step].round(2), rotation=45)
    # 指定纵坐标的数据，并绘制图片
    plt.plot(df['y'])
    # 指定需要在图片上显示的标题
    plt.title(title)
    # 显示图片并阻塞，直到关闭图片
    plt.show()

# 绘制损失率
def plot_loss(title, loss_rate):
    sns.set_style("darkgrid")    
    plt.plot()
    ax = sns.lineplot(data=loss_rate, color='royalblue')
    ax.set_title(title, size=14, fontweight='bold')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    plt.show()

# 绘制真实值的曲线和预测值的曲线
def plot_stock_pred(title, stock_data, pred_data):
    sns.set_style("darkgrid")    
    plt.plot()
    ax = sns.lineplot(x=pred_data.index, y=pred_data[0], label="pred", color='tomato')
    ax = sns.lineplot(x=stock_data.index, y=stock_data[0], label="real", color='royalblue')
    ax.set_title(title, size=14, fontweight='bold')
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_xticklabels('')
    plt.show()


if __name__ == '__main__':
    sample1()