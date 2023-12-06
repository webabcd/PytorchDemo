import torch
import math

# 自动求导的简单说明
def sample1():
    # requires_grad 用于表示是否需要自动求导
    k = torch.tensor(1.23, requires_grad=True)
    x = torch.tensor(4.56, requires_grad=True)
    b = torch.tensor(7.89, requires_grad=True)
    y = k*x + b
    # 反向传播以便自动计算梯度
    y.backward()
    # 获取 x 的导数（对于一次函数来说，x 的导数就是斜率 k）
    print(x.grad) # tensor(1.2300)

    x = torch.tensor(0., requires_grad=True)
    y = torch.sin(x)
    y.backward()
    # sin(x) 的导数是 cos(x)
    print(x.grad) # tensor(1.)

    x = torch.tensor([0, math.pi/2, math.pi, math.pi/2*3, math.pi*2], requires_grad=True)
    y = torch.sin(x)
    # 标量可以直接 backward()
    # 但是这里的 y 是个矢量，对于非标量来说，其在调用 backward() 时要传入相同形状的参数
    y.backward(torch.ones(5))
    # sin(x) 的导数是 cos(x)
    print(x.grad) # tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00,  1.1925e-08,  1.0000e+00])

    x = torch.tensor(1.11, requires_grad=True)
    y = x * 2
    y = y * 2
    y = y * 2
    y.backward()
    # 张量的所有梯度将会自动累加到​​.grad​​ 属性
    print(x.grad) # tensor(8.)
    

if __name__ == '__main__':
    # 自动求导的简单说明
    sample1()

