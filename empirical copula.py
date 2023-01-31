import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# 导入数据

os.chdir(r"D:\量化金融风险管理\第二次作业")
df = pd.read_table("期权+股价.txt", sep="\t", header=0, encoding="UTF-8", index_col=0)
stock = (np.asarray(df))
apple = (stock[:, 0]).tolist()
amd = (stock[:, 1]).tolist()

''''
如果想绘制（C1, X2）的 copula 图像，那么把文件“股价数据.txt” 中 apple 的日度股价数据按照 Eq(7) 替换为期权的日度价格即可。计算方法如下：
option = []

K = 50
w = 1

for i in apple:
    p = 0.5 * (i - K) * (1 + math.erf((i - K) / (2 * w ** 2) ** 0.5)) + w / (2 * math.pi) ** 0.5 * math.exp(
        -(i - K) ** 2 / (2 * w ** 2))
    option.append(p)
    p = 0
'''

# 定义联合经验分布函数
def cdf(x, y):
    n = len(stock)
    F = 0
    for i in stock:
        if i[0] <= x and i[1] <= y:
            F += 1
    F = F / n
    return F

# 定义边缘经验分布函数
def mcdf(x, data):
    n = len(stock)
    F = 0
    for i in data:
        if i <= x:
            F += 1
    F = F / n
    return F


# the Python code of Eq (5) and (6)
def invmcdf(u, data):
    F = 1000000
    for i in data:
        if mcdf(i, data) >= u and i < F:
            F = i
    return F


# 定义 empirical copula 的 CDF
def copula(u, v):
    C = cdf(invmcdf(u, apple), invmcdf(v, amd))
    return C


# 定义一维的 Dirac delta，用于平滑一元经验分布的 pdf
def dirac(x, x0):
    band = 3
    d = math.exp(-1 / (2 * band ** 2) * (x - x0) ** 2) / ((2 * math.pi) ** 0.5 * band)
    return d


# 定义二维的 Dirac delta，用于平滑二元经验分布的 pdf
def twodirac(x, y, x0, y0):
    band = 3
    d = math.exp(-1 / (2 * band ** 2) * ((x - x0) ** 2 + (y - y0) ** 2)) / ((2 * math.pi) ** 1 * band ** 2)
    return d


# 定义二元经验分布的 pdf
def pdf(x, y):
    f = 0
    for i in stock:
        f += twodirac(x, y, i[0], i[1])
    f = f / len(stock)
    return f

# 定义一元（边缘）经验分布的 pdf
def mpdf(x, data):
    f = 0
    for i in data:
        f += dirac(x, i)
    f = f / len(stock)
    return f


# 定义 empirical copula 的 pdf
def pcopula(u, v):
    p = pdf(invmcdf(u, apple), invmcdf(v, amd)) / (mpdf(invmcdf(u, apple), apple) * mpdf(invmcdf(v, amd), amd))
    return p


# 作图
u = np.arange(0, 1.1, 0.1)
X, Y = np.meshgrid(u, u)

Z = []
for i in range(0, 11):
    Z.append([])
    for j in range(0, 11):
        Z[i].append(pcopula(X[i][j], Y[i][j]))

Z = np.asarray(Z)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('u', fontsize=15)
ax.set_ylabel('v', fontsize=15)
ax.set_zlabel('C', fontsize=15)
ax.set_xlim(0, 1)  # X轴，横向向右方向
ax.set_ylim(0, 1)  # Y轴,左向与X,Z轴互为垂直
ax.set_zlim(0, 20)  # 竖向为Z轴
surf = ax.plot_surface(X, Y, Z, cmap='rainbow')
fig.colorbar(surf, shrink=0.3, aspect=10)
plt.title('The cdf of the empirical copula of (X1, X2)', fontsize=15)
plt.show()
