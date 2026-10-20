# 笔记

## 1.张量相关

### 创建部分
| 函数名 | 内容 |
|---|---|
| torch.tensor | 创建一个张量，类似于数组，把内容写在括号内 |
| torch.zeros | 生成一个指定形状的全0张量 |
| torch.ones | 生成一个指定形状的全1张量 |
| torch.randn | 生成一个指定形状的N(0,1)随机数张量 |
| torch.randint(l,r,size) | 生成一个指定形状的[l,r)随机数张量 |
| torch.arange(l,r,step) | 生成一个起于l终于r, 步长为step的等差数列向量 |

### 属性部分
| 函数名 | 内容 |
|---|---|
| .shape | 返回一个tuple子类表示的形状(各维度的轴长) |
| .dtype | 返回一个张量的数据类型 |

### 变形部分
**定义术语:** storage是一块一维内存, stride是告诉我们访问内存的方式以呈现逻辑形状
| 函数名 | 内容 |
|---|---|
| .view | 一个张量本质是一块一维内存, view()即把这块内存按照stride用指定形状表示出来(注意内存还是原来那块, 故访问方式依然是按原来方式访问, 例如转置后张量不连续不能view) |
| .reshape | 和view基本相同, 不同仅在于张量不连续时先把该张量拷贝展平再重新规划形状(原理在连续时完全同view) |
| .permute | 直接重新规划形状, 但实际上只修改了stride, 故张量基本非连续, 需拷贝后才能view | 
| .unsqueeze | 在指定位置插入size=1的维度 |

### 索引
以下以2维为例
| 概念 | 内容 |
|---|---|
| 切片x[..., ...] | 生成一个view, 按各维度要求截取, 如x[:,1]即取出所有行第1列 |
| 布尔索引 | 例如x[x>0]即取x中正数拍成一维, 另外也可以赋值比如x[x>0]=1, 而且x>0可以生成一个和x形状相同的bool张量 |

### 运算
**+, - ,*, /:** 形状相同的张量之间逐个运算(注意如果形状不同但同一维度较小值为1可以按广播规则进行运算)

**@：** 矩阵相乘, 如果三维即以上则最后两维进行矩阵乘, 前面按广播规则处理

### 搬设备
张量默认创建在cpu上, 但张量在CPU张量和GPU张量不能相互计算, 要通过.cpu()把张量搬回到CPU, 或是.to('cuda')搬到GPU

---

## 2.梯度相关
### 数学基础
梯度计算($\mathbf{y}=f(\mathbf{x})$)
|   | $\mathbf{x}为(1,)$ | $\mathbf{x}为(n,1)$ | $\mathbf{x}为(n,k)$
|---|---|---|---|
| $\mathbf{x}为(1,)$ | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(1,)$ |  $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(1,n)$ |$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(k,n)$|
| $\mathbf{x}为(m,1)$ | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,1)$ |  $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,n)$ |$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,k,n)$|
| $\mathbf{x}为(m,l)$ | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,l)$ |  $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,l,n)$ |$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}为(m,l,k,n)$|

特别地, 
1. $\mathbf{y}=\mathbf{Ax},\ \frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\mathbf{A}$
2. $\mathbf{y}=\mathbf{x^{T}A},\ \frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\mathbf{A^T}$
3. $\mathbf{y}=\mathbf{x^{T}Ax},\ \frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\mathbf{A^T+A}$

<br />

性质方面:$\mathbf{链式法则}$可以从标量的链式法则$\frac{\partial z}{\partial x}=\frac{\partial z}{\partial y}\frac{\partial y}{\partial x}$直接拓展而来
### 自动求导(内部计算图实现)
链式法则: $\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u_n}\frac{\partial u_n}{\partial u_{n-1}}...\frac{\partial u_1}{\partial x}$

正向累计: $\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u_n}(\frac{\partial u_n}{\partial u_{n-1}}(...(\frac{\partial u_2}{\partial u_1}\frac{\partial u_1}{\partial x})))$

反向累计/传递: $\frac{\partial y}{\partial x}=((\frac{\partial y}{\partial u_n}\frac{\partial u_n}{\partial u_{n-1}})...)\frac{\partial u_1}{\partial x}$
### 代码相关
|函数名|内容|
|---|---|
|x.require_grad_(True)| 告诉pytorch x是需要计算梯度的变量(也就是计算图中的叶子节点), 也可以在创建的时候声明, 例如x=torch.arange(4,requires_grad=True)|
|y.backward()| 反向传播函数计算y关于x每个分量的梯度, x.grad即存储着所求梯度|
|u=y.detach()| u不当成x的函数而是常数|

注意: 如果更改y, 还要求x.grad之前记得x.grad.zero_()不然会累加到原梯度上无法正确存储

## 3.线性回归

### 线性模型
$$
y=<\mathbf{w},\mathbf{x}>+b
$$
(这是一个单层的$\mathbf{神经网络}$)

### 平方损失(MSE)
$$
\ell(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
$$

### 训练数据
利用已知的n个样本, $\mathbf{X}=[\mathbf{x_1},\mathbf{x_2},...,\mathbf{x_n}]^T$以及$\mathbf{y}=[y_1,y_2,...,y_n]^T$来决定参考值$\mathbf{w},b$

### 训练损失
$$
\ell(\mathbf{X}, \mathbf{y}, \mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \langle \mathbf{x}_i, \mathbf{w} \rangle - b)^2 = \frac{1}{2n} \| \mathbf{y} - \mathbf{X}\mathbf{w} - b \|^2
$$
目标是最小化损失来学习参数:
$$
\mathbf{w}^*, b^*=arg\min_{\mathbf{w},b}\ell(\mathbf{X}, \mathbf{y}, \mathbf{w}, b) 
$$
我们首先做以下变换:(之后$\mathbf{X},b$都代表下述)
$$
\mathbf{X}\leftarrow [\mathbf{X},1],\mathbf{w} \leftarrow \begin{bmatrix} \mathbf{w} \\ b \end{bmatrix}
$$
此时有:
$$
\ell(\mathbf{X}, \mathbf{y}, \mathbf{w}, b) =\frac{1}{2n} \| \mathbf{y} - \mathbf{X}\mathbf{w}  \|^2
$$
由于是凸函数我们可以求出其显式最优解为:
$$
\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 梯度下降
讲解视频:https://www.bilibili.com/video/BV19f421Q7CL/?spm_id_from=333.337.search-card.all.click&vd_source=3d3c1fb49ccaed840ac82a024c1feb1a
- 选一个初始值$\mathbf{w_0}$
- 重复迭代参数(t=1,2,3...)
$$
\mathbf{w}_t = \mathbf{w}_{t-1} - \eta \frac{\partial \ell}{\partial \mathbf{w}_{t-1}}
$$
- 沿梯度方向增加损失函数值, 其中$\eta$是学习率

然而求梯度这件事对于计算机来说很"贵", 我们可以采取$\mathbf{小批量随机梯度下降}$的策略:

