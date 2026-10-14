import torch
A=torch.randn(2,3,4) #batch=2(2张图), height(行)=3, width(列)=4
print(A)
B=A.permute(0,2,1)#permute成(2,4,3), 数据没有变
print(B)
x=torch.randint(1,11,(2,3))
y=torch.ones(3,5,dtype=torch.int64)
z=torch.mm(x,y)
print(x)
print(y)
print(z)
P=torch.randn(8,3,32,32)#假图像
print(P[0].size())#第0张图像
print(P[:,0].size())#所有图像的红色通道
m=torch.tensor([[2],[4],[5],[8]])
n=torch.tensor([1,2,3])#本质是计算时stride为0
print(m+n)