# -*- coding: utf-8 -*-
"""
练习4 的「无 torch」先行验证版（numpy 镜像）
================================================================================
用途：还没装好 PyTorch 时，先用 numpy 把第 4 题的结论一次跑通、眼见为实。
原理：numpy 的「基本切片」语义与 PyTorch 完全一致（都是 view + stride 机制），
      唯一区别是 numpy 的 strides 以“字节”为单位，而 torch 的 stride 以“元素”为单位。

用法：
    python day01_ex04_numpy_check.py
"""
import numpy as np

SEP = "=" * 78
ok = 0
fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
    print(f"    [{'OK ' if cond else 'FAIL'}] {label}")
    return bool(cond)


def probe(name, t, base=None):
    """stride 换算成“元素个数”，方便和 torch 对照"""
    el = np.array(t.strides) // t.itemsize
    s = (f"    {name:<18} shape={str(t.shape):<18} "
         f"stride={str(tuple(el)):<20} contiguous={t.flags['C_CONTIGUOUS']}")
    if base is not None:
        s += f"  shared={np.shares_memory(t, base)}"
    print(s)
    return t


print(SEP)
print("练习4（numpy 镜像版）：batch 假图像 (8,3,32,32) 的切片实验")
print(SEP)

# 用 arange 而不是随机数，这样每个像素的值 == 它在内存里的下标，一眼能看出布局
x = np.arange(8 * 3 * 32 * 32, dtype=np.float32).reshape(8, 3, 32, 32)

probe("x", x)
print("    布局 NCHW：8 张图 × 3 通道 × 32 高 × 32 宽")
print("    因为值就是内存下标，可以直接读出偏移：")
print(f"      x[0,0,0,0] = {x[0,0,0,0]}      (第0张图 第0通道 第0行 第0列)")
print(f"      x[0,1,0,0] = {x[0,1,0,0]}    (换到第1个通道)")
print(f"      x[1,0,0,0] = {x[1,0,0,0]}   (换到第1张图)  -> 3072 = 3*32*32")
print("")

a = x[0]
b = x[:, 0]
c = x[:, 0:1]

probe("x[0]", a, base=x)
probe("x[:, 0]", b, base=x)
probe("x[:, 0:1]", c, base=x)
print("")

check("x.shape == (8,3,32,32)", x.shape == (8, 3, 32, 32))
check("x[0].shape == (3,32,32)  —— 第 0 张图的全部 3 个通道 (C,H,W)",
      a.shape == (3, 32, 32))
check("x[:, 0].shape == (8,32,32) —— 全部 8 张图的第 0 号通道 (N,H,W)",
      b.shape == (8, 32, 32))
check("x[:, 0:1].shape == (8,1,32,32) —— 同上但保留通道维",
      c.shape == (8, 1, 32, 32))
check("x[0] 与 x 共享内存（是 view，零拷贝）", np.shares_memory(a, x))
check("x[:, 0] 也与 x 共享内存（是 view，零拷贝）", np.shares_memory(b, x))
check("x[0] 仍然连续（stride=(1024,32,1)）", a.flags['C_CONTIGUOUS'])
check("x[:, 0] 不连续（stride=(3072,32,1)，图与图之间有空隙）", not b.flags['C_CONTIGUOUS'])
print("")

# 证据 1：连续性差异 -> 决定 .view() 能不能用（numpy 的 reshape 同理会拷贝）
print(f"    x[0].reshape(-1).shape   = {a.reshape(-1).shape}    <- 可以，且不拷贝")
print(f"    x[:,0].reshape(-1)[:6]   = {b.reshape(-1)[:6]}  <- 能跑，但触发了拷贝")
check("x[:,0].reshape(-1) 首 4 个元素来自第 0 张图的 R 通道（逻辑行优先顺序）",
      np.array_equal(b.reshape(-1)[:4], np.array([0., 1., 2., 3.], dtype=np.float32)))
check("x[:,0].reshape(-1) 第 1025 个元素来自第 1 张图的 R 通道",
      b.reshape(-1)[1024] == 3072.0)
print("")

# 证据 2：view 就是“同一块内存的另一个视角”（放在最后做，因为它会改数据）
a[0, 0, 0] = 999.0
check("改 x[0][0,0,0] 后 x[0,0,0,0] 也变成 999（共享内存）", x[0, 0, 0, 0] == 999.0)
print("")

print(SEP)
print(f"结论汇总： {ok} 项通过, {fail} 项失败")
print("在 torch 里把 numpy 换成 torch 就是练习4 的答案：")
print("  x[0].shape     -> torch.Size([3, 32, 32])   第0张图，保留通道")
print("  x[:, 0].shape  -> torch.Size([8, 32, 32])   所有图的第0通道")
print("  x[:, 0:1].shape-> torch.Size([8, 1, 32, 32]) 同上但保维")
print(SEP)
