# -*- coding: utf-8 -*-
"""
Day01 · 张量基本功 —— 5 个必做练习（完整可运行版）
================================================================================
对应《学习计划/Week01/Day01/README.md》里的「必做练习 1~5」。

用法：
    python day01_tensor.py

每个练习都带 self-check（✅/❌），全部 ✅ 就说明你今天这关过了。
"""
import torch

SEP = "=" * 78


def check(label, cond):
    """打勾自检"""
    print(f"    [{'OK ' if cond else 'FAIL'}] {label}")
    return bool(cond)


def probe(name, t, base=None):
    """统一探针：形状 / stride / 是否连续 / 是否与原张量共享内存"""
    s = (f"    {name:<18} shape={str(t.shape):<18} "
         f"stride={str(t.stride()):<20} contiguous={t.is_contiguous()}")
    if base is not None:
        s += f"  shared={t.data_ptr() == base.data_ptr()}"
    print(s)
    return t


# ---------------------------------------------------------------------------
def ex1():
    print(SEP)
    print("练习1：建一个 (2,3,4) 的随机张量，解释 3 个维度分别代表什么")
    print(SEP)

    A = torch.randn(2, 3, 4)          # 约定：NCHW 里的 NHW；或 (batch, 行, 列)

    print(f"    A.shape = {tuple(A.shape)}   dtype={A.dtype}   device={A.device}")
    print(f"    A.stride() = {A.stride()}   元素个数 numel = {A.numel()}")
    print("    维度含义（按 CV 惯例）：")
    print("      dim0 = 2  -> batch：2 张图（或 2 个样本）")
    print("      dim1 = 3  -> height：每张图 3 行")
    print("      dim2 = 4  -> width ：每行 4 个像素")
    print("    stride 解读：dim0 跨一步跳 12 个内存元素(=3*4)，dim1 跳 4 个，dim2 跳 1 个")

    check("A 的形状是 (2,3,4)", tuple(A.shape) == (2, 3, 4))
    check("A 是连续张量", A.is_contiguous())
    check("A.numel() == 2*3*4", A.numel() == 24)
    return A


# ---------------------------------------------------------------------------
def ex2(A):
    print(SEP)
    print("练习2：把它 permute 成 (2,4,3)，说清数据有没有变")
    print(SEP)

    B = A.permute(0, 2, 1)            # 交换后两维

    probe("A", A)
    probe("B=A.permute(0,2,1)", B, base=A)

    print(f"    B[0,0] = {B[0, 0].tolist()}")
    print(f"    A[0,:,0] = {A[0, :, 0].tolist()}   <- 完全一样")
    print("    结论：数据一个都没动（零拷贝、共享storage），只是“读取顺序”改变了；")
    print("          所以 B 不是连续张量 —— 这正是 .view() 会失败的根源。")

    check("B 的形状是 (2,4,3)", tuple(B.shape) == (2, 4, 3))
    check("B 与 A 共享内存（零拷贝）", B.data_ptr() == A.data_ptr())
    check("B[0,0] 等于 A[0,:,0]（数据没变）", torch.equal(B[0, 0], A[0, :, 0]))
    check("B 不是连续张量", not B.is_contiguous())

    try:
        B.view(-1)
        print("    [注意] B.view(-1) 居然成功了？请检查 torch 版本")
    except RuntimeError as e:
        print(f"    B.view(-1) 报错 -> {str(e)[:70]}...")
        print("    正解：B.contiguous().view(-1) 或 B.reshape(-1)（会拷贝一份）")
    return B


# ---------------------------------------------------------------------------
def ex3():
    print(SEP)
    print("练习3：做一次 (2,3) @ (3,5)，解释为什么结果是 (2,5)")
    print(SEP)

    x = torch.randint(1, 11, (2, 3))
    y = torch.ones(3, 5, dtype=torch.int64)
    z = torch.mm(x, y)                # 等价于 x @ y

    print(f"    x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}  z.shape={tuple(z.shape)}")
    print(f"    z = {z.tolist()}")
    print("    规则：@ 只对最后两维做矩阵乘，要求 “左的最后一维 == 右的倒数第二维”（这里都是 3），")
    print("          这个公共维被“消掉”，剩下的维拼起来 -> (2, 5)。")
    print("          即 (2,3) @ (3,5) -> (2,5)；前面的 batch 维只做广播，不参与收缩。")

    check("z 的形状是 (2,5)", tuple(z.shape) == (2, 5))
    check("z == x.sum(dim=1, keepdim=True)（因为 y 全是 1）",
          torch.equal(z, x.sum(dim=1, keepdim=True).expand(2, 5)))
    return z


# ---------------------------------------------------------------------------
def ex4():
    print(SEP)
    print("练习4：造一个 batch 的假图像 (8,3,32,32)，做 x[0].shape、x[:,0].shape 并解释")
    print(SEP)

    x = torch.randn(8, 3, 32, 32)     # N=8, C=3, H=W=32
    probe("x", x)
    print("    布局 NCHW：8 张图 × 3 通道(RGB) × 32 高 × 32 宽")
    print(f"    x.stride() = {x.stride()}  ->  换一张图跳 3072 个元素，换一个通道跳 1024 个")

    a = x[0]                          # 整数索引：消掉 batch 维
    b = x[:, 0]                       # dim0 全取，dim1 取 0：消掉通道维
    c = x[:, 0:1]                     # 切片 0:1：保留通道维，长度 1

    print("")
    probe("x[0]", a, base=x)
    probe("x[:, 0]", b, base=x)
    probe("x[:, 0:1]", c, base=x)

    print("")
    print(f"    x[0].shape      = {tuple(a.shape)}   -> 第 0 张图的全部 3 个通道 (C,H,W)")
    print(f"    x[:, 0].shape   = {tuple(b.shape)}   -> 全部 8 张图的第 0 号通道(比如 R) (N,H,W)")
    print(f"    x[:, 0:1].shape = {tuple(c.shape)}   -> 同上，但通道维被保留 (N,1,H,W)")

    check("x[0].shape == (3,32,32)", tuple(a.shape) == (3, 32, 32))
    check("x[:, 0].shape == (8,32,32)", tuple(b.shape) == (8, 32, 32))
    check("x[:, 0:1].shape == (8,1,32,32)", tuple(c.shape) == (8, 1, 32, 32))
    check("x[0] 与 x 共享内存（是 view，不是拷贝）", a.data_ptr() == x.data_ptr())
    check("x[:, 0] 也与 x 共享内存（是 view）", b.data_ptr() == x.data_ptr())

    # 证据 1：view 是“同一块内存的另一个视角”
    x[0, 0, 0, 0] = 999.0
    check("改 x[0,0,0,0] 后 x[0][0,0,0] 也变了（共享内存）", x[0, 0, 0, 0].item() == 999.0)

    # 证据 2：连续性差异直接决定 view 能不能用
    print("")
    check("x[0] 是连续的（所以能零拷贝 view）", a.is_contiguous())
    check("x[:, 0] 不是连续的（图与图之间有空洞）", not b.is_contiguous())
    print(f"    x[0].view(-1).shape = {tuple(a.view(-1).shape)}   <- 一张图展平成 3072 维向量")
    try:
        b.view(-1)
        print("    [注意] x[:,0].view(-1) 居然成功了？请检查 torch 版本")
    except RuntimeError as e:
        print(f"    x[:,0].view(-1) 报错 -> {str(e)[:70]}...")
    r = b.reshape(-1)
    print(f"    x[:,0].reshape(-1).shape = {tuple(r.shape)}  "
          f"shared={r.data_ptr() == x.data_ptr()}  <- 发生了拷贝")
    print("    展平顺序是“逻辑行优先”：先第0张图的R通道，再第1张图的R通道 ...")

    # 证据 3：整数索引 vs 长度1切片
    print("")
    print("    关键区别：x[:, 0] 把通道维“吃掉”了；x[:, 0:1] 只是把该维长度变成 1。")
    print("    需要 4 维形状时用 x[:, 0:1] 或 x[:, 0].unsqueeze(1)。")
    check("x[:, 0].unsqueeze(1).shape == x[:, 0:1].shape",
          tuple(b.unsqueeze(1).shape) == tuple(c.shape))
    return x


# ---------------------------------------------------------------------------
def ex5():
    print(SEP)
    print("练习5：广播实验 —— (4,1) 和 (1,3) 相加，结果形状是多少？为什么？")
    print(SEP)

    p = torch.arange(4).float().view(4, 1)     # (4,1)
    q = torch.arange(3).float().view(1, 3)     # (1,3)
    r = p + q

    print(f"    p.shape={tuple(p.shape)}  q.shape={tuple(q.shape)}  (p+q).shape={tuple(r.shape)}")
    print(f"    p+q =\n{r}")
    print("    广播规则（从最后一维往前对齐）：")
    print("      1) 先把两个形状右对齐，缺的前面补 1  -> (4,1) 与 (1,3)")
    print("      2) 每一维：相等则保留；有一个是 1 则“拉伸”成另一个的大小；都既不等也不是 1 -> 报错")
    print("      3) 结果形状 = 每一维取最大值      -> (4,3)")
    print("    内部实现：尺寸为 1 的维在 stride 里是 0，不真正复制数据（零拷贝广播）。")

    check("(p+q).shape == (4,3)", tuple(r.shape) == (4, 3))
    check("p 被拉伸成 stride 0 的广播", p.expand(4, 3).stride() == (1, 0))
    check("r[2,1] == p[2,0] + q[0,1]", r[2, 1].item() == p[2, 0].item() + q[0, 1].item())
    return r


# ---------------------------------------------------------------------------
def main():
    print(f"torch version = {torch.__version__}   cuda available = {torch.cuda.is_available()}")
    A = ex1()
    ex2(A)
    ex3()
    ex4()
    ex5()
    print(SEP)
    print("全部跑完。接下来：把每一节的“逐行解释”写进 notes/day01.md 再 git commit。")


if __name__ == "__main__":
    main()
