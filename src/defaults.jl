# defaults.jl — 全局默认值（与 MPSSimulator 语义一致）
#
# 所有作用型接口（`apply!` / `measure!` / `simulate` 等）接受的
# `trunc` / `wmax` / `search` 关键字默认取这里的常量。

"""
    DefaultTruncation

默认截断方案：按维数截断到 `50`（与 MPSSimulator 的 `DefaultMPSTruncation` 一致）。
"""
const DefaultTruncation =  truncdimcutoff(50, 1.0e-8)

"""
    DefaultConjWindow

Clifford 帧共轭允许的最大支撑窗大小（比特数）：入站非 Clifford 门经帧共轭后
若支撑超过该上限，先执行帧折叠（`fold_frame!`）再作用。
"""
const DefaultConjWindow = 8

"""
    DefaultSearchDisentangler

两比特（截断型）更新时是否搜索最优 Clifford 解纠缠门（CAMPS 核心增强）。
由于截断损失在局域 Clifford 左乘/右乘下不变，搜索只需覆盖几个非局域类
代表元（CNOT / iSWAP / SWAP），成本很低，故默认开启。
"""
const DefaultSearchDisentangler = true
