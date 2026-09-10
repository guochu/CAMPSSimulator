# CAMPSSimulator.jl

基于 **Clifford 增广矩阵乘积态（Clifford-augmented MPS，CAMPS）** 的通用量子线路
模拟后端，是 [`QuantumCircuits`](../../../QuantumCircuits) 线路 IR 的一个后端实现，
与同目录的 [`MPSSimulator`](../../../MPSSimulator) **接口对齐但内核独立**，可互换使用
并交叉验证。

核心思想是把态写成“Clifford 酉帧 + 魔法 MPS”的乘积：

```math
|\psi\rangle \;=\; U_{\rm frame}\,|\varphi\rangle .
```

* `U_frame`（[`CliffordFrame`](states.md)）——所有入站 **Clifford 门**
  （H / S / CX / CZ / SWAP / √X …）以精确矩阵形式累积在帧中，**不触碰 MPS、零截断**；
* `|φ⟩`（普通 MPS）——只承载“魔法（非稳定子）”成分，其键维由 `trunc` 控制。

相比普通 MPS 模拟器，CAMPS 的收益是：**当 Clifford 纠缠能长期留在帧外时，MPS 用更小的
键维即可达到相同精度**（在 Clifford 主导 + 稀疏 magic 的深线路上可差一个数量级以上，
见 [tests.md](tests.md)）。

约定（与 QuantumCircuits / MPSSimulator 一致）：比特索引 **1-based、小端序**
（qubit 1 = 最低有效位）；门矩阵 `positions` 的**第一个比特为矩阵最高位**
（MSB-first）。

## 一个简单的例子：Bell 态

```julia
using QuantumCircuits, CAMPSSimulator

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))

ψ = simulate(c, zero_state(2))        # 终态是 CAMPS（simulate! 就地）
println("max bond = ", max_bond_dimension(mps(ψ)))   # 1：H 与 CX 都进了帧
println("probabilities = ", marginal_probabilities(ψ, 1))
```

等价地，也可以绕开 `Circuit`，直接用矩阵作用（`positions[1]` = 最高位）：

```julia
using LinearAlgebra
H  = [1 1; 1 -1] ./ sqrt(2)
CX = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
ψ = zero_state(2)
apply!(ψ, H, (1,))
apply!(ψ, CX, (1, 2))
w = [1, 0, 0, 1] ./ sqrt(2)
println(abs(dot(statevector(ψ), w))^2)   # ≈ 1.0
```

## 长程门、多比特门与截断

非相邻比特的任意 `k` 比特门（含 `k > 2` 的 Toffoli 等）自动经相邻 SWAP 网络
**聚集 → 作用 → 散开**（见 [gates.md](gates.md)）。所有作用型接口
（`apply!` / `simulate` / `measure!`）都接受 `trunc` 关键字：

```julia
ψ = zero_state(6)
# CRZ(θ=0.7)@(q1, q6)：非相邻两比特非 Clifford 门
# → 帧共轭后自动 SWAP 聚集 → 作用 → 散开，并按 trunc 截断
G = Matrix{ComplexF64}(I, 4, 4); G[4, 4] = exp(im * 0.7)
apply!(ψ, G, (1, 6); trunc = truncdim(16))
```

（相邻两比特的 Clifford 门则直接进帧、零截断；只有非 Clifford / 非相邻的魔法门才会
走 [gates.md](gates.md) 描述的聚集与截断路径。）

* Clifford 门（≤2 比特）：零截断进帧，MPS 键维不变；
* 非 Clifford 门（T / RY / … 以及一切 >2 比特门）：先经帧共轭
  `frame† g frame` 再作用到 MPS 上按 `trunc` 截断；共轭支撑超过 `wmax`
  时先折叠帧（[policies.md](policies.md)）。

## 特性一览

* **`|ψ⟩ = U_frame|φ⟩` 双表示**：`CAMPS{T,R,P}` 全类型参数化（immutable），
  `frame(ψ)` / `mps(ψ)` 访问两半；
* **混合正则形式存储**：站点张量保持右正则，`svectors` 记录各键真实 Schmidt 谱；
  门作用为 Hastings 式更新，**幺正演化全程不破坏规范 / 键谱**；
* **可切换的帧更新策略（FramePolicy）**：`FoldFramePolicy`（默认，折入即清空）与
  `BondRefreshPolicy`（折入 + 逐键重选 Clifford 解纠缠并留帧），
  见 [policies.md](policies.md)；
* **截断处 Clifford 解纠缠搜索（CAMPS 增强）**：只搜索非局域类代表元
  （I / CNOT / iSWAP / SWAP），误差永不超过普通截断；
* **任意拓扑门**：任意位置、任意比特数的门都支持；
* **QuantumCircuits.Interface 后端**：`CAMPSBackend` 强模拟 + 逐 shot 采样、
  中途测量、经典条件反馈（`IfOp`），见 [backend.md](backend.md)；
* **帧可视化 / 统计**：`draw_frame`（画风同 `QuantumCircuits.draw`）、
  `frame_stats`、`frame_gate_name`。

## 环境要求与安装

- Julia ≥ 1.10；
- 本地依赖：[`QuantumCircuits`](../../../QuantumCircuits)（线路 IR 与接口协议）；
- 测试 / 性能验证额外用到 [`MPSSimulator`](../../../MPSSimulator)（普通 MPS 参照）。

本仓库为本地开发布局（与 `QuantumCircuits` / `MPSSimulator` 平级），两种接入方式：

```julia
# 方式一：开发环境经 LOAD_PATH 解析（顺序很重要：QuantumCircuits 必须先入栈）
push!(LOAD_PATH, ".../QuantumCircuits")
push!(LOAD_PATH, ".../CAMPSSimulator")

# 方式二：Pkg 开发挂接
using Pkg
Pkg.develop(path = ".../QuantumCircuits")
Pkg.develop(path = ".../CAMPSSimulator")
```

## 文档导航

* [states.md](states.md) —— 态表示：MPS 与键谱、CAMPS、Clifford 帧、初始化；
* [policies.md](policies.md) —— **帧更新策略**：`FoldFramePolicy` 与
  `BondRefreshPolicy` 的具体实现方式与差异（本文档重点）；
* [gates.md](gates.md) —— 门作用内核：Hastings 更新、SWAP 聚集/散开、截断方案、
  解纠缠搜索、共轭窗与折叠时机；
* [measure.md](measure.md) —— 测量 / 期望 / 幅度 / 态矢物化 / 帧可视化；
* [backend.md](backend.md) —— QuantumCircuits 桥接层与 `CAMPSBackend`；
* [tests.md](tests.md) —— 测试覆盖与 CAMPS vs MPS 的效果对照结论。
