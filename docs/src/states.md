# 态与表示

CAMPSSimulator 的态为 **CAMPS（Clifford 增广 MPS）**：物理态写成
`|ψ⟩ = U_frame |φ⟩`，其中 `|φ⟩` 是普通纯态 MPS（承载魔法成分），`U_frame` 是
Clifford 帧（承载全部稳定子纠缠）。底层 MPS 的表示与规范约定与 MPSSimulator
**完全一致**（右正则张量 + 键谱 `svectors`），因此两套内核可以互相引用、交叉验证。

## 数据布局（底层 MPS，混合正则形式）

底层 MPS 为 `FiniteMPSAlgorithms.CanonicalMPS` 的**薄封装**（字段 `core`，
[`src/mps.jl`](../../src/mps.jl)）：存储、键谱与正交化复用 FiniteMPSAlgorithms，
封装层保持 `scaling == 1`（范数在站点数据中）。底层布局为：

* **站点张量**：`N` 个 `A::Array{T,3}`，轴序 = `(左键, 物理维, 右键)`，
  开放边界（`size(ψ[1],1)==1`、`size(ψ[end],3)==1`），物理维恒为 2。
  **量子态就是张量网络的普通收缩**——网络中不额外含权重结点；
* **键谱 `core.s`**：长度 `N+1` 的 Schmidt 谱数组，`core.s[b+1]` 为键 `b`
  （站点 `b` 与 `b+1` 之间）的 **Schmidt 谱（真实奇异值）**；边界键为 `[1]`，
  `missing` 表示谱未知（构造后被 `canonicalize!` 填上）。

当所有站点张量**右正则**（`(左键, 物理×右键)` 矩阵行正交）且键谱正确时，态处于
**混合正则形式**（`iscanonical(mps(ψ))` 为 `true`）：

* 前缀约化密度矩阵 = `diag(Λ²)`；
* **幺正演化严格保持该形式**：门内核用 Hastings 式谱预条件 SVD（见
  [gates.md](gates.md)），每一步都把新键谱写回 `svectors`，截断也是按各键
  bipartition 的**真实 Schmidt 奇异值**进行；
* 非幺正算子（测量投影 / 重置）会破坏正则性——状态仍精确，调用方随后用
  `canonicalize!` 恢复。

## CAMPS 态类型

```julia
struct CAMPS{T<:Number,R<:Real,P<:FramePolicy}
    data::MPS{T,R}            # 魔法 MPS |φ⟩
    frame::CliffordFrame{T}   # Clifford 帧 U_frame
    policy::P                 # 帧更新策略（类型参数化，避免动态分发）
end
```

（[`src/camps.jl`](../../src/camps.jl)，另见 [policies.md](policies.md)。）

要点：

* 结构体 **immutable 且完全类型参数化**：`T` = 标量类型、`R = real(T)`、
  `P <: FramePolicy` 为策略类型——没有 UnionAll / 抽象字段装箱，利于 JIT；
* 内部 `data` / `frame` 仍是可变对象，因此 `fold_frame!` / `apply!` 等仍然就地演化；
* 切换策略用 `with_policy(ψ, policy)`（浅引用换策略标记，返回新 CAMPS），
  不要直接改字段；
* `mps(ψ)` / `frame(ψ)` 返回两半的浅引用（深拷贝用 `copy(ψ)`，逐站点 + 帧门都复制）。

## Clifford 帧

`CliffordFrame` 是有序的局域 Clifford 门列表（[`src/cliffframe.jl`](../../src/cliffframe.jl)）：

```julia
mutable struct CliffordFrame{T<:Number}
    gates::Vector{LocalOp{T}}
end
```

* `LocalOp(pos, mat)`：`pos` **升序**、矩阵按**内部小端序**存储
  （bit0 ↔ 最小站点 = LSB），维数 `2^|pos| × 2^|pos|`；
* **`gates` 列表顺序 = 作用顺序**：`gates[1]` 最先作用、最靠近 MPS；
  态 = `U_frame|φ⟩`，其中 `U_frame = gates[end] ⋯ gates[1]`；
* 入站 Clifford 门 `push!` 到列表末尾（作用在最顶层）；
* 截断处选中的解纠缠门 `C` 以 `adjoint(C)` **前插**（作用在 MPS 侧），
  见 [policies.md](policies.md)。

帧为空 `isempty(frame)` 表示 `U_frame = I`。

### 帧可视化

`draw_frame(frame[, n])` 以 ASCII 画风（同 `QuantumCircuits.draw`）打印帧，
`frame_stats` 计数、`frame_gate_name` 给门命名（含任意控制端朝向的 CX 识别）：

```
q[1]: ─H──●────
          │    
q[2]: ────X──S─
```

上例 = `H@q1 → CX(控制=q1, 目标=q2) → S@q2`：CNOT 控制端 `●` 与目标端 `X`
之间有纵向连接线 `│`。详见 [measure.md](measure.md) 帧可视化一节。

## 初始化

`|0…0⟩`、计算基、直积编码、随机态——返回 **CAMPS（帧为空）**：

* `zero_state([T=ComplexF64,] n)` —— `|0…0⟩`；
* `onehot_state([T,] bits)` —— 计算基态（`bits[k]` = qubit `k` 取值）；
* `qubit_encoding_state([T,] θs)` —— 直积态 `cos(πθ/2)|0⟩ + sin(πθ/2)|1⟩`；
* `rand_state([T,] n; D=8, rng=...)` —— 键维上限 `D` 的随机态（自动 `canonicalize!` 填谱）。

模拟态统一使用复标量（默认 `ComplexF64`），因此实 Clifford 门也无需运行时类型转换。

## 查询与规范型

* 键维：`bond_dimensions(ψ)` / `max_bond_dimension(ψ)`（作用于 CAMPS 时取
  其 MPS 部分，即**魔法部分的键维**）；
* 规范：`iscanonical(mps(ψ))`（需先取 `mps(ψ)`，CAMPS 本身无此方法）；
* 恢复规范 / 键谱：`canonicalize!(mps(ψ); trunc=NoTruncation(), normalize=true)`，
  返回最大单键截断误差；
* 范数：`norm(ψ)` = `norm(mps(ψ))`（帧为酉，物理范数即 MPS 范数）；
  `normalize!(ψ)` 就地归一到单位范数。

## 与 MPSSimulator 的关系

| | MPSSimulator | CAMPSSimulator |
|---|---|---|
| 态 | `AbstractMPS`（MPS / DensityOperatorMPS） | `CAMPS{T,R,P}` = 帧 × MPS |
| 底层 MPS 布局 | 右正则张量 + `svectors` | 相同（独立实现） |
| 门内核 | Hastings 逐键更新 | 相同思想（独立实现） |
| 长程 / 多比特门 | SWAP 聚集 → 作用 → 散开 | 相同 |
| 噪声 / 混合态 | 支持（局域维 4） | **不支持（v1）** |
| 特色 | 完整可观测量 / 采样协议 | Clifford 帧 + 两种帧更新策略 |

CAMPSSimulator **不依赖也不修改** MPSSimulator 的源码，仅在测试 / 性能验证中
把它当作普通 MPS 参照后端。
