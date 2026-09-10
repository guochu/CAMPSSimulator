# 帧更新策略：FoldFramePolicy 与 BondRefreshPolicy

本文档深入讲清两种“帧更新策略”（`FramePolicy`）的**具体实现方式**与
**差异**。它们回答同一个问题：

> 当 Clifford 帧必须“折入” MPS（`fold_frame!`）时，怎么折？折完帧里还留什么？

## 0. 什么时候需要“折叠帧”？

帧的语义是 `|ψ⟩ = U_frame |φ⟩`：Clifford 门进帧**不碰 MPS**，这只有在帧能保持
“局域”时才划算。出现以下情况就必须把帧作用进 MPS：

1. **共轭超窗**：要作用一个非 Clifford 门（魔法）时，需先算有效算子
   `g_eff = frame† · g · frame`（`conj_through_frame`）。逐帧门共轭会让
   `g_eff` 的支撑（涉及站点数）膨胀；一旦超过 `wmax`（默认 `DefaultConjWindow = 8`），
   继续共轭就不如直接把帧折进 MPS，再局部地作用 `g`。
   见 [gates.md](gates.md)。
2. **测量 / 相干重置**：`measure!` 与 `reset_qubit_zero!` 需要在计算基上做数据侧
   投影（非幺正），投影前把帧折掉，使表示回到“纯 MPS + 空帧”再投影。
3. 用户显式调用 `fold_frame!(ψ; trunc)`。

折叠把累积的 Clifford 结构一次（或逐门）压进 MPS，**必然引起键维增长**；若超过
`trunc` 上限就发生截断。策略决定“压进去时是否尝试把其中一部分 Clifford 结构再捞回
帧外”，从而影响**截断后的精度与键维轨迹**。

## 1. 两种策略的类型

```julia
abstract type FramePolicy end

"""经典帧折叠：折入时普通截断、清空帧（当前默认）。"""
struct FoldFramePolicy <: FramePolicy end

"""折入 + 逐键重选解纠缠（per-bond Clifford disentangler 刷新）。"""
struct BondRefreshPolicy <: FramePolicy end

const DefaultFramePolicy = FoldFramePolicy()   # 默认策略
```

CAMPS 态把策略作为类型参数（`CAMPS{T,R,P}`），因此策略在编译期就确定，没有动态分发。
切换策略：

```julia
ψ  = zero_state(6)
ψr = with_policy(ψ, BondRefreshPolicy())     # 浅引用换策略标记
```

`with_policy` **不复制**底层 data / frame，只是换策略标签；之后对 `ψr` 的 `apply!` /
`fold_frame!` / `measure!` 都走新策略。注意同一个底层态不能同时用两个策略演化
（会互相污染），需要各演一份时请先 `copy(ψ)`。

## 2. fold_frame! 的实现（两策略共享框架）

`fold_frame!(ψ; trunc)`（[`src/camps.jl`](../../src/camps.jl)）的核心：

```julia
function fold_frame!(ψ::CAMPS; trunc=DefaultTruncation)
    err = 0.0
    refresh = ψ.policy isa BondRefreshPolicy     # 唯一分叉点
    newgates = LocalOp{ComplexF64}[]
    for g in copy(ψ.frame.gates)                  # 按作用顺序 g₁, g₂, …（内→外）
        e, C = apply_matrix!(ψ.data, g.mat, g.pos;
                             trunc=trunc, search=refresh)
        err = max(err, e)
        if refresh && C !== nothing && length(g) == 2
            push!(newgates, LocalOp(sort(g.pos), adjoint(C)))
        end
    end
    empty!(ψ.frame.gates)
    refresh && append!(ψ.frame.gates, newgates)
    return err                                     # 最大单键截断误差
end
```

要点：

* 对帧门列表的**一个快照**迭代（`copy`），因为循环中会 `empty!` 帧；
* 帧门**按作用顺序内→外**逐个压进 MPS：`g₁` 最先作用，因此先折 `g₁`
  （紧贴数据），最后折最外层 `g_m`，复合结果就是 `U_frame|φ⟩`；
* 每个门经 `apply_matrix!`（Hastings 式，见
  [gates.md](gates.md))落到数据上，可能逐键截断；
* `err` 取所有门截断误差的最大值返回，供调用方参考；
* **两策略唯一区别**在 `search` 关键字与“折完往帧里放什么”：
  * Fold → `search=false`，普通截断，折完帧**清空**；
  * Refresh → `search=true`，每个两比特门折入时若发生截断，尝试选一个
    Clifford 解纠缠门 `C`，并把 `adjoint(C)` 记回帧。

## 3. FoldFramePolicy：折入即清空（经典做法）

### 实现

```julia
struct FoldFramePolicy <: FramePolicy end   # 无任何字段

# fold_frame! 内：
#   search = false         → apply_matrix! 走普通截断
#   newgates 不收集任何东西
#   empty!(frame.gates)    → 帧清空
```

### 语义

折帧 = 把 `U_frame` 完整地“兑现”到 MPS 上，然后忘掉帧：

```text
FoldFramePolicy:
  折入前: |ψ⟩ = g_m ⋯ g_1 |φ⟩,  帧 = [g₁ … g_m]（Clifford，未触碰 MPS）
  折入中: 逐门 g₁ → g₂ → … → g_m 落到 MPS（可截断）
  折入后: |ψ⟩ ≈ |φ'⟩（纯 MPS，普通截断近似），帧 = ∅
```

* 好处：表示回到“纯 MPS”，后续任何操作（含非 Clifford 门、测量）都能直接进行，
  无需再共轭 / 再折叠；
* 代价：**累积的稳定子纠缠一次性全部进入 MPS**。若这一坨 Clifford 纠缠本来很“深”
  （如多层 Clifford ladder），它会瞬间撑满键维并触发截断，丢掉 CAMPS 想保留的结构。
  这正是随机 / 高 magic 线路里 Fold 相对 MPS 无优势的根源（见 [tests.md](tests.md)）。

## 4. BondRefreshPolicy：折入 + 逐键重选解纠缠门

### 思路

折入一个两比特 Clifford 门 `g` 时，若按 `trunc` 截断会丢权重，先把门块做一次
**Clifford 解纠缠**：在两物理腿上先作用一个两比特 Clifford `C`（从极小候选集里选），
使 `C∘g` 作用后的 Schmidt 谱衰减更快，**在同一 `D` 下截断丢弃范数²最小**；截断后的
MPS 承载“已被 `C` 部分解纠缠”的内容，而 `adjoint(C)` 放回帧里，待外层随后继续折 /
作用时再被吸收。稳定子结构尽量留在帧外。

### 实现（三个关键部件）

**① 候选集为什么只有 4 个**（[`cliffframe.jl`](../../src/cliffframe.jl)）：

```julia
const DISENTANGLER_CANDIDATES =  # 经 is_clifford 过滤后保留
    [I, CNOT(control=bit1→bit0), iSWAP, SWAP]
```

两比特截断损失（丢弃奇异值范数²）在**单个站点的局域 Clifford 左乘/右乘**下不变，
因此不必搜索全部两比特 Clifford 群，只需覆盖“非局域等价类”的代表元。数值验证这些
矩阵都是 Clifford。iSWAP 与 CNOT 类不等价，故单列；候选还动态过滤掉非 Clifford。

**② 选 C：`_best_disentangle(B, Λ, trunc)`**（[`src/kernels.jl`](../../src/kernels.jl)）：

1. `B` = 门已作用到两站点块上的张量（尚未截断），`Λ` = 左键谱（预条件）；
2. 先按**恒等**求截断丢弃范数²：`besterr = _discard2(B, Λ, keep)`，
   `keep = min(D, Dl·2, 2·Dr)`；若 `besterr == 0`（根本没截断）→ 立即返回
   `(nothing, 0)`，省掉全部候选开销；
3. 逐个候选 `C ≠ I`：`_discard2(C·B, Λ, keep)`；若严格小于当前最优
   （容差 `1e-14`）则更新 `bestC`；发现 `0`（完全解纠缠成乘积态）提前跳出；
4. 返回 `(C, err)`；`C === nothing` 表示恒等已最优。

```text
             ┌─ besterr==0（秩 ≤ keep，无截断）────► 返回 nothing（不插入）
_best_disentangle
             └─ 否则遍历 {CNOT, iSWAP, SWAP}，找使丢弃范数² 最小的 C
```

**③ 用 C 落盘并回填帧**：

* 折帧路径（`fold_frame!`）：`e, C = apply_matrix!(…; search=true)`；
  若 `C ≠ nothing` 且该门是两比特门，则 `push!(newgates, adjoint(C))`，
  折完全部后 `empty!` 帧再整批 `append!`。回填顺序与折入顺序一致
  （`adjoint(C₁)` 在 `adjoint(C₂)` 之前 = 更靠近 MPS），保持嵌套语义；
* 直接魔法路径（见下文第 6 节）：`pushfirst!`
  前插到帧首，同样是最靠近 MPS 的位置；
* `C` 以 `adjoint(C)` 回填：因为折入时真正落到 MPS 的物理作用等价于
  `C∘g`（先解纠缠再截断），帧里必须留 `adjoint(C)` 才能在最终表示
  `|ψ⟩ = U_frame |φ'⟩` 中把“多出的 C”抵消（“解纠缠”的单位逆操作）。

### 语义

```text
BondRefreshPolicy（折入发生截断时）:
  折入前: |ψ⟩ = g_m ⋯ g_1 |φ⟩, 帧 = [g₁ … g_m]
  折入中: 对两比特 g_k，先作用 C_k∘g_k 再截断；记录 adj(C_k)
  折入后: MPS 只承载“解纠缠后”的残差 |φ'⟩；
          帧 = [adj(C₁) … adj(C_m)]（仅截断过的门留 C，其余没有）
```

折完 MPS 的键维通常比 Fold 小（或同样小但误差更小），稳定子结构以少量
`adj(C)` 的形式继续留在帧外，等下一次魔法 / 截断时再被共轭吸收。

## 5. 两种策略的差异对照

| 维度 | FoldFramePolicy（默认） | BondRefreshPolicy |
|---|---|---|
| 折入时 `search` | `false`（普通截断） | `true`（逐键解纠缠搜索） |
| 截断前是否先作用 `C` | 否 | 是（先 `C∘g` 再 SVD） |
| 折完帧 | **恒为空** | **可能残留** `adj(C)` 序列（仅截断过的门） |
| MPS 承载 | 全部稳定子纠缠 + 魔法 | 已被 `C` 解纠缠的残差 + 魔法 |
| 额外开销 | 无 | 每次两比特折入若截断 → ≤3 次额外 SVD 估值 |
| 误差 | 普通截断误差 | ≤ 普通截断误差（`C=I` 也是候选） |
| 后续帧共轭 | 无（帧空） | 需把后续门共轭穿过少量 `adj(C)` |

### 何时**完全一致**

折入过程中**没有发生任何截断**时：`_best_disentangle` 在恒等处即返回
`(nothing, 0)`，两策略都逐门精确折入、帧都清空，得到**同一个 MPS、同一个 fidelity**。
例如“Clifford 主导 + 稀疏 magic + 折入不超键维”的场景（如 §tests Config C：CAMPS
在 `D=2` 即 1.0、MPS 从没被截断），Fold 与 Refresh 的轨迹、最终帧、结果完全相同。

### 何时**不同**

只有当某个两比特门**折入时真的触发截断**、且非平凡 `C`（CNOT / iSWAP / SWAP）
能减少丢弃权重时，Refresh 才会与 Fold 分道扬镳：

* 效果例（放大版 1D chain，n=16，[tests.md](tests.md) Config A）：
  Fold 在 `D=8` 才到 0.99，Refresh 在 `D=4` 已 1.0——折帧截断点重选 Clifford
  有**实际精度收益**；
* 反例（Config B，n=22）：中段 `D=4/8` 时 Fold 反而优于 Refresh
  （0.0104 / 0.291 vs 0.0 / 0.0156），到 `D=16` 两者同时 1.0。说明 Refresh
  **并不单调更优**，其收益与折入时截断的“形状”线路相关；
* 机制上限：两比特跨键 Schmidt 秩 ≤ 2，`D ≥ 2` 时两比特截断通常根本不发生、
  恒等恒最优，两策略自然一致；Refresh 真正露出差异需要 `D = 1`
  （或截断键秩 > D）且跨键态可被 CNOT/iSWAP 类压成近乘积态。
  `policy_refresh_demo`（30 Clifford + 2 magic，`wmax=2` 强制折帧，`D=1`）里，
  Fold 折后 `frame = 0`，Refresh 折后 `frame = 3（CX×3）` 或 `1（CX×1）`，
  而 MPS 键维保持 1——即“魔法以外的东西尽量没进 MPS”。

## 6. `search` 与 policy 的分工（常见误解）

`apply!` 的 `search` 关键字与 `FramePolicy` **是两回事**：

* `search`（默认 `DefaultSearchDisentangler = true`）控制**直接作用非 Clifford
  魔法门时**的截断是否搜索解纠缠门（[gates.md](gates.md))——
  这条路径上两种 policy 的表现可以一模一样：都要么带 `C` 截断，要么不带；
* `FramePolicy` 只控制**折帧（`fold_frame!`）**时 `search` 的值与帧的处置：
  Fold 强制 `search=false` 且清空帧；Refresh 用 `search=true` 并把 `adj(C)` 回填。

因此“FoldFramePolicy 完全不使用解纠缠”是**不准确**的——它只保证**折入时**不做；
平时魔法门直接作用时，只要 `search=true`（默认），Fold 同样会做截断处解纠缠并
`pushfirst!` 回填 `adj(C)`（这是 CAMPS 内核增强，与策略正交）。

## 7. 策略对测量 / 重置的影响

`measure!` / `reset_qubit_zero!` 投影前若帧非空都会先 `fold_frame!(ψ; trunc)`
（见 [measure.md](measure.md))，因此策略也影响非幺正路径：

* Fold：折完帧空 → 数据侧投影即物理投影，随后 `canonicalize!` 恢复规范，语义干净；
* Refresh：折完可能残留 `adj(C)`。若残留门与投影位点不相交，投影仍精确；若相交，
  严格坍缩需把投影共轭进帧（v1 未实现）。**因此含中途测量的线路建议用
  FoldFramePolicy**；Refresh 面向“深 Clifford、无中途测量”的强模拟场景。

## 8. 选型建议

| 场景 | 建议 |
|---|---|
| 一般用途 / 含测量 / 不确定 | `FoldFramePolicy`（默认，简单、语义干净） |
| Clifford-heavy 深线路、折入必截断、想要最小键维 | 试 `BondRefreshPolicy`，按 fidelity(D) 曲线决定 |
| 折入从不截断（`D` 足够大） | 两者等价，用默认即可 |

`tests.md` 给出了 Fold / Refresh / MPSSimulator 三者的 fidelity(D) 扫描对照，
以及每种场景下“谁更小 D 达到同精度”的结论。
