# 门作用内核

本文档描述任意 `k` 比特门如何作用到 CAMPS 态的 MPS 部分上，并保持**右正则性 + 键谱**
不变（Hastings 式更新）。核心代码在 [`src/kernels.jl`](../../src/kernels.jl) 与
[`src/mps.jl`](../../src/mps.jl)。

## 0. 矩阵约定（很重要）

* 所有用户侧接口（`apply!` / `simulate` / 后端）接受 **QuantumCircuits 约定**的门矩阵：
  `positions` 的**第一个比特为矩阵最高位**（MSB-first），比特 1-based；
* 内部（`LocalOp`、MPS 块更新）统一用 **内部小端序**：`pos` 升序、bit0 = 最小站点，
  矩阵按该约定存储。入口处 `_qc_to_internal` 做一次重排（[`src/cliffframe.jl`](../../src/cliffframe.jl)）。

> 这一点是历史 bug（“U2 无法识别”）的根源：两比特匹配若按未交换的模板做，控制端
> 落在内部高位时会被误判。`frame_gate_name` 因此对每个两比特模板同时检查
> `m` 与位交换后的 `SWAP·m·SWAP`。

## 1. 作用路径总览（CAMPS）

```julia
apply!(ψ::CAMPS, m, positions; trunc, search, wmax)
```

1. 若 `k ≤ 2` 且 `is_clifford(m)` → **进帧**：`push!(frame.gates, LocalOp(pos, m_internal))`，
   MPS 完全不动（零截断）。`is_clifford` 只对 1/2 比特判定（>2 一律走稠密路径）；
2. 否则走非 Clifford 路径 `_apply_nonclifford!`：
   * 帧非空时先共轭 `op_eff = frame† op frame`（见第 7 节）；
   * 若共轭支撑超过 `wmax` → 先 [`fold_frame!`](policies.md)（按当前策略折）再重试；
   * `op_eff` 为单比特 → `_apply_single!`（直接作用物理腿，无截断）；
   * 否则 `apply_matrix!`（Hastings 截断，可选解纠缠搜索，见第 6 节），
     并在截断发生时把选中的 `adjoint(C)` 前插回帧。

## 2. 单比特门

`_apply_single!(psi, m, a)`：把 2×2 矩阵作用到站点 `a` 的物理腿上。不改变右正则性、
不改变任何键的 Schmidt 谱（单比特局部酉在左右键上都是等距），因此无截断、无谱更新。

## 3. 两比特门：Hastings 逐键谱预条件 SVD

以相邻站点 `a, a+1` 为例（小端序、bit0 = 站点 `a`）。MPS 以“右正则张量 + `svectors`”
存储，键 `a` 的谱 `Λ = svectors[a]` 是**预条件子**：

1. 收缩两站点为块 `B = G·Θ`（Θ 维数 `(Dl, 4, Dr)`），即门先作用到物理腿；
2. 对键 `a+1`（切在 `a` 与 `a+1` 之间）做**谱加权 SVD**：
   `W = Λ .* B` 重排成 `(Dl·2) × (2·Dr)` 后 `svd`——由于张量右正则，
   `W` 的奇异值就是该 bipartition 的**真实 Schmidt 奇异值**；
3. 按 `trunc` 确定保留数 `r`（至少 1），截断误差 = 丢弃奇异值平方和；
4. **Hastings 左重构** `Ah = B·v†`（未加权块 × 截断后的 `Vt`），避免除以小奇异值；
   站点 `a` 写回 `Ah`，站点 `a+1` 写回 `v`，新谱写入 `svectors[a+1]`。

```text
                    ┌──────────────┐        (Dl·2)×(2·Dr)
  |φ⟩ 右正则        │ Λ .* B       │──svd──►  U S Vt ──按 trunc 取前 r──► v = Vt[1:r,:]
  B = G(Θ(a,a+1))   └──────────────┘
  Ah = B · v†   →   站点 a（左张量，保持右正则）
  v          →   站点 a+1（右张量）
  svectors[a+1] = S[1:r]
```

`k > 2` 的相邻块走 `_apply_block!`：门作用到 `k` 站点窗口后，**块内自左向右**对
`k-1` 个键逐一做上述谱预条件 SVD（每步把当前键谱作预条件、更新张量与谱），一次
sweep 完成。两步都只改窗口内的键谱，规范（右正则）由算法本身维持。

## 4. SWAP 聚集与散开（任意位置的任意比特门）

MPS 内核只支持把门作用到**链上相邻连续块**。若 `positions`（升序）不连续：

1. **`_gather!`**：对每个参与比特自左向右逐个安放——第 `k` 个参与比特（原在
   `positions[k]`）向左相邻 SWAP 直到目标槽 `t = positions[1]+k-1`；每次相邻交换
   记录为 `(s-1) => s` 存入 `swaps`；
2. 在连续块 `[a, a+k-1]` 上作用门（见第 3 节）；
3. **`_scatter!`**：按 `swaps` 的**逆序**回放相邻交换，把被挤开的比特送回原位，
   恢复“站点 k ↔ qubit k”的一一对应。

```text
  例：CZ(q1, q6)，n = 6（positions 升序 [1, 6]，目标块 [1, 2]）
  位置:   1   2   3   4   5   6
  gather: q1  q2  q3  q4  q5  q6
     交换(5,6) → q1 q2 q3 q4 q6 q5
     交换(4,5) → q1 q2 q3 q6 q4 q5
     交换(3,4) → q1 q2 q6 q3 q4 q5
     交换(2,3) → q1 q6 q2 q3 q4 q5     ← q1、q6 相邻于块 [1,2]，可作用 CZ
  scatter: 逆序回放 (2,3)(3,4)(4,5)(5,6) → 恢复站点↔比特对应
```

注意：`positions` 传入时升序，聚集**不改变参与比特的相对顺序**——第 `k` 个参与比特
最终落在 `positions[1]+k-1`，因此**聚集后块内自左向右顺序 = 升序 positions = 门矩阵
的内部小端序位序**，门可直接作用、无需额外重排。

**为什么 `_scatter!` 必须逆序回放**：`swaps` 是按“先把较左的参与比特放到位、再处理更右
的参与比特”压栈的；若正序重放，前面已安放的比特会被后面的移动再次推走。相邻 SWAP
自逆，逆序重放恰好是 gather 置换的逆置换，可整体撤销聚集。

每次相邻 SWAP 本质上是一个两比特 SWAP 酉门（`_apply_two!`），在 `trunc` 非空时可能
截断——`_gather!` / `_scatter!` 返回累计误差，供 `apply_matrix!` 判断是否归一化。

## 5. apply_matrix! 的收尾：只归一化，不整链规范化

```julia
function apply_matrix!(psi, m, positions_sorted; trunc, search)
    # … gather（若非相邻）→ 作用（2 体可选搜索 / k 体块）→ scatter …
    if max(swaperr, scattererr, err) > 0          # 任一环节截断过
        nrm = norm(psi)
        nrm > 0 && _scale!(psi, inv(nrm))         # 轻量整体归一化
    end
    return err, C
end
```

* Hastings 更新**天然保持规范**（右正则 + 键谱正确），因此**不在每个门之后都
  `canonicalize!`**；
* 截断会丢弃权重使范数 < 1，这里只做一次**轻量归一化**（缩放首张量与键谱，保持概率/
  期望语义）；
* 若截断过大导致精度问题，那是 `trunc` 参数设置问题，不应由内核用额外的整链规范化
  掩盖。需要彻底恢复规范时，用户可随时 `canonicalize!(mps(ψ); …)`。

## 6. 截断处解纠缠搜索（CAMPS 增强）

`apply!` 的 `search=true`（默认开启，`DefaultSearchDisentangler`）且截断方案为
`TruncateDim` 时，相邻两比特门在截断前先做一次候选评估（`_apply_contiguous!`）：

1. 先算门作用后的块 `B`（未截断），按恒等评估丢弃范数²；
2. `_best_disentangle`：若恒等即零丢弃（无截断 / 秩 ≤ keep）→ 直接普通截断，零开销；
3. 否则从候选集 `{CNOT(bit1→bit0), iSWAP, SWAP}` 中找使丢弃范数² **最小**的 `C`
   （见 [policies.md](policies.md) 的候选集
   理论依据）；
4. 命中非恒等 `C` 时走 `_apply_two_c!`（作用 `C∘G` 后按规范两体更新截断），
   并把 `adjoint(C)` 回填进帧（CAMPS 层 `pushfirst!`）。

由于候选只含 Clifford 且以恒等为基线，**误差永不超过普通截断**——最坏情形就是退回
普通截断。

## 7. 共轭窗与折叠时机

`conj_through_frame(frame, op; wmax)` 把“位于帧顶层之上”的算子 `op` 推过帧到 MPS
侧，返回 `frame† op frame`：

* 从 `frame.gates` 的**末尾（最顶层）向首（最靠近 MPS）**逐个共轭；只有与当前算子
  支撑相交的帧门才需要处理（`_conj_by`：不相交直接返回原算子；相交则嵌入到并集支撑
  后算 `H†·op·H`）；
* 算子支撑 = 原支撑 ∪ 各相交帧门支撑，会膨胀；一旦 `length(cur) > wmax`
  （默认 8）立即返回 `nothing`——继续共轭不如折叠。

折叠触发后，`_apply_nonclifford!` 递归：先 `fold_frame!`（按当前 FramePolicy，
见 [policies.md](policies.md))，再用空帧重试共轭（此时必然成功，因为帧已空）。

```text
非 Clifford 门 g 到达
   │
   ├─ 帧空 ────────────────► 直接作用到 MPS（截断 / 搜索）
   │
   └─ 帧非空
        │ conj_through_frame（逐门共轭，支撑膨胀）
        ├─ 支撑 ≤ wmax ────► 作用 conj 后的算子
        └─ 支撑 > wmax ────► fold_frame!(按策略) → 重试
```

## 8. 截断方案与默认值

`TruncationScheme` 与 MPSSimulator 同名同语义（re-export 自
`FiniteMPSAlgorithms`，见 [`src/tensorops.jl`](../../src/tensorops.jl)）：

| 类型 / 构造 | 语义 |
|---|---|
| `NoTruncation()` | 不截断，保留全部奇异值 |
| `truncdim(D)` / `TruncateDim` | 只保留最大的 `D` 个 |
| `truncrelerr(ϵ=…)` / `TruncateRelError` | 丢弃相对范数² 低于 `ϵ²` 的奇异值 |
| `truncdimcutoff(D, ϵ)` / `TruncateDimCutoff` | 先按 `ϵ` 定截断点、再封顶 `D`、至少留 `add_back` |

默认值集中定义于 [`src/defaults.jl`](../../src/defaults.jl)：

* `DefaultTruncation = truncdimcutoff(50, 1e-8)` —— 所有作用型接口的 `trunc` 默认；
* `DefaultConjWindow = 8` —— 帧共轭最大支撑窗；
* `DefaultSearchDisentangler = true` —— 截断处解纠缠搜索默认开启。

## 9. 保持键谱正确的两条保证

1. **门路径**：单比特门不动谱；多比特 Hastings 更新把每一步 SVD 得到的真实奇异值
   写回 `svectors`，右正则由算法维持 → `apply!` 全程 `iscanonical(mps(ψ))` 为真；
2. **非幺正路径**（测量投影 / 重置）可能破坏右正则 / 谱，接口内部随后调用
   `canonicalize!(mps(ψ); trunc=NoTruncation(), normalize=true)` 恢复（精确，不截断）。
