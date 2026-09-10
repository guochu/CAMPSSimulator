# QuantumCircuits 桥接层与后端

CAMPSSimulator 实现 `QuantumCircuits.Interface` 的 `Backend` 协议，可直接把
`Circuit` 演化到 `CAMPS` 态，与 MPSSimulator / VQC（稠密后端）共享同一套线路 IR、
可互换使用并交叉验证。

## 1. 指令 → 核心原语

`apply!(state::CAMPS, op, [params]; trunc, search, wmax)` 把 IR 指令就地作用到态上
（[`src/quantumcircuits/ops.jl`](../../src/quantumcircuits/ops.jl)）：

| 指令 | 行为 |
|---|---|
| `GateOp` | `mat(op)` 取矩阵 → `apply!(state, m, qubits(op); …)`（见 [gates.md](gates.md)） |
| `MeasOp` | 对每个被测比特 `measure!`（坍缩，结果丢弃；要保留结果请走 `simulate`） |
| `ReinitOp` | 逐个 `reset_qubit_zero!` |
| `BarrierOp` | 无操作 |
| `BlockOp` | 先 `unroll` 展开再逐条执行 |
| `IfOp` | 需要经典寄存器状态，请通过 `simulate` 执行 |
| `ChannelOp` | **不支持**：v1 是纯态表示（帧 + 纯 MPS），抛明确错误 |

参数绑定：`GateOp` 的符号参数经 `params` 环境（`_ParamEnv`）解析——支持 `Vector`
（按 `parameters(c)` 顺序）、`Dict`（`Symbol` / `Param` / `ParamVector` 键）。

## 2. 线路演化

```julia
simulate(c::Circuit, s::CAMPS; params=nothing,
         trunc=DefaultTruncation, search=true, wmax=8)   # 非就地（copy）
simulate!(c::Circuit, s::CAMPS; ...)                     # 就地
```

演化是“指令顺序推进 + 经典寄存器运行时”：

* `MeasOp` 的测量结果写回 `ClassicalStore`，供后续 `IfOp` 条件执行；
* `IfOp` 从 store 取经典位判断走 then / otherwise 分支；
* `BlockOp` 展开执行；`BarrierOp` 跳过。
* 符号参数线路：`simulate(c, s; params=[0.1, 0.2, …])`。

## 3. CAMPSBackend（Interface 后端）

```julia
CAMPSBackend(; dtype=ComplexF64, trunc=DefaultTruncation,
             search=true, wmax=8) <: Backend
```

* `shots = 0`（默认）：**强模拟**，`SimResult.state` 为 `CAMPS` 终态；
* `shots > 0`：**逐 shot 采样**——每个 shot 都从 `|0…0⟩` 重新演化，测量 /
  经典反馈写回寄存器，结果按 `counts_key` 约定聚合成 `Dict{String,Int}`
  （键如 `"c:00"`）；
* `seed`：强模拟忽略；采样时先 `Random.seed!(seed)`（`measure!` 使用全局 RNG），
  保证同种子同结果；
* 能力声明 `supports(cap)`: `:statevector`、`:mid_measure`；**无 `:noise`**
  （信道指令不被支持）。

```julia
using QuantumCircuits, CAMPSSimulator

c = Circuit(2)
push!(c, H(1)); push!(c, CX(1, 2))
push!(c, measure(1, c.cregs[1][1])); push!(c, measure(2, c.cregs[1][2]))

b = CAMPSBackend(trunc = truncdim(16))
rs = simulate(c, b; shots = 1024, seed = 7)
rs.counts                     # Dict{String,Int}
```

## 4. 接口对齐与互换

| 能力 | MPSSimulator | CAMPSSimulator |
|---|---|---|
| `simulate(c, state)` | `zero_state_mps` 等（MPS） | `zero_state` 等（CAMPS） |
| `simulate(c, backend; shots)` | `MatrixProductBackend` | `CAMPSBackend` |
| `measure` / `measure!` 协议 | 同 | 同（非就地版本返回 `(out, b, p)`） |
| 噪声 / 混合态 | 支持 | 不支持（v1） |
| 特色 | 完整可观测量代数 | Clifford 帧 + 两种 FramePolicy |

`CAMPSBackend` 的能力声明为 `:statevector` 与 `:mid_measure`（无 `:noise`）。
两后端对同一线路应给出（数值上）相同的强模拟结果；测试 / 性能验证正是用
MPSSimulator 作为“普通 MPS”参照来交叉验证（[tests.md](tests.md)）。

## 5. v1 边界

* **无噪声 / 混合态**：`ChannelOp` 抛错；不支持密度算符；
* **中途测量 × BondRefreshPolicy**：测量前按策略折帧；若 Refresh 折入留下
  `adj(C)` 且与投影位点相交，严格坍缩语义需把投影共轭进帧（未实现）。含中途
  测量的线路建议用默认 `FoldFramePolicy`（[policies.md](policies.md)）；
* **`statevector` 物化限 ≤ 24 比特**；
* 多比特期望（张量积观测量）限 ≤ 6 个互异站点。
