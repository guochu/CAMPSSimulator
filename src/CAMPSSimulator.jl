"""
    CAMPSSimulator

基于 **Clifford 增广 MPS（CAMPS）** 的通用量子线路模拟后端（与同目录
`MPSSimulator` 平行、接口对齐但内核独立）。

态表示：`|ψ⟩ = U_frame |φ⟩`。其中

* `U_frame`（[`CliffordFrame`](@ref)）——Clifford 帧：所有入站 Clifford 门
  （H / S / CNOT / SWAP …）以精确矩阵形式进帧，**不触碰 MPS、零截断**；
* `|φ⟩`（`MPS`）——承载“魔法（非稳定子）”部分的矩阵乘积态。

核心原语（见 `src/camps.jl`）：

* Clifford 门 → 进帧（O(1) / 门，精确）；
* 非 Clifford 门 → 共轭 `frame† g frame` 后作用到 MPS 并 SVD 截断；
  共轭支撑超过 `wmax` 时自动折叠帧（`fold_frame!`）再作用；
* 两比特截断处可选 **Clifford 解纠缠搜索**（CAMPS 增强）：截断损失在
  局域 Clifford 左乘/右乘下不变，只需搜索非局域类代表元（CNOT / iSWAP /
  SWAP），成本低且误差**永不超过**普通截断。

约定（与 MPSSimulator / QuantumCircuits 一致）：比特 1-based 小端序；
门矩阵 `positions[1]` = 最高位；`apply!` / `simulate` 等作用型接口接受
`trunc`（`TruncationScheme`）、`search`（解纠缠开关）、`wmax`（共轭窗）
关键字。

**v1 边界**：纯态（不含噪声信道 `ChannelOp` / 密度算符）；测量在中途时
先折叠帧再局部投影（概率计算始终在帧内精确）。

核心接口：

* `apply!` / `apply` / `simulate` / `simulate!`（QuantumCircuits 桥接）；
* `measure!` / `marginal_probabilities` / `expectation` / `amplitude` /
  `statevector` / `normalize!` / `fold_frame!`；
* 初始化：`zero_state` / `onehot_state` / `qubit_encoding_state` /
  `rand_state`；
* 后端：`CAMPSBackend`（`QuantumCircuits.simulate(c, backend; shots)`）。
"""
module CAMPSSimulator

using LinearAlgebra
using Random
using QuantumCircuits
using QuantumCircuits: GateOp, ChannelOp, MeasOp, ReinitOp, BarrierOp, IfOp, BlockOp,
                       Operation, Circuit, Cond, Param, ClbitRef, CReg, ParamVector,
                       qubits, mat, kraus, parameters, unroll
import QuantumCircuits: nqubits, measure
using QuantumCircuits.Interface: Backend, SimResult

# ── 截断 / 默认参数 ─────────────────────────────────────────────────────────
include("tensorops.jl")
include("defaults.jl")

# ── 态 ──────────────────────────────────────────────────────────────────────
include("mps.jl")
include("cliffframe.jl")

# ── 内核与 CAMPS 核心操作 ───────────────────────────────────────────────────
include("kernels.jl")
include("camps.jl")
include("initializers.jl")

# ── QuantumCircuits 桥接层 ──────────────────────────────────────────────────
include("quantumcircuits/ops.jl")
include("quantumcircuits/classical.jl")
include("quantumcircuits/simulate.jl")
include("quantumcircuits/backend.jl")

# ── 导出 ────────────────────────────────────────────────────────────────────
export # 截断方案
    TruncationScheme, NoTruncation, TruncateDim, TruncateCutoff, TruncationDimCutoff,
    truncdim, trunccutoff, truncdimcutoff,
    DefaultTruncation, DefaultConjWindow, DefaultSearchDisentangler,
    # 态
    CAMPS,
    zero_state, onehot_state, qubit_encoding_state, rand_state,
    bond_dimensions, max_bond_dimension, mps, frame,
    # 核心操作
    apply, apply!, fold_frame!, measure!, marginal_probabilities,
    expectation, amplitude, statevector, normalize!, reset_qubit_zero!,
    canonicalize!, iscanonical,
    is_clifford,
    # 帧更新策略（可切换的“算法”）
    FramePolicy, FoldFramePolicy, BondRefreshPolicy, DefaultFramePolicy,
    with_policy,
    # Clifford 帧可视化 / 统计
    draw_frame, frame_stats, frame_gate_name,
    # QuantumCircuits.Interface 再导出
    simulate, simulate!,
    # 桥接层
    ClassicalStore, CAMPSBackend

end # module
