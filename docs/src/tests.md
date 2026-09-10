# 测试与效果对照

## 1. 单元测试

`test/runtests.jl` 顶层 `@testset "CAMPSSimulator"`，逐文件 include：
`clifford.jl` → `draw.jl` → `gates.jl` → `stability.jl` → `measure.jl` → `backend.jl`。

```sh
cd CAMPSSimulator/test
julia --startup-file=no --project=<含 QuantumCircuits 的环境> -e 'using Test; include("runtests.jl")'
```

当前：**651/651 通过（约 25–35 s，单线程）**。覆盖：

* **纯 Clifford 线路零截断**（[clifford.jl](../../test/clifford.jl)）：60 门随机
  Clifford 在极端小键维下 MPS 键维保持 1、与精确态矢 fidelity = 1；GHZ 幅度 /
  期望（含帧内多观测量张量积）；
* **Clifford 判定**：H/S/X/CX/CZ 为真、T 为假、任意全局相位仍识别；
* **帧可视化 / 命名**（[draw.jl](../../test/draw.jl)，46 项）：`frame_gate_name`
  对任意控制端朝向的 CNOT 都识别为 `CX`；`_cnot_ctl` 判定控制端 1/2；
  `draw_frame` 的 `●`/`X`/`│` 同列对齐、两种朝向、跨线 CNOT 贯通、
  SWAP/CZ 记号、golden 逐字排布；`frame_stats` 计数；
* **门作用与截断**（[gates.jl](../../test/gates.jl)）：Clifford+T / RX / CRZ
  精确演化、一般多比特门（含 Toffoli 任意位置）、SWAP 聚集 / 散开、截断方案；
* **规范与键谱保持**（[stability.jl](../../test/stability.jl)）：Hastings 内核
  演化后 `iscanonical` 保持、`svectors` 即真实 Schmidt 谱、随机线路截断不破坏
  规范 / 范数语义；
* **测量 / 坍缩 / 幅度 / 期望**（[measure.jl](../../test/measure.jl)）；
* **后端集成**（[backend.jl](../../test/backend.jl)）：`CAMPSBackend` 强模拟 /
  逐 shot 采样与直接作用结果交叉一致。

> 测试环境为本地沙箱：`JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH=/tmp/jdepot:…`
> 单线程运行；退出码 `1` 可能来自沙箱对 `/proc/.../mem` 的 JIT 审计噪音，以
> 最后一个 `Test Summary` 为准。

## 2. 效果对照（CAMPS vs MPSSimulator）

完整脚本、记录与图见 `performance/`（**已在 `.gitignore` 中，不随仓库分发**；
脚本仍留在本地可复跑）。方法学：

* 公平性判据**不是 wall-clock，而是“用更小键维达到与 MPS 相同精度”**；
* 参考态 = 两引擎各自 `D=128` 的高 D 收敛态，并打印二者互 fidelity 以标定参考可信度；
* 报告 fidelity(D) 相对该参考的值，以及达到 0.9/0.99 的最小 D。

### 2.1 Clifford 主导 / 1D chain（CAMPS 主场）

线路结构（`performance/rqc/bench_chain_scaling.jl`）：每模块 = `1×T@随机点`
（稀疏 magic）+ `ncliff` 层 Clifford ladder（每 qubit 40% 随机单比特 Clifford +
每条相邻边一个 CX）。

**Config A：n=16，L=2×3 层，seed=1（127 门）**

| D | Fold | Refresh | MPS |
|---|---|---|---|
| 2 | 0.0625 | 7.2e-5 | 4.9e-5 |
| 4 | 0.257 | **1.0** | 2.5e-5 |
| 8 | 1.0 | 1.0 | 0.021 |
| 16 | 1.0 | 1.0 | 0.25 |
| 32 | 1.0 | 1.0 | 1.0 |
| 达 0.99 最小 D | 8 | **4** | 32 |

→ CAMPS 只需 D=4（Refresh）/ D=8（Fold），MPS 需 D=32（4~8× 键维优势）；
Refresh 在 D=4 即精确——**折帧截断点重选 Clifford 有实际精度收益**
（[policies.md](policies.md)）。

**Config B：n=22，L=3×2 层，seed=3（185 门）** — 达 0.99 的最小 D：
Fold / Refresh = 16，MPS = 32（2× 优势）。中段（D=4/8）Fold 反而优于 Refresh
（0.0104/0.291 vs 0.0/0.0156），说明 Refresh **并非单调更优**。

**Config C：n=22，L=2×3 层，seed=5（177 门，推荐）**

| D | Fold | Refresh | MPS |
|---|---|---|---|
| 2 | **1.0** | **1.0** | 0.0 |
| 4 | 1.0 | 1.0 | 0.0 |
| 8 | 1.0 | 1.0 | 4e-6 |
| 16 | 1.0 | 1.0 | 9.3e-4 |
| 32 | 1.0 | 1.0 | 0.083 |
| 达 0.99 最小 D | 2 | 2 | ≥128 |

→ **CAMPS 在 D=2 已完全精确**（帧承载全部 175 个 Clifford 门，MPS 从未被截断），
而 MPS 需 D≥128 → **≥64× 键维优势**。这是 CAMPS 相对普通 MPS 的“效果兑现点”。

**帧图**（Config C 末端帧前 24 门，`draw_frame`；两 policy 帧完全相同——
无截断折入时二者等价）：

```
 q[1]: ─H───────────────────●───────────────────
                            │
 q[2]: ────√X───────────────X──●────────────────
                               │
 q[3]: ────────────────────────X──●─────────────
                                  │
 q[4]: ───────────────────────────X──●──────────
                                     │
   ⋮  （●→X 斜阶梯 = ladder 相邻 CNOT，全部被帧吸收，MPS 键维保持 1）
```

`frame_stats: CX×126, S×24, √X×15, H×10`。

### 2.2 两种 policy 的分化演示

`performance/rqc/policy_refresh_demo.jl`：30 门相邻 Clifford + 2 个 T，
`wmax=2` 强制折帧、`D=1` 使折入必截断：

| seed | Fold 折后帧 | Refresh 折后帧 | MPS 键维 |
|---|---|---|---|
| 1 | 0（清空） | **CX×3** | 1 |
| 2 | 0 | 0（恒等已最优） | 1 |
| 3 | 0 | **CX×1** | 1 |

→ Refresh 折入把解纠缠门留在帧外、MPS 只承载魔法残差；
Fold 折入则把这些 Clifford 结构全部压进 MPS（D=1 必截断）。

### 2.3 Shor 算法（两种 QFT）

* **SemiclassicalQFT**（逐比特测量 + 经典反馈）：测量频繁打断帧，CAMPS 无键维
  优势，N=15/21 各后端分解结果一致且几乎瞬时；
* **LNNQFT**（Fowler 近邻 + 完整 QPE，无中途测量）：正确性一致
  （N=15/21 → (5,3)/(3,5)/(7,3)）。真正走完整求阶的 N=15 seed=2：
  VQC ≈ 51.6 s、MPSSimulator(D=8) ≈ 16.3 s、CAMPS(D=4/8) ≈ 121–133 s——
  小规模键维需求仅个位数，CAMPS 无更小 D 优势且更慢（约 8×）。
* 帧行为验证：Shor 求阶电路首段（SemiclassicalQFT 到第一个测量前）
  3 次 Clifford 全进帧、0 次折叠、MPS 键维保持 1——证明帧确实在工作。

### 2.4 结论（线路相关）

1. **优势场景**：Clifford 主导 + 稀疏 magic + 较深 Clifford ladder
   （§2.1）→ ≥64× 键维优势；差异取决于“Clifford 纠缠能否在 magic 之间持续留在帧外”；
2. **无优势场景**：随机 / 浅层（Sycamore 网格）高 magic、Shor 两种 QFT
   → fidelity(D) 曲线与 MPS 纠缠，甚至更差；
3. **BondRefresh 是真实机制**，但触发面窄：只在“折入发生截断且非恒等 C 减少
   丢弃权重”时与 Fold 分化（见 [policies.md](policies.md)）；
4. 引擎 wall-clock 未达 MPSSimulator 水准（每门共轭 / 折帧开销），局部规范维持
   是后续优化主线（见 [gates.md](gates.md)）。

## 3. 遗留 / 未来工作

* 把解纠缠重选从“两比特截断点”推广到 SWAP / 多键块内部截断点，扩大 Refresh 收益面；
* 中途测量 × Refresh 的严格坍缩（把投影共轭进帧，见
  [backend.md](backend.md)）；
* 引擎性能：对齐 MPSSimulator 的局部规范维持 / 避免每门归一的轻量化。
