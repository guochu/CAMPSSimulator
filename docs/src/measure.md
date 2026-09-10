# 测量、期望与态查询

概率、期望、幅度全部**在帧内精确**计算：先把观测量共轭进帧
（`U_frame† O U_frame`），得到“Pauli 弦 + 符号”的精确表示，再作用到 MPS 副本上收缩。
只有**需要非幺正落盘**（测量坍缩 / 相干重置）时才折叠帧。

## 1. 单比特边缘概率与测量

**概率**（不折叠、不破坏态）：`marginal_probabilities(ψ, q)` 返回 `[p₀, p₁]`。

```julia
q = 3
p0, p1 = marginal_probabilities(ψ, q)
```

算法：把计算基投影写作 Pauli 展开 `|b⟩⟨b| = (I + (-1)^b Z)/2`，再用
`conj_pauli_frame(frame, q, 3)` 把 `Z_q` 共轭进帧，得到
`U† Z_q U = sign · ⊗_s σ_{p(s)}`（`sign = ±1`，`p(s) ∈ {X,Y,Z}`）的**精确符号表示**
（无窗上限，逐帧门按 Pauli 传播规则化简）。最后对 MPS 副本逐站点作用单比特 σ 再
与 bra 收缩得 `⟨Z_q⟩`，`p₀ = (1+⟨Z⟩)/2`。

**测量坍缩**：`measure!(ψ, q; trunc)` 就地返回比特 `0/1`：

1. 计算 `[p₀,p₁]` 并按它们采样（结果概率精确）；
2. 若帧非空 → `fold_frame!(ψ; trunc)`（按当前 FramePolicy，见
   [policies.md](policies.md)）；
3. 数据侧作用投影 `P_b = |b⟩⟨b|`（非幺正）；
4. `canonicalize!(mps(ψ); trunc=NoTruncation(), normalize=true)` 恢复右正则与键谱。

多比特测量与整链测量：

```julia
measure!(ψ, [1, 3])      # 依次测量 qubit 1、3
measure!(ψ)              # 全链测量，返回编码为 Int 的结果（bit (q-1)）
```

非就地协议（QuantumCircuits `measure`）：`QuantumCircuits.measure(ψ, q; trunc)`
返回 `(out, b, p_b)`，其中 `out = copy(ψ)`。

## 2. 期望值

* **单比特**：`expectation(ψ, m, q)`，`m` 为 2×2 Hermitian。分解为
  `m = c₀I + Σ cₚσₚ`，每项经帧共轭得 Pauli 弦后作用 MPS 副本收缩——不构造满矩阵；
* **多比特张量积**：`expectation(ψ, ops)`，`ops` 为互异站点上的
  `q => 2×2` 对列表（至多 6 个站点）。逐项 Pauli 展开、对每组合把各共轭弦顺序
  作用到同一份 MPS 副本上再与 bra 收缩；站点互异，任何对易 / 重叠情况都精确。

## 3. 幅度与态矢物化

* `amplitude(ψ, bits)`：计算基 `|bits⟩` 的振幅。构造 `|ω⟩ = onehot(bits)`，
  把帧的**逆序伴随门**（`reverse(frame.gates)`，每门 `adjoint(g)`）作用上去得
  `U†|bits⟩`，再 `conj(dot(mps(ψ), ω)) = ⟨bits|U|φ⟩*`；
* `statevector(ψ)`：先把 MPS 收缩成态矢，再**按顺序作用全部帧门**
  （站点 1 = LSB）。仅限小比特数（`n ≤ 24`），用于小系统交叉验证。

```julia
a000 = amplitude(ψ, [0, 0, 0])
v    = statevector(ψ)          # 2^n 维向量
```

## 4. 归一化

`normalize!(ψ)` 就地归一到单位 2-范数（帧为酉，只需缩放 MPS）。
`norm(ψ) == norm(mps(ψ))`。

## 5. 帧可视化

给帧“拍照”的三个辅助（[`src/cliffframe.jl`](../../src/cliffframe.jl)，均有单元测试）：

* `draw_frame([io=stdout], frame[, n])` —— ASCII/Unicode 线路图，画风严格对齐
  `QuantumCircuits.draw`：每比特一根横线（前缀 `q[i]: `、水平段 `─`），横轴一列 =
  一帧门（`gates[1]` 最先作用）；两比特门横跨的相邻比特行之间插入连接行画 `│`；
  CNOT 控制端 `●` / 目标端 `X`（任意控制端朝向都能画对）、SWAP/iSWAP 两端 `✕`、
  CZ 为 `●`/`Z`，识别不出的门显示 `?`。不传 `n` 时按帧内最大比特推断：

  ```
  q[1]: ─H──●────
            │    
  q[2]: ────X──S─
  ```

* `frame_stats(frame)` —— `Dict{String,Int}`，各门类型计数 + `"total"`；
* `frame_gate_name(op)` —— 给单个帧门命名（两比特匹配方向不变：同时检查
  `m` 与位交换后的 `SWAP·m·SWAP`）。

用途：调试“Clifford 到底进了多少帧”、对比不同策略留下的帧（
[policies.md](policies.md)）、把帧画进文档 / 论文插图。
