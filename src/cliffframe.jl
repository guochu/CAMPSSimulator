# cliffframe.jl — Clifford 帧：局域 Clifford 门、Clifford 判定、共轭、
# Pauli 弦传播与解纠缠候选门
#
# 约定（全包统一）：
#   * 局域算子 `LocalOp(pos, mat)`：`pos` 升序，矩阵 `mat` 小端序
#     （bit0 ↔ 最小站点 = LSB），大小 2^|pos| × 2^|pos|；
#   * `CliffordFrame.gates` 列表顺序 = 作用顺序：`gates[1]` 最先作用
#     （最靠近 MPS）。态 = `U|φ⟩`，`U = gates[end]⋯gates[1]`；
#   * 入站 Clifford 门追加到列表末尾（作用在顶层）；
#   * 截断处选中的解纠缠门 C 以 `adjoint(C)` 前插（作用在 MPS 侧）。

using LinearAlgebra

const _PAULI_I = ComplexF64[1 0; 0 1]
const _PAULI_X = ComplexF64[0 1; 1 0]
const _PAULI_Y = ComplexF64[0 -im; im 0]
const _PAULI_Z = ComplexF64[1 0; 0 -1]
const _PAULI1 = [_PAULI_I, _PAULI_X, _PAULI_Y, _PAULI_Z]

"""
    LocalOp

内部局域算子：`pos`（升序）与矩阵（小端序，bit0=最小站点）。帧门与
共轭中的稠密算子都用该类型表示。
"""
struct LocalOp{T<:Number}
    pos::Vector{Int}
    mat::Matrix{T}

    function LocalOp{T}(pos::AbstractVector{Int}, mat::AbstractMatrix) where {T<:Number}
        new{T}(sort(collect(Int, pos)), convert(Matrix{T}, mat))
    end
end
LocalOp(pos::AbstractVector{Int}, mat::AbstractMatrix{T}) where {T<:Number} =
    LocalOp{T}(pos, mat)
Base.length(op::LocalOp) = length(op.pos)
Base.:(==)(a::LocalOp, b::LocalOp) = a.pos == b.pos && a.mat == b.mat

"""
    CliffordFrame

Clifford 帧：有序局域 Clifford 门列表（`gates[1]` 最先作用）。
帧为空表示作用在 MPS 上的 Clifford 部分为单位元。
"""
mutable struct CliffordFrame{T<:Number}
    gates::Vector{LocalOp{T}}
end
CliffordFrame{T}() where {T<:Number} = CliffordFrame{T}(LocalOp{T}[])
CliffordFrame() = CliffordFrame{ComplexF64}()
Base.length(f::CliffordFrame) = length(f.gates)
Base.isempty(f::CliffordFrame) = isempty(f.gates)

# ── 帧可视化 / 统计（仿 QuantumCircuits.draw 的 ASCII 风格）─────────────────

# 常用门表：用 |tr(U†V)| ≈ 维数判同名（忽略全局相位）
const _KNOWN1 = [
    ("H",   ComplexF64[1 1; 1 -1] ./ sqrt(2)),
    ("S",   ComplexF64[1 0; 0 im]),
    ("S†",  ComplexF64[1 0; 0 -im]),
    ("T",   ComplexF64[1 0; 0 exp(im * pi / 4)]),
    ("T†",  ComplexF64[1 0; 0 exp(-im * pi / 4)]),
    ("X",   ComplexF64[0 1; 1 0]),
    ("Y",   ComplexF64[0 -im; im 0]),
    ("Z",   ComplexF64[1 0; 0 -1]),
    ("√X",  (ComplexF64[1 0; 0 1] - im * ComplexF64[0 1; 1 0]) / sqrt(2)),
    ("√Y",  (ComplexF64[1 0; 0 1] - im * ComplexF64[0 -im; im 0]) / sqrt(2)),
    ("I",   ComplexF64[1 0; 0 1]),
]
const _KNOWN2 = [
    ("CX",  ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]),
    ("CZ",  ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]),
    ("SWAP",ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]),
    ("iSWAP",ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]),
]

const _SWAP2 = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

"""
    frame_gate_name(op) -> String

给帧门起个可读名字（H/S/CX/CZ/SWAP/iSWAP/√X…；识别不出用 `?`）。
两比特匹配是**方向不变**的：同时检查 `m` 与 `SWAP·m·SWAP`（位交换），
因为内部门一律按“站点升序 + 小端序”存储，控制端可能落在任一 qubit。
"""
function frame_gate_name(op::LocalOp)
    k = length(op)
    m = op.mat
    if k == 1
        for (nm, g) in _KNOWN1
            size(g) == size(m) || continue
            abs(tr(g' * m)) > 2 - 1e-6 && return nm
        end
        return "?"
    elseif k == 2
        for (nm, g) in _KNOWN2
            abs(tr(g' * m)) > 4 - 1e-6 && return nm
            abs(tr((_SWAP2 * g * _SWAP2)' * m)) > 4 - 1e-6 && return nm  # 位交换等价
        end
        return "?"
    end
    return "?$k"
end

"CNOT 控制端判断：内部小端两比特位 (p1<p2, p1=bit0)。返回控制端 1/2；非 CNOT 返回 0。"
function _cnot_ctl(m::AbstractMatrix)
    size(m) == (4, 4) || return 0
    ctl_p1 = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]   # 控制=bit0(=p1)
    ctl_p2 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]   # 控制=bit1(=p2)
    abs(tr(ctl_p1' * m)) > 4 - 1e-6 && return 1
    abs(tr(ctl_p2' * m)) > 4 - 1e-6 && return 2
    return 0
end

"""
    frame_stats(frame) -> Dict{String,Int}

帧内门类型计数与总数（键 `:total`）。
"""
function frame_stats(fr::CliffordFrame)
    d = Dict{String,Int}()
    for g in fr.gates
        nm = frame_gate_name(g)
        d[nm] = get(d, nm, 0) + 1
    end
    d["total"] = length(fr.gates)
    return d
end

# 一列 = 一个门：列内记录各线上要画的符号（cells: wire → text）与该门
# 横跨的线范围 [lo, hi]。渲染模型与 QuantumCircuits.draw 一致：比特行与行
# 之间按需插入“连接行”（画跨线门的纵向 `│`）。
struct _FrameCol
    cells::Dict{Int,String}
    lo::Int
    hi::Int
end

function _frame_col(g::LocalOp)
    lo, hi = extrema(g.pos)
    cells = Dict{Int,String}()
    if length(g) == 1
        cells[g.pos[1]] = frame_gate_name(g)
    elseif length(g) == 2
        nm = frame_gate_name(g)
        ctl = _cnot_ctl(g.mat)
        if ctl > 0
            # CNOT(CX)：控制端 ●、目标端 X（同 QuantumCircuits 的 CX 画法）
            cells[g.pos[ctl]] = "●"
            cells[g.pos[3-ctl]] = "X"
        elseif nm == "CZ"
            cells[g.pos[1]] = "●"      # CZ 对称：按位序记控制 ● / 标记 Z
            cells[g.pos[2]] = "Z"
        elseif nm == "SWAP" || nm == "iSWAP"
            cells[g.pos[1]] = "✕"      # 同 QuantumCircuits：两端画 ✕
            cells[g.pos[2]] = "✕"
        else
            cells[g.pos[1]] = nm       # 其它两比特门：两端标同名
            cells[g.pos[2]] = nm
        end
    else
        cells[g.pos[1]] = frame_gate_name(g)   # 多比特：名字标首线，其余连线
    end
    return _FrameCol(cells, lo, hi)
end

"""
    draw_frame([io=stdout], frame[, n])

以 ASCII 线路图打印 Clifford 帧，画风严格对齐 QuantumCircuits.draw：

```
q[1]: ─H──●────
          │    
q[2]: ────X──S─
```

上例 = `H@q1 → CX(控制=q1, 目标=q2) → S@q2`：CNOT 控制端 `●` 与目标端 `X`
之间有纵向连接线 `│`。规则：

* 每比特一根横线，前缀 `q[i]: `，横线用 `─` 填充，横轴一列 = 一帧门
  （`gates[1]` 最先作用）；
* 两比特门横跨的相邻比特行之间插入一行连接线（该列画 `│`）；
* CNOT 控制端 `●`、目标端 `X`（SWAP/iSWAP 两端 `✕`，CZ 为 `●`/`Z`，
  其余两比特门两端标同名；识别不出的门显示 `?`）。

不传 `n` 时按帧内出现的最大比特推断。
"""
function draw_frame(io::IO, fr::CliffordFrame, n::Int=maximum(maximum(g.pos) for g in fr.gates; init=0))
    n >= 1 || return print(io, "（空帧）")
    cols = [_frame_col(g) for g in fr.gates]

    # 每列宽 = max(3, 最长符号 textwidth + 2)；符号与连接棒都按列居中
    widths = Int[]
    for col in cols
        w = 3
        for t in values(col.cells)
            w = max(w, textwidth(t) + 2)
        end
        push!(widths, w)
    end

    prefixes = [string("q[", i, "]: ") for i in 1:n]
    maxp = maximum(textwidth, prefixes; init=0)

    _center(s, w, f) = begin
        pad = max(w - textwidth(s), 0)
        l = pad ÷ 2
        repeat(f, l) * s * repeat(f, pad - l)
    end

    FQ = "─"      # 水平线
    VL = "│"      # 跨线门连接棒
    for r in 1:n
        if r > 1
            # 线 r-1 与线 r 之间的连接行：凡横跨此间隙的门列画 `│`，
            # 没有任何跨线门的间隙不输出该行（同 QuantumCircuits）。
            row = String[]
            hasline = false
            for (i, col) in enumerate(cols)
                if col.lo < r <= col.hi
                    push!(row, _center(VL, widths[i], " "))
                    hasline = true
                else
                    push!(row, repeat(" ", widths[i]))
                end
            end
            hasline && (print(io, repeat(" ", maxp)); println(io, join(row)))
        end
        print(io, lpad(prefixes[r], maxp))
        for (i, col) in enumerate(cols)
            cell = get(col.cells, r, nothing)
            if cell === nothing
                print(io, repeat(FQ, widths[i]))
            else
                print(io, _center(cell, widths[i], FQ))
            end
        end
        println(io)
    end
    return nothing
end
draw_frame(fr::CliffordFrame, n::Int=maximum(maximum(g.pos) for g in fr.gates; init=0)) =
    draw_frame(stdout, fr, n)

# ── Clifford 判定（1、2 比特矩阵）──────────────────────────────────────────

function _single_pauli(n::Int, code::Int)
    # code ∈ 0..4^n-1，每个 2 比特位选 I/X/Y/Z；返回张量积
    σ = ComplexF64[1;;]                                  # 1×1
    for t in 0:(n-1)
        σ = kron(_PAULI1[((code >> (2t)) & 0b11) + 1], σ)
    end
    return σ
end

"矩阵是否（至多一个符号 + 全局相位 ±1 意义下）等于单个非平凡 Pauli 弦的镜像。"
function _is_pauli_image(A::AbstractMatrix{<:Number}, k::Int; atol::Real=1e-8)
    d = 1 << k
    (size(A, 1) == size(A, 2) == d) || return false
    # 遍历所有非平凡 Pauli 弦（4^k-1 个）
    for code in 1:(d*d-1)
        σ = _single_pauli(k, code)
        c = real(tr(σ * A)) / d
        if abs2(c - 1) < atol || abs2(c + 1) < atol
            # A ≈ ±σ：检查残差
            r = norm(A .- (c .* σ))
            r < sqrt(d) * atol && return true
        end
    end
    return false
end

"""
    is_clifford(m) -> Bool

判断局域酉矩阵 `m`（1 或 2 比特）是否为 Clifford 门（含任意全局相位）。
3 比特及以上返回 `false`（走一般稠密路径，仍精确）。
"""
function is_clifford(m::AbstractMatrix{<:Number})
    s = size(m, 1)
    s == size(m, 2) || return false
    k = round(Int, log2(s))
    (1 <= k <= 2 && (1 << k) == s) || return false
    d = s
    # 非平凡 Pauli（d²-1 个）的镜像都应为 ±Pauli
    for code in 1:(d*d-1)
        P = _single_pauli(k, code)
        A = m * P * m'
        _is_pauli_image(A, k) || return false
    end
    return true
end

# ── 比特重排：QuantumCircuits 约定 → 内部小端序 ────────────────────────────

"""
    _qc_to_internal(m, positions) -> Matrix

把 QuantumCircuits 约定的门矩阵（`positions[1]` = 最高位，位轴按列表序）
重排为内部小端序（`positions` 升序、最小站点 = LSB）。
"""
function _qc_to_internal(m::AbstractMatrix, positions::AbstractVector{Int})
    k = length(positions)
    d = 1 << k
    (size(m, 1) == size(m, 2) == d) || throw(DimensionMismatch("gate size mismatch"))
    T = promote_type(eltype(m), ComplexF64)
    mc = Matrix{T}(m)
    srt = sortperm(collect(Int, positions); rev=true)
    srt == collect(1:k) && return mc
    perm = (srt..., srt .+ k...)
    return reshape(permutedims(reshape(mc, ntuple(_ -> 2, 2k)...), perm), d, d)
end

# ── 稠密算子共轭 ───────────────────────────────────────────────────────────

"把 `M`（长度 `k` 的 bit 轴，由 `bits` 指定在 m 位空间中的位置，1-based）嵌入 m 位空间。"
function _embed(M::AbstractMatrix{T}, bits::Vector{Int}, m::Int) where {T}
    k = length(bits)
    d = 1 << k
    dim = 1 << m
    E = zeros(promote_type(T, ComplexF64), dim, dim)
    allslots = setdiff(collect(1:m), bits)               # 其它位的槽位（1-based，映射到 bit index-1）
    nfree = length(allslots)
    for o in 0:((1 << nfree) - 1)
        base = 0
        for t in 1:nfree
            ((o >> (t - 1)) & 1) == 1 && (base |= 1 << (allslots[t] - 1))
        end
        for a in 0:(d-1), b in 0:(d-1)
            ia = base
            ib = base
            for t in 0:(k-1)
                ia |= ((a >> t) & 1) << (bits[t+1] - 1)
                ib |= ((b >> t) & 1) << (bits[t+1] - 1)
            end
            E[ia+1, ib+1] = M[a+1, b+1]
        end
    end
    return E
end

"""
    _conj_by(op, H) -> op′

`op ← H† op H`（稠密，支撑扩张到并集）。
"""
function _conj_by(op::LocalOp{T}, H::LocalOp) where {T}
    isempty(intersect(op.pos, H.pos)) && return op
    newpos = sort(unique(vcat(op.pos, H.pos)))
    m = length(newpos)
    idxO = [findfirst(==(p), newpos) for p in op.pos]
    idxH = [findfirst(==(p), newpos) for p in H.pos]
    Oe = _embed(op.mat, idxO, m)
    He = _embed(H.mat, idxH, m)
    return LocalOp(newpos, He' * Oe * He)
end

"""
    conj_through_frame(frame, op; wmax) -> Union{Nothing,LocalOp}

把算子 `op`（作用在物理帧、位于帧顶层之上）推过帧到 MPS 侧：
返回 `frame† op frame`。过程中支撑超过 `wmax` 个站点时返回 `nothing`
（调用方应折叠帧后重试）。
"""
function conj_through_frame(frame::CliffordFrame, op::LocalOp; wmax::Int=DefaultConjWindow)
    cur = op
    for H in reverse(frame.gates)
        isempty(intersect(cur.pos, H.pos)) && continue
        cur = _conj_by(cur, H)
        length(cur) > wmax && return nothing
    end
    return cur
end

# ── CAMPS 截断搜索候选门 ──────────────────────────────────────────────────
#
# 截断损失（丢弃奇异值范数²）在单个站点的局域 Clifford 左乘/右乘下不变，
# 因此只需搜索“非局域类”代表元：I、CNOT(bit0→bit1)、iSWAP、SWAP。
# （iSWAP 类与 CNOT 类不等价；经数值验证均为 Clifford。）

function _gate_cnot()
    G = zeros(ComplexF64, 4, 4)
    # control = bit1（站点 a+1）? CNOT 控制=bit1(高位)，目标=bit0
    for c in 0:1, t in 0:1
        G[t+1+2c, ((t ⊻ c) + 1) + 2c] = 1.0
    end
    return G
end

function _gate_iswap()
    # iSWAP = diag(1, i, i, 1)·SWAP 的酉变体：矩阵 [[1,0,0,0],[0,0,im,0],[0,im,0,0],[0,0,0,1]]
    return ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]
end

const DISENTANGLER_CANDIDATES = begin
    Id = Matrix{ComplexF64}(I, 4, 4)
    cn = _gate_cnot()
    isw = _gate_iswap()
    sw = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    cs = [Id, cn, isw, sw]
    # 数值验证后仅保留 Clifford 候选（恒等 + 至少一个非平凡类）
    cs = filter(c -> is_clifford(c), cs)
    length(cs) >= 2 || error("disentangler candidates degenerated to identity only")
    cs
end
