# camps.jl — CAMPS 态类型与核心操作
#
# CAMPS 态 = `|ψ⟩ = U_frame |φ⟩`：
#   * `data`：纯态 MPS `|φ⟩`（承载“魔法/非稳定子”部分，键维按 `trunc` 控制）；
#   * `frame`：Clifford 帧（承载全部稳定子纠缠，矩阵形式精确跟踪）。
#
# 语义：一切用户可见操作都作用在物理帧 `|ψ⟩` 上。
#
# * Clifford 门 → 追加进帧（不触碰 MPS，精确、O(1) 每门）；
# * 非 Clifford 门 → 共轭 `frame† g frame` 后作用到 MPS 上并截断
#   （共轭支撑超过 `wmax` 时先折叠帧）；
# * 截断处（两比特窗口、`TruncateDim`、搜索开启）选择最优 Clifford
#   解纠缠门降低截断损失（CAMPS 增强，误差永不超过普通截断）。

# ── 帧更新策略（FramePolicy）：清理帧/处理截断的两种“算法” ─────────────
#
# 当非 Clifford 门经帧共轭后支撑超过 `wmax`，必须把（一部分）帧“折入”
# MPS 才能继续本地化作用。不同策略决定折入方式，影响键维-精度折衷：
#
# * [`FoldFramePolicy`](@ref)（当前默认）：整帧顺序折入、`search=false`
#   普通截断、折入后帧清空 —— 实现简单，但会把累积的稳定子纠缠全部
#   交给 MPS，容易丢掉 CAMPS 想保留的结构（随机/深 Clifford 场景劣势）；
# * [`BondRefreshPolicy`](@ref)（新）：折入时在**每个两比特截断点重选
#   Clifford 解纠缠门**（`search=true`），把 `adjoint(C)` 放回帧 ——
#   即“折入 + 逐键重选 per-bond disentangler 刷新”，让稳定子结构尽量
#   留在帧外，MPS 只承载魔法成分。
abstract type FramePolicy end

"""
    FoldFramePolicy() <: FramePolicy

经典帧折叠：折入时普通截断、清空帧（当前默认行为）。
"""
struct FoldFramePolicy <: FramePolicy end

"""
    BondRefreshPolicy() <: FramePolicy

折入 + 逐键重选解纠缠（per-bond Clifford disentangler 刷新）：
折入每个两比特门时若发生截断，选择降低截断损失最小的 Clifford `C`，
并把 `adjoint(C)` 放回帧，使稳定子纠缠尽量保留在帧中。
"""
struct BondRefreshPolicy <: FramePolicy end

const DefaultFramePolicy = FoldFramePolicy()

"""
    CAMPS{T,R,P<:FramePolicy}

Clifford 增广 MPS 态：`|ψ⟩ = U_frame|φ⟩`。字段全部**完全参数化**（immutable，
无 UnionAll/抽象字段装箱）：

* `data::MPS{T,R}` —— 承载非稳定子部分的 MPS（类型参数齐备）；
* `frame::CliffordFrame{T}` —— Clifford 帧；
* `policy::P`（`P<:FramePolicy`）—— 帧更新策略，类型参数化避免动态分发。

虽然结构体本身 immutable，内部 `data` / `frame` 仍是可变对象，因此
`fold_frame!` 等在位演化依旧可用；切换策略请用 [`with_policy`](@ref)
构造新态，而不是字段赋值。公开构造请用 [`zero_state`](@ref) 等初始化
函数，或 `CAMPS(psi_mps, frame, [policy])`。
"""
struct CAMPS{T<:Number,R<:Real,P<:FramePolicy}
    data::MPS{T,R}
    frame::CliffordFrame{T}
    policy::P
end
CAMPS(data::MPS{T,R}, frame::CliffordFrame{T}) where {T<:Number,R<:Real} =
    CAMPS{T,R,typeof(DefaultFramePolicy)}(data, frame, DefaultFramePolicy)

"""
    with_policy(ψ, policy) -> CAMPS

返回策略为 `policy` 的新 CAMPS 态（不复制底层 data/frame，仅换策略标记）。
"""
with_policy(ψ::CAMPS, policy::FramePolicy) = CAMPS(ψ.data, ψ.frame, policy)

Base.eltype(::CAMPS{T}) where {T} = T
Base.length(ψ::CAMPS) = length(ψ.data)
Base.copy(ψ::CAMPS) =
    CAMPS(copy(ψ.data), CliffordFrame(copy(ψ.frame.gates)), ψ.policy)
QuantumCircuits.nqubits(ψ::CAMPS) = length(ψ)

"底层 MPS（浅引用；拷贝请用 `copy(ψ.data)`）。"
mps(ψ::CAMPS) = ψ.data
"Clifford 帧。"
frame(ψ::CAMPS) = ψ.frame

# ── 折叠 ────────────────────────────────────────────────────────────────────

"""
    fold_frame!(ψ; trunc=DefaultTruncation) -> Float64

把整个 Clifford 帧作用进 MPS（`U|φ⟩`），按 `ψ.policy` 决定帧的处置：
* [`FoldFramePolicy`](@ref)：普通截断、折入后帧清空；
* [`BondRefreshPolicy`](@ref)：折入每个两比特门时若发生截断则重选
  Clifford 解纠缠门并把 `adjoint(C)` 放回帧（per-bond 刷新）。

返回最大单键截断误差。
"""
function fold_frame!(ψ::CAMPS; trunc::TruncationScheme=DefaultTruncation)
    err = 0.0
    refresh = ψ.policy isa BondRefreshPolicy
    newgates = LocalOp{ComplexF64}[]
    for g in copy(ψ.frame.gates)
        e, C = apply_matrix!(ψ.data, g.mat, g.pos; trunc=trunc, search=refresh)
        err = max(err, e)
        if refresh && C !== nothing && length(g) == 2
            push!(newgates, LocalOp(sort(g.pos), adjoint(C)))
        end
    end
    empty!(ψ.frame.gates)
    refresh && append!(ψ.frame.gates, newgates)
    return err
end

# ── 门作用 ──────────────────────────────────────────────────────────────────

"""
    apply!(ψ::CAMPS, m, positions; trunc, search, wmax) -> ψ

把 `k` 比特局域矩阵 `m`（QuantumCircuits 约定：`positions[1]` = 最高位，
1-based）就地作用到物理态 `|ψ⟩` 上。
"""
function apply!(ψ::CAMPS, m::AbstractMatrix, positions::NTuple{N,Int};
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow) where {N}
    n = length(ψ)
    allunique(positions) || throw(ArgumentError("duplicate qubit positions"))
    all(q -> 1 <= q <= n, positions) || throw(ArgumentError("qubit position out of range [1, $n]"))
    d = 1 << N
    (size(m, 1) == size(m, 2) == d) ||
        throw(DimensionMismatch("matrix size $(size(m)) does not match $N qubit(s)"))
    return _apply_camps!(ψ, m, collect(Int, positions); trunc=trunc, search=search, wmax=wmax)
end

function _apply_camps!(ψ::CAMPS, m::AbstractMatrix, positions::Vector{Int};
                       trunc::TruncationScheme, search::Bool, wmax::Int)
    k = length(positions)
    # Clifford 门（1/2 比特）：进帧，MPS 不动（帧矩阵统一复标量）
    if k <= 2 && is_clifford(m)
        op = LocalOp(positions, Matrix{ComplexF64}(_qc_to_internal(m, positions)))
        push!(ψ.frame.gates, op)
        return ψ
    end
    op = LocalOp(positions, Matrix{ComplexF64}(_qc_to_internal(m, positions)))
    return _apply_nonclifford!(ψ, op; trunc=trunc, search=search, wmax=wmax)
end

function _apply_nonclifford!(ψ::CAMPS, op::LocalOp{T};
                             trunc::TruncationScheme, search::Bool, wmax::Int) where {T}
    if !isempty(ψ.frame)
        conjop = conj_through_frame(ψ.frame, op; wmax=wmax)
        if conjop === nothing
            # 共轭光锥超过 wmax：折叠帧（折成一般 k 比特门序列）后重试
            fold_frame!(ψ; trunc=trunc)
            return _apply_nonclifford!(ψ, op; trunc=trunc, search=search, wmax=wmax)
        end
        op = conjop
    end
    if length(op) == 1
        _apply_single!(ψ.data, op.mat, op.pos[1])
    else
        err, C = apply_matrix!(ψ.data, op.mat, op.pos; trunc=trunc, search=search)
        # CAMPS 截断增强：把选中的解纠缠门 C 以 adjoint(C) 前插进帧
        if C !== nothing && err > 0
            pushfirst!(ψ.frame.gates, LocalOp(sort(op.pos), adjoint(C)))
        end
    end
    return ψ
end

apply(ψ::CAMPS, m::AbstractMatrix, positions::Union{NTuple{N,Int},Vector{Int}};
      trunc::TruncationScheme=DefaultTruncation,
      search::Bool=DefaultSearchDisentangler,
      wmax::Int=DefaultConjWindow) where {N} =
    apply!(copy(ψ), m, Tuple(positions); trunc=trunc, search=search, wmax=wmax)

"就地归一化（帧为酉，只需缩放 MPS）。"
function normalize!(ψ::CAMPS)
    nrm = norm(ψ.data)
    nrm > 0 || throw(ArgumentError("cannot normalize a zero state"))
    _scale!(ψ.data, inv(nrm))
    return ψ
end

"态 2-范数（物理态范数 = MPS 范数，因帧酉）。"
LinearAlgebra.norm(ψ::CAMPS) = norm(ψ.data)

# ── Pauli 弦期望（帧内精确）────────────────────────────────────────────────

"⟨φ| sign * ⊗_s σ_{p(s)} |φ⟩（`p(s)∈1:X,2:Y,3:Z`；逐站点单比特矩阵精确作用）。"
function _string_expect(psi::MPS, factors::Dict{Int,Int}, sign::Float64)
    T = eltype(psi)
    ps = sort!(collect(keys(factors)))
    m2 = copy(psi)
    for s in ps
        p = factors[s]
        σ = p == 1 ? _PAULI_X : p == 2 ? _PAULI_Y : _PAULI_Z
        _apply_single!(m2, σ, s)
    end
    val = real(dot(psi, m2))
    return sign * val
end

"""
    conj_pauli_frame(frame, q, pidx=3)

帧共轭 `U† σ_{pidx}(q) U`（pidx：1=X、2=Y、3=Z；默认 Z）。
返回 `(sign, factors)`，`factors::Dict{Int,Int}` 为 `s → pauli`
（pauli：1=X、2=Y、3=Z），精确符号表示，无窗上限。
"""
conj_pauli_frame(frame::CliffordFrame, q::Int, pidx::Int=3) =
    _conj_pauli_impl(frame, q, pidx)

function _conj_pauli_impl(frame::CliffordFrame, q::Int, pidx::Int)
    sign = 1.0
    factors = Dict{Int,Int}(q => pidx)
    for H in reverse(frame.gates)
        any(s -> s in H.pos, keys(factors)) || continue
        if length(H.pos) == 1
            s = H.pos[1]
            p = get(factors, s, 0)
            p == 0 && continue
            h = _PAULI1[p+1]
            hc = H.mat' * h * H.mat
            bcode, bc = _extract_pauli1(hc)
            sign *= bc
            delete!(factors, s)
            bcode == 0 || (factors[s] = bcode)
        else
            i, j = H.pos
            pi = get(factors, i, 0)
            pj = get(factors, j, 0)
            (pi == 0 && pj == 0) && continue
            h = kron(_PAULI1[pj+1], _PAULI1[pi+1])     # bit0 = 站点 i
            hc = H.mat' * h * H.mat
            bestc = 0.0
            bestcode = 0
            for code in 0:15
                σ = _single_pauli(2, code)
                cc = real(tr(σ * hc)) / 4
                abs(cc) > abs(bestc) && (bestc = cc; bestcode = code)
            end
            abs(bestc) > 1e-8 || error("frame conjugation produced non-Pauli image")
            sign *= signbit(bestc) ? -1.0 : 1.0
            delete!(factors, i)
            delete!(factors, j)
            pi2 = bestcode & 0b11
            pj2 = (bestcode >> 2) & 0b11
            pi2 == 0 || (factors[i] = pi2)
            pj2 == 0 || (factors[j] = pj2)
        end
    end
    return sign, factors
end

"提取单比特 Hermitian 2×2 的 ±Pauli 成分，返回 (code, ±1)；code∈0..3（0=I）。"
function _extract_pauli1(A::AbstractMatrix{<:Number})
    bestc = 0.0
    bestcode = 0
    for code in 0:3
        cc = real(tr(_PAULI1[code+1] * A)) / 2
        abs(cc) > abs(bestc) && (bestc = cc; bestcode = code)
    end
    return bestcode, signbit(bestc) ? -1.0 : 1.0
end

# ── 单比特边缘概率 / 测量 ─────────────────────────────────────────────────

"qubit `q` 的计算基边缘概率 `[p₀, p₁]`（帧内精确）。"
function marginal_probabilities(ψ::CAMPS, q::Int)
    n = length(ψ)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    sign, factors = conj_pauli_frame(ψ.frame, q, 3)
    ez = _string_expect(ψ.data, factors, sign)
    p0 = (1 + ez) / 2
    p1 = (1 - ez) / 2
    return [max(p0, 0.0), max(p1, 0.0)]
end

"""
    measure!(ψ::CAMPS, q::Integer; trunc=DefaultTruncation) -> Int

测量 qubit `q`（计算基）并就地坍缩。帧非空时先折叠（折叠以 `trunc`
逐门截断），再局部投影；概率计算始终在帧内精确进行。
"""
function measure!(ψ::CAMPS, q::Integer; trunc::TruncationScheme=DefaultTruncation)
    q = Int(q)
    p0, p1 = marginal_probabilities(ψ, q)
    outcome = rand() * (p0 + p1) < p0 ? 0 : 1
    p = outcome == 0 ? p0 : p1
    p > 1e-14 || throw(ArgumentError("cannot collapse onto zero-probability outcome"))
    if !isempty(ψ.frame)
        fold_frame!(ψ; trunc=trunc)
    end
    # 局部投影 |b⟩⟨b|（非幺正，随后 canonicalize 恢复规范与键谱）
    P = zeros(ComplexF64, 2, 2)
    P[outcome+1, outcome+1] = 1.0
    _apply_single!(ψ.data, P, q)
    canonicalize!(ψ.data; trunc=NoTruncation(), normalize=true)
    return outcome
end

measure!(ψ::CAMPS, qs::AbstractVector{Int}; trunc::TruncationScheme=DefaultTruncation) =
    Int[measure!(ψ, q; trunc=trunc) for q in qs]

measure!(ψ::CAMPS; trunc::TruncationScheme=DefaultTruncation) =
    (o = 0; for q in 1:length(ψ); o |= measure!(ψ, q; trunc=trunc) << (q - 1); end; o)

# QuantumCircuits measure 协议（非就地）
function QuantumCircuits.measure(ψ::CAMPS, q::Integer;
                                 trunc::TruncationScheme=DefaultTruncation)
    out = copy(ψ)
    p0, p1 = marginal_probabilities(out, q)
    b = measure!(out, q; trunc=trunc)
    return out, b, (b == 0 ? p0 : p1)
end

# ── 单比特局域观测量期望 ───────────────────────────────────────────────────

"""
    expectation(ψ::CAMPS, m::AbstractMatrix, q::Int) -> Real

单比特 Hermitian 观测量 `m` 在物理态上的期望 `⟨ψ|m_q|ψ⟩`（Pauli 分解 + 帧内精确）。
"""
function expectation(ψ::CAMPS, m::AbstractMatrix, q::Int)
    (size(m) == (2, 2)) || throw(DimensionMismatch("single-qubit observable required"))
    T = eltype(ψ)
    c0 = real(tr(m)) / 2
    acc = c0
    for (p, σ) in enumerate((_PAULI_X, _PAULI_Y, _PAULI_Z))
        c = real(tr(σ * m)) / 2
        abs(c) < 1e-12 && continue
        sign, factors = conj_pauli_frame(ψ.frame, q, p)
        acc += c * _string_expect(ψ.data, factors, sign)
    end
    return real(acc)
end

"""
    expectation(ψ::CAMPS, ops) -> Real

对若干互异站点上的单比特局域算子的张量积求期望（`ops` 为 `q => 2×2` 对列表）。
逐项 Pauli 展开后把各共轭弦顺序作用到 MPS 副本上再与 bra 收缩（无需合并同
站点因子，任何对易/重叠情况都精确）。
"""
function expectation(ψ::CAMPS, ops::AbstractVector{<:Pair{Int,<:AbstractMatrix}})
    isempty(ops) && return 1.0
    length(ops) <= 6 || throw(ArgumentError("multi-qubit expectation with >6 sites not supported"))
    paulis = [(1, _PAULI_X), (2, _PAULI_Y), (3, _PAULI_Z)]
    qs = Int[]
    choices = Vector{Any}[]
    for (q, m) in ops
        push!(qs, q)
        ch = [(0, real(tr(m)) / 2)]
        for (p, σ) in paulis
            c = real(tr(σ * m)) / 2
            abs(c) < 1e-12 || push!(ch, (p, c))
        end
        push!(choices, ch)
    end
    allunique(qs) || throw(ArgumentError("duplicate qubits in expectation"))
    acc = 0.0
    for combo in Iterators.product(choices...)
        coeff = 1.0
        strings = Tuple{Float64,Dict{Int,Int}}[]
        for (q, ch) in zip(qs, combo)
            p, c = ch
            coeff *= c
            if p != 0
                sg, fac = conj_pauli_frame(ψ.frame, q, p)
                push!(strings, (sg, fac))
            end
        end
        isempty(strings) && (acc += coeff; continue)
        # Õ₁⋯Õₖ|φ⟩：逐弦顺序作用（弦内为互异站点的单比特 σ，精确）
        wk = copy(ψ.data)
        for (sg, fac) in strings
            _string_apply!(wk, fac, sg)
        end
        acc += coeff * real(dot(ψ.data, wk))
    end
    return acc
end

"把 `sign * ⊗_s σ_{p(s)}`（`factors: s→p`，p:1=X,2=Y,3=Z）作用到 MPS 副本上。"
function _string_apply!(psi::MPS, factors::Dict{Int,Int}, sign::Float64)
    for s in sort!(collect(keys(factors)))
        p = factors[s]
        σ = p == 1 ? _PAULI_X : p == 2 ? _PAULI_Y : _PAULI_Z
        _apply_single!(psi, σ, s)
    end
    sign != 1.0 && _scale!(psi, sign)
    return psi
end

# ── 幅度 / 态矢 ─────────────────────────────────────────────────────────────

"""
    amplitude(ψ::CAMPS, bits::AbstractVector{Int}) -> Complex

计算基 `|bits⟩`（`bits[i]` = qubit `i` 取值）的振幅 `⟨bits|ψ⟩`。
"""
function amplitude(ψ::CAMPS, bits::AbstractVector{Int})
    n = length(ψ)
    length(bits) == n || throw(DimensionMismatch("bits length must equal nqubits"))
    all(b -> b == 0 || b == 1, bits) || throw(ArgumentError("bits must be 0 or 1"))
    T = eltype(ψ)
    ω = onehot_mps(T, collect(Int, bits))
    # |ω⟩ = U†|bits⟩：按帧逆序应用伴随门
    for g in reverse(ψ.frame.gates)
        apply_matrix!(ω, adjoint(g.mat), g.pos; trunc=NoTruncation(), search=false)
    end
    # ⟨bits|ψ⟩ = ⟨φ| (U†|bits⟩) 的共轭
    return conj(dot(ψ.data, ω))
end

"""
    statevector(ψ::CAMPS) -> Vector

把态物化为态矢量（站点 1 = LSB）。仅适用于小比特数。
"""
function statevector(ψ::CAMPS)
    n = length(ψ)
    n <= 24 || throw(ArgumentError("statevector materialization limited to <=24 qubits"))
    T = eltype(ψ)
    v = _statevector_from_mps(ψ.data)
    v = _apply_frame_vec(ψ.frame, v)
    return v
end

"把帧门（按顺序 g₁⋯g_m）作用到态矢量上。"
function _apply_frame_vec(frame::CliffordFrame, v::AbstractVector{T}) where {T}
    n = round(Int, log2(length(v)))
    for g in frame.gates
        v = _vec_gate_apply(v, g.mat, g.pos, n)
    end
    return v
end

function _vec_gate_apply(v::AbstractVector{T}, m::AbstractMatrix, pos::Vector{Int},
                         n::Int) where {T}
    k = length(pos)
    dim = 1 << n
    (size(m) == (1 << k, 1 << k)) || throw(DimensionMismatch("vec gate size mismatch"))
    out = zeros(promote_type(T, eltype(m)), dim)
    idxs = [p - 1 for p in pos]                       # 向量位（bit(site)=site-1，升序）
    gatebits = 0
    for t in 1:k
        gatebits |= 1 << idxs[t]
    end
    othermask = (dim - 1) ⊻ gatebits
    for i in 0:(dim-1)
        others = i & othermask
        # 提取 i 在 idxs 位上的矩阵行号（矩阵 bit (t-1) ↔ 向量位 idxs[t]）
        ai = 0
        for t in 1:k
            ai |= ((i >> idxs[t]) & 1) << (t - 1)
        end
        acc = zero(eltype(out))
        for combo in 0:((1 << k) - 1)
            j = others
            for t in 1:k
                ((combo >> (t - 1)) & 1) == 1 && (j |= 1 << idxs[t])
            end
            acc += m[ai+1, combo+1] * v[j+1]
        end
        out[i+1] = acc
    end
    return out
end
