# kernels.jl — 一般 k 比特门作用（Hastings 式，保持规范与键谱）
#
# 说明（与 MPSSimulator 一致）：
#   * MPS 保持“张量右正则 + 键谱记录各键 Schmidt 谱”；
#   * 任意 N 体门作用于相邻块时：块内自左向右逐键做谱预条件 SVD
#     （`W = Λ .* B`），截断严格按该键 bipartition 的 Schmidt 奇异值；
#   * Hastings 左重构 `Ah = B·v†`（未加权块 × v†）避免除以小奇异值，
#     无截断时状态精确重构；新键谱写入 `core.s`；
#   * 相邻内容交换（SWAP 聚集）直接复用 `FiniteMPSAlgorithms.swap!`
#     （Hastings 式交换，正则形式保持）；
#   * 单比特门直接作用物理腿，不改变右正则性与键谱。
#   * 两比特更新保留本地实现（CAMPS 特有：截断处的 Clifford 解纠缠搜索，
#     见 `_apply_contiguous!` / `_best_disentangle`）。
#
# 每个多比特门作用（含 SWAP 网络）结束会在 `apply_matrix!` 里检测范数漂移
# 并归一化（缩放首张量与键谱），保证键谱恒为真实 Schmidt 谱。

# ── 块收缩 / 门作用 ────────────────────────────────────────────────────────

"把站点 `a:a+k-1` 收缩为块张量（Dl, 2^k, Dr），物理指标 p = Σ p_j 2^j（bit0=站点 a）。"
function _contract_window(psi::MPS, a::Int, k::Int)
    Θ = reshape(copy(psi[a]), size(psi[a], 1), 2, size(psi[a], 3))
    for j in 1:(k-1)
        A = psi[a+j]
        Dl, P, B = size(Θ)
        B == size(A, 1) || throw(DimensionMismatch("window contraction bond mismatch"))
        C = reshape(Θ, Dl * P, B) * reshape(A, B, 2 * size(A, 3))
        Θ = reshape(C, Dl, P * 2, size(A, 3))
    end
    return Θ
end

"把 k 体门 `G`（小端序、bit0=站点 a）作用到块张量物理轴上。"
function _apply_gate_tensor(G::AbstractMatrix, Θ::AbstractArray{T,3}) where {T}
    dn = size(Θ, 2)
    (size(G, 1) == size(G, 2) == dn) || throw(DimensionMismatch("gate size mismatch"))
    Θp = permutedims(Θ, (2, 1, 3))
    out = G * reshape(Θp, dn, :)
    return permutedims(reshape(out, dn, size(Θ, 1), size(Θ, 3)), (2, 1, 3))
end

"Hastings 左重构：`Ah[l,p,α] = Σ_rest B[l,p,rest] conj(v[α,rest])`。"
function _hastings_left(B::AbstractArray{T,3}, v::AbstractMatrix, d::Int) where {T}
    Bm = reshape(B, size(B, 1) * d, :)
    return reshape(Bm * adjoint(v), size(B, 1), d, size(v, 1))
end

"""
    _apply_block!(psi, G, a, k, trunc) -> err

把 k 体门 `G`（小端序、bit0=站点 a）作用到相邻站点 `a:a+k-1`（就地，
Hastings 式逐键更新），返回最大单键截断误差。要求键谱已初始化且
进入时态处于规范形式。
"""
function _apply_block!(psi::MPS, G::AbstractMatrix, a::Int, k::Int,
                       trunc::TruncationScheme)
    svectors_uninitialized(psi) &&
        throw(ArgumentError("gate update requires initialized svectors"))
    B = _apply_gate_tensor(G, _contract_window(psi, a, k))
    Dl, P, Dr = size(B)
    Λ = psi.core.s[a]                         # 块左键谱（预条件子）
    err = 0.0
    for j in 1:(k-1)
        W = reshape(Λ .* B, Dl * 2, (P ÷ 2) * Dr)
        F = svd(W; full=false)
        s = F.S
        r, e = _truncation_keep(s, trunc)
        r = max(r, 1)
        err = max(err, e)
        v = F.Vt[1:r, :]
        Ah = _hastings_left(B, v, 2)
        psi[a+j-1] = reshape(Ah, Dl, 2, r)
        psi.core.s[a+j] = convert(Vector{real(eltype(psi))}, s[1:r])
        B = reshape(v, r, P ÷ 2, Dr)
        Λ = s[1:r]
        Dl = r
        P = P ÷ 2
    end
    psi[a+k-1] = reshape(B, Dl, 2, Dr)
    return err
end

"两比特窗口收缩（保留给搜索 / SWAP 等既有调用）。"
_contract_two(psi::MPS, a::Int) = _contract_window(psi, a, 2)

_apply_gate2(G::AbstractMatrix, B::AbstractArray{T,3}) where {T} =
    _apply_gate_tensor(G, B)

"""
    _apply_two!(psi, G, a, trunc) -> err

把两比特门 `G`（小端序、bit0=站点 a）作用到相邻站点 `a, a+1`（就地，
Hastings 式），返回截断误差。要求键谱已初始化且键谱正确。
"""
function _apply_two!(psi::MPS, G::AbstractMatrix, a::Int, trunc::TruncationScheme)
    svectors_uninitialized(psi) &&
        throw(ArgumentError("two-site update requires initialized svectors"))
    B = _apply_gate2(G, _contract_two(psi, a))
    Dl, _, Dr = size(B)
    Λ = psi.core.s[a]
    W = reshape(Λ .* B, Dl * 2, 2 * Dr)
    F = svd(W; full=false)
    s = F.S
    r, e = _truncation_keep(s, trunc)
    r = max(r, 1)
    v = F.Vt[1:r, :]
    Ah = _hastings_left(B, v, 2)
    psi[a] = reshape(Ah, Dl, 2, r)
    psi.core.s[a+1] = convert(Vector{real(eltype(psi))}, s[1:r])
    psi[a+1] = reshape(v, r, 2, Dr)
    return e
end

"""
    _apply_two_c!(psi, G, a, trunc, C) -> (err)

用给定的解纠缠门 `C` 对 `G` 后的块作用并截断：`C` 先作用于物理腿，
截断在“规范两体态”上进行，返回截断误差。
"""
function _apply_two_c!(psi::MPS, G::AbstractMatrix, a::Int,
                       trunc::TruncationScheme, C::Matrix)
    B = _apply_gate2(C, _apply_gate2(G, _contract_two(psi, a)))
    Dl, _, Dr = size(B)
    Λ = psi.core.s[a]
    W = reshape(Λ .* B, Dl * 2, 2 * Dr)
    F = svd(W; full=false)
    s = F.S
    r, e = _truncation_keep(s, trunc)
    r = max(r, 1)
    v = F.Vt[1:r, :]
    Ah = _hastings_left(B, v, 2)
    psi[a] = reshape(Ah, Dl, 2, r)
    psi.core.s[a+1] = convert(Vector{real(eltype(psi))}, s[1:r])
    psi[a+1] = reshape(v, r, 2, Dr)
    return e
end

"候选解纠缠门作用后的截断误差（丢弃平方和），`keep` 为目标保留维数。"
function _discard2(B::AbstractArray{T,3}, Λ::AbstractVector, keep::Int) where {T}
    Dl = size(B, 1)
    W = reshape(Λ .* B, Dl * 2, 2 * size(B, 3))
    s = svdvals(W)
    k = min(length(s), keep)
    k >= length(s) && return 0.0
    return sum(abs2, view(s, k+1:length(s)))
end

"""
    _best_disentangle(B, Λ, trunc) -> (Cbest, besterr)

从非局域类代表元中选使截断损失最小的两比特 Clifford `C`（CAMPS 增强）。
返回 `(C, err)`；`C === nothing` 表示恒等即最优。
"""
function _best_disentangle(B::AbstractArray{T,3}, Λ::AbstractVector,
                           trunc::TruncateDim) where {T}
    Dl = size(B, 1)
    keep = min(trunc.D, Dl * 2, 2 * size(B, 3))
    I4 = Matrix{ComplexF64}(I, 4, 4)
    bestC = nothing
    besterr = _discard2(B, Λ, keep)
    iszero(besterr) && return bestC, besterr
    for C in DISENTANGLER_CANDIDATES
        isapprox(C, I4; atol=1e-12) && continue
        Bc = _apply_gate2(C, B)
        e = _discard2(Bc, Λ, keep)
        if e < besterr - 1e-14
            bestC = C
            besterr = e
        end
        iszero(e) && break
    end
    return bestC, besterr
end

# ── 单比特门 ────────────────────────────────────────────────────────────────

"单比特门（2×2）作用到站点 `a`（就地；保持右正则与键谱）。"
function _apply_single!(psi::MPS, m::AbstractMatrix, a::Int)
    A = psi[a]
    T = promote_type(eltype(A), eltype(m))
    Dl, d, Dr = size(A)
    out = similar(A, T)
    for l in 1:Dl, r in 1:Dr
        @inbounds for p in 1:2
            acc = zero(T)
            for q in 1:2
                acc += m[p, q] * A[l, q, r]
            end
            out[l, p, r] = acc
        end
    end
    psi[a] = out
    return psi
end

# ── SWAP 聚集 / 散开（任意位置、任意比特数的门）──────────────────────────
#
# 为什么需要 SWAP 网络：
#   MPS 门内核（`_apply_block!` / `_apply_two!`）只支持把门作用到**链上相邻
#   的连续块**（Hastings 逐键 SVD 必须按“站点 k ↔ qubit k”的链上键顺序）。
#   当用户给的门落在任意比特（例如 qubit 3 与 qubit 7）时，先把这些参与
#   比特用相邻 SWAP 一一搬到位（聚集，复用 `FiniteMPSAlgorithms.swap!`），
#   在连续块上作用门，再用相邻 SWAP 把被临时挤开的其它比特送回原位（散开）。
#
# 约定：`positions` 传入时已升序。聚集不改变参与比特之间的相对顺序——
#   `positions[k]` 最终落在 `positions[1]+k-1`，因此**聚集后块内自左向右的
#   顺序 = 升序 positions = 门矩阵的小端序位序**（bit j ↔ 第 j 小的站点），
#   与 `m`（内部小端序）直接对应，无需额外重排。

"""
    _gather!(psi, positions, trunc) -> swaps

把 `positions`（升序）上的比特**聚集**到连续块 `positions[1] … positions[1]+N-1`
（就地；每次相邻交换复用 `FiniteMPSAlgorithms.swap!`，截断可能造成范数丢失，
由调用方在门作用完成后统一检测并归一化）。返回交换序列 `(s-1) => s`，
供 [`_scatter!`](@ref) 逆序撤销。
"""
function _gather!(psi::MPS, positions::Vector{Int}, trunc::TruncationScheme)
    swaps = Pair{Int,Int}[]
    for k in 2:length(positions)
        s = positions[k]                     # 第 k 个参与比特当前所在位置
        t = positions[1] + k - 1             # 它在聚集后块中的目标位置
        while s > t
            swap!(psi.core, s - 1; trunc=trunc)
            push!(swaps, (s - 1) => s)       # 记录这次相邻交换（散开时需回放）
            s -= 1
        end
    end
    return swaps
end

"""
    _scatter!(psi, swaps, trunc) -> psi

按**逆序**回放 [`_gather!`](@ref) 记录下的相邻交换序列，把被临时挤开的
比特送回原位（就地），恢复“站点 k ↔ qubit k”的对应关系。
（相邻 SWAP 自逆，逆序重放恰好是 gather 置换的逆置换。）
"""
function _scatter!(psi::MPS, swaps::Vector{Pair{Int,Int}}, trunc::TruncationScheme)
    for (i, j) in reverse(swaps)
        swap!(psi.core, i; trunc=trunc)
    end
    return psi
end

"""
    apply_matrix!(psi, m, positions_sorted; trunc, search) -> (err, Cused)

把内部小端序门矩阵 `m`（长度任意 k，`positions_sorted` 升序）作用到 MPS
（就地，Hastings 更新）。非相邻位置自动 SWAP 聚集再散开。

多比特门完整流程：
1. 若 `positions_sorted` 已是相邻连续块 `[a, a+k-1]`：直接作用
   （k=2 走带解纠缠搜索的 `_apply_contiguous!`，k>2 走 `_apply_block!`）；
2. 否则先 `_gather!`：用相邻 SWAP 把参与比特搬到连续块 `[a, a+k-1]`
   （相对顺序不变，门矩阵位序一致），作用门；
3. `_scatter!`：逆序回放交换，把无关比特送回原位，恢复站点↔比特一一对应；
4. 检测范数漂移（截断会丢弃权重使范数 <1）：漂移超出容差则整态归一化
   （缩放首张量与键谱，保持概率/期望语义与键谱 = 真实 Schmidt 谱）。
"""
function apply_matrix!(psi::MPS, m::AbstractMatrix,
                       positions_sorted::Vector{Int};
                       trunc::TruncationScheme=DefaultTruncation,
                       search::Bool=DefaultSearchDisentangler)
    k = length(positions_sorted)
    if k == 1
        _apply_single!(psi, m, positions_sorted[1])
        return 0.0, nothing
    end
    a = positions_sorted[1]
    contiguous = positions_sorted == collect(a:(a + k - 1))
    swaps = contiguous ? Pair{Int,Int}[] : _gather!(psi, positions_sorted, trunc)
    if k == 2
        err, C = _apply_contiguous!(psi, m, a, trunc; search=search)
    else
        err = _apply_block!(psi, m, a, k, trunc)
        C = nothing
    end
    _scatter!(psi, swaps, trunc)
    nrm = norm(psi)
    if abs(nrm - 1) > 1e-12 && nrm > 0
        _scale!(psi, inv(nrm))
    end
    return err, C
end

"相邻两比特（规范 Hastings + 可选解纠缠搜索）更新；返回 (err, Cused)。"
function _apply_contiguous!(psi::MPS, m::AbstractMatrix, a::Int,
                            trunc::TruncationScheme; search::Bool)
    if search && trunc isa TruncateDim
        # 无需再硬性判断两侧键维是否超过 D：`_best_disentangle` 内部会
        # 先按恒等评估丢弃权重（无截断/无收益时立即返回 nothing）。
        Gb = _apply_gate2(m, _contract_two(psi, a))
        C, e0 = _best_disentangle(Gb, psi.core.s[a], trunc)
        if C !== nothing
            err = _apply_two_c!(psi, m, a, trunc, C)
            return err, C
        end
    end
    return _apply_two!(psi, m, a, trunc), nothing
end

# ── 态矢收缩（小系统）───────────────────────────────────────────────────────

"把 MPS 收缩为态矢量。bit (site k) = 2^(k-1)（站点 1 = LSB）。"
function _statevector_from_mps(psi::MPS)
    n = length(psi)
    T = eltype(psi)
    cur = fill(one(T), 1, 1)
    for k in 1:n
        A = psi[k]
        Dl, d, Dr = size(A)
        S = size(cur, 2)
        new = zeros(T, Dr, d * S)
        for p in 1:d
            Ap = @view A[:, p, :]
            new[:, (p-1)*S .+ (1:S)] .+= transpose(Ap) * cur
        end
        cur = new
    end
    return vec(cur)
end
