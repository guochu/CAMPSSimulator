# mps.jl — 纯态 MPS（规范保持：右正则张量 + 键谱 svectors）
#
# 表示（与 MPSSimulator 一致）：
#   * `data`：N 个站点张量 `A::Array{T,3}`，轴序 = (左键, 物理维, 右键)，物理维 2；
#   * 量子态 = 张量网络普通收缩（网络不含额外权重结点）；
#   * `svectors::Vector{Vector{R}}`（长度 N+1）：`svectors[b]` 为键 `b` 的
#     Schmidt 谱（真实奇异值），边界键为 [1]。
#
# 任意 N 体门（相邻块，Hastings 式逐键谱预条件 SVD）与单比特门都保持
# 右正则性 / 谱正确；截断严格按 bipartition Schmidt 奇异值进行
# （见 `kernels.jl`）。

"""
    MPS{T}

物理维 2 的纯态 MPS（开放边界），携带键谱 `svectors`（可为 `nothing` 表示
谱未知）。内部类型，通常经 [`CAMPS`](@ref) 使用。
"""
mutable struct MPS{T<:Number,R<:Real}
    data::Vector{Array{T,3}}
    svectors::Union{Nothing,Vector{Vector{R}}}

    function MPS{T,R}(data::Vector{<:AbstractArray{<:Number,3}},
                      svectors::Union{Nothing,Vector{Vector{R}}}) where {T<:Number,R<:Real}
        R == real(T) || throw(ArgumentError("singular value type must be real(T)"))
        _check_mps_data(data, svectors)
        new{T,R}(convert(Vector{Array{T,3}}, data), svectors)
    end
end

function MPS(data::Vector{<:AbstractArray{T,3}},
             svectors::Union{Nothing,Vector{Vector{R}}}=nothing) where {T<:Number,R<:Real}
    return MPS{T,R}(data, svectors)
end

function _check_mps_data(data::Vector{<:AbstractArray{<:Number,3}},
                          svectors::Union{Nothing,Vector{<:AbstractVector{<:Real}}})
    L = length(data)
    L >= 1 || throw(ArgumentError("MPS must have at least one site"))
    for k in 2:L
        size(data[k-1], 3) == size(data[k], 1) ||
            throw(DimensionMismatch("bond dimension mismatch between site $(k-1) and $k"))
    end
    (size(data[1], 1) == 1 && size(data[end], 3) == 1) ||
        throw(ArgumentError("open boundary MPS requires boundary bond dimensions of 1"))
    all(A -> size(A, 2) == 2, data) ||
        throw(ArgumentError("physical dimension of each site must be 2"))
    if svectors !== nothing
        length(svectors) == L + 1 ||
            throw(DimensionMismatch("length of svectors must be length(sites)+1"))
    end
    return nothing
end

Base.eltype(::MPS{T}) where {T} = T
Base.length(psi::MPS) = length(psi.data)
Base.copy(psi::MPS) =
    MPS([copy(A) for A in psi.data],
        psi.svectors === nothing ? nothing : [copy(v) for v in psi.svectors])
Base.getindex(psi::MPS, k::Int) = psi.data[k]
Base.setindex!(psi::MPS, A::AbstractArray{<:Number,3}, k::Int) = (psi.data[k] = A; psi)
Base.:(==)(a::MPS, b::MPS) =
    length(a) == length(b) && a.data == b.data && _sv_eq(a.svectors, b.svectors)
_sv_eq(a::Nothing, b::Nothing) = true
_sv_eq(a::Nothing, ::Any) = false
_sv_eq(::Any, b::Nothing) = false
_sv_eq(a, b) = a == b

"键维数组（长度 L+1）。"
bond_dimensions(psi::MPS) = Int[size(psi[1], 1); [size(A, 3) for A in psi.data]]
"最大键维。"
max_bond_dimension(psi::MPS) = maximum(bond_dimensions(psi))
"键谱是否未初始化。"
svectors_uninitialized(psi::MPS) = psi.svectors === nothing

"整态乘标量 `λ`（就地；缩放首张量与内键谱以保持一致）。"
function _scale!(psi::MPS, λ::Number)
    psi.data[1] .*= λ
    if psi.svectors !== nothing
        for k in 2:length(psi)
            psi.svectors[k] .*= abs(λ)
        end
    end
    return psi
end

# ── 张量收缩原语（环境 / 内积 / 范数）─────────────────────────────────────

"左→右转移：`E′[r₁,r₂] = Σ_{l₁,p,l₂} conj(A[l₁,p,r₁]) E[l₁,l₂] B[l₂,p,r₂]`。"
function _transfer(E::AbstractMatrix, A::AbstractArray{T,3}, B::AbstractArray{S,3}) where {T,S}
    Dl1, d, Dr1 = size(A)
    Dl2, d2, Dr2 = size(B)
    (d == d2 && size(E, 1) == Dl1 && size(E, 2) == Dl2) ||
        throw(DimensionMismatch("transfer dimension mismatch"))
    ET = promote_type(T, S, eltype(E), Float64)
    T1 = reshape(E * reshape(B, Dl2, d * Dr2), Dl1, d, Dr2)
    Abc = conj.(A)
    E2 = zeros(ET, Dr1, Dr2)
    for p in 1:d
        E2 .+= transpose(@view Abc[:, p, :]) * (@view T1[:, p, :])
    end
    return E2
end

"""
    LinearAlgebra.norm(psi::MPS)

态 2-范数（双网络全收缩，不依赖规范形式）。
"""
function LinearAlgebra.norm(psi::MPS)
    E = fill(one(real(eltype(psi))), 1, 1)
    for k in 1:length(psi)
        E = _transfer(E, psi[k], psi[k])
    end
    return sqrt(max(real(E[1, 1]), zero(real(E[1, 1]))))
end

"""
    LinearAlgebra.dot(a::MPS, b::MPS)

重叠 `⟨a|b⟩`。
"""
function LinearAlgebra.dot(a::MPS, b::MPS)
    length(a) == length(b) || throw(DimensionMismatch("MPS length mismatch"))
    E = fill(one(real(eltype(a))), 1, 1)
    for k in 1:length(a)
        E = _transfer(E, a[k], b[k])
    end
    return E[1, 1]
end

# ── 规范检查 ────────────────────────────────────────────────────────────────

"单张量是否右正则（矩阵 (l × (p,r)) 行正交）。"
function _isrightcanonical(A::AbstractArray{T,3}; atol::Real=1e-8) where {T}
    Dl, d, Dr = size(A)
    M = reshape(A, Dl, d * Dr)
    Id = Matrix{real(T)}(I, Dl, Dl)
    return isapprox(M * M', Id; atol=atol)
end
isrightcanonical(psi::MPS; kwargs...) = all(A -> _isrightcanonical(A; kwargs...), psi.data)

"""
    iscanonical(psi::MPS; atol) -> Bool

psi 是否处于规范形式：所有张量右正则，且 `svectors` 与各键真实 Schmidt 谱
一致（前缀约化密度矩阵 = diag(Λ²)）。
"""
function iscanonical(psi::MPS; atol::Real=1e-8)
    isrightcanonical(psi; atol=atol) || return false
    svectors_uninitialized(psi) && return false
    E = fill(one(real(eltype(psi))), 1, 1)
    for k in 1:(length(psi)-1)
        E = _transfer(E, psi[k], psi[k])
        isapprox(E, Diagonal(psi.svectors[k+1] .^ 2); atol=atol) || return false
    end
    return true
end

# ── 精确规范变换与截断 ─────────────────────────────────────────────────────

"""
    canonicalize!(psi::MPS; trunc=NoTruncation(), normalize=true) -> Float64

把 psi 扫描为规范形式（张量右正则 + svectors 记录各键 Schmidt 谱）：
先自左向右 QR 等距化（精确），再自右向左 SVD 捕获谱（`trunc` 非空时逐键
截断，兼作压缩）。返回最大单键截断误差。`normalize=true` 归一到单位范数。
"""
function canonicalize!(psi::MPS; trunc::TruncationScheme=NoTruncation(),
                       normalize::Bool=true)
    L = length(psi)
    T = eltype(psi)
    R = real(T)
    # 1) 左扫：QR 等距化（保证右扫 SVD 谱为真实 Schmidt 谱）
    for k in 1:(L-1)
        A = psi[k]
        Dl = size(A, 1)
        M = reshape(A, Dl * 2, :)
        F = qr(M)
        q = Matrix(F.Q)
        r = F.R
        psi[k] = reshape(q, Dl, 2, size(r, 1))
        A2 = psi[k+1]
        psi[k+1] = reshape(reshape(r, size(r, 1), size(A2, 1)) *
                           reshape(A2, size(A2, 1), 2 * size(A2, 3)),
                           size(r, 1), 2, size(A2, 3))
    end
    # 2) 右扫：SVD 捕获 Schmidt 谱
    sv = Vector{Vector{R}}(undef, L + 1)
    sv[1] = [one(R)]
    sv[L+1] = [one(R)]
    err = 0.0
    for k in L:-1:2
        A = psi[k]
        Dl = size(A, 1)
        M = reshape(A, Dl, :)
        u, s, vt, e = _tsvd_trunc(M; trunc=trunc)
        err = max(err, e)
        psi[k] = reshape(vt, size(vt, 1), 2, size(A, 3))
        sv[k] = convert(Vector{R}, s)
        A2 = psi[k-1]
        psi[k-1] = reshape(reshape(A2, size(A2, 1) * 2, size(A2, 3)) *
                           (u .* reshape(s, 1, :)),
                           size(A2, 1), 2, size(vt, 1))
    end
    psi.svectors = sv
    if normalize
        n = norm(psi)
        n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
        psi.data[1] ./= n
        for k in 2:L
            psi.svectors[k] ./= n
        end
    end
    return err
end

"""
    _tsvd_trunc(M; trunc) -> (u, s, vt, err)

`M ≈ u * Diagonal(s) * vt`（截断按奇异值降序），返回截断平方误差 `err`。
"""
function _tsvd_trunc(M::AbstractMatrix{T}; trunc::TruncationScheme=NoTruncation()) where {T}
    F = svd(M; full=false)
    s = F.S
    r, e = _truncation_keep(s, trunc)
    r = max(r, 1)
    return F.U[:, 1:r], s[1:r], F.Vt[1:r, :], e
end

"""
    LinearAlgebra.normalize!(psi::MPS) -> psi

就地归一化到单位 2-范数（缩放首张量；键谱不随之缩放，调用方可随后
`canonicalize!` 恢复规范）。
"""
function LinearAlgebra.normalize!(psi::MPS)
    n = norm(psi)
    n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
    psi.data[1] ./= n
    return psi
end
