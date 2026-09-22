# mps.jl — 纯态 MPS：FiniteMPSAlgorithms.CanonicalMPS 的薄封装
#
# 表示（与 MPSSimulator 一致）：
#   * 底层为 `CanonicalMPS`：站点张量 `A::Array{T,3}`（轴序 = 左键, 物理维, 右键，
#     物理维 2，开放边界），量子态 = 张量网络普通收缩；
#   * 键 Schmidt 谱存于 `core.s`（`missing` 表示谱未知）；封装层保持
#     `scaling(core) == 1`（范数在站点数据中），FMA 内核把范数吸进
#     `scaling` 后用 `_sync!` 折回。
#
# 正则化（`canonicalize!`）、范数 / 重叠（`norm` / `dot`）与规范检查
# （`iscanonical`）全部复用 FiniteMPSAlgorithms 的现成实现。

"""
    MPS(core::CanonicalMPS)
    MPS(data, [svectors])

物理维 2 的纯态 MPS（开放边界），`FiniteMPSAlgorithms.CanonicalMPS` 的薄封装，
键谱可为 `missing`（表示谱未知）。内部类型，通常经 [`CAMPS`](@ref) 使用。
"""
struct MPS{T<:Number,R<:Real}
    core::CanonicalMPS{T,R}
end

function MPS(core::CanonicalMPS{T,R}) where {T,R}
    R == real(T) || throw(ArgumentError("singular value type must be real(T)"))
    all(A -> size(A, 2) == 2, core.data) ||
        throw(ArgumentError("physical dimension of each site must be 2"))
    return MPS{T,R}(core)
end

function MPS(data::AbstractVector{<:AbstractArray{T,3}},
             svectors::Union{Nothing,AbstractVector}=nothing) where {T<:Number}
    dat = convert(Vector{Array{T,3}}, collect(data))
    return MPS(svectors === nothing ? CanonicalMPS(dat) : CanonicalMPS(dat, svectors))
end

Base.eltype(::Type{MPS{T,R}}) where {T,R} = T
Base.eltype(::MPS{T,R}) where {T,R} = T
Base.length(psi::MPS) = length(psi.core)
Base.getindex(psi::MPS, k::Int) = psi.core[k]
Base.setindex!(psi::MPS, A::AbstractArray{<:Number,3}, k::Int) = (psi.core[k] = A; psi)

_svec_copy(s) = ismissing(s) ? missing : copy(s)

Base.copy(psi::MPS) = MPS(copy(psi.core))

function Base.:(==)(a::MPS, b::MPS)
    length(a) == length(b) || return false
    a.core.data == b.core.data || return false
    return all(_svec_eq(a.core.s[i], b.core.s[i]) for i in eachindex(a.core.s))
end
_svec_eq(a::Missing, b::Missing) = true
_svec_eq(a::Missing, b) = false
_svec_eq(a, b::Missing) = false
_svec_eq(a, b) = a == b

"键维数组（长度 L+1）。"
bond_dimensions(psi::MPS) = Int[size(psi[1], 1); [size(A, 3) for A in psi.core.data]]
"最大键维。"
max_bond_dimension(psi::MPS) = maximum(bond_dimensions(psi))
"键谱是否未初始化。"
svectors_uninitialized(psi::MPS) = svectors_uninitialized(psi.core)

"整态乘标量 `λ`（就地；缩放首张量与内键谱以保持键谱 = 数据真实谱）。"
function _scale!(psi::MPS, λ::Number)
    psi.core[1] = psi.core[1] * λ
    for k in 2:length(psi)
        s = psi.core.s[k]
        ismissing(s) || (psi.core.s[k] = abs(λ) .* s)
    end
    return psi
end

"把 FMA 内核暂存的 per-site scaling 折回站点数据（恢复 scaling ≡ 1 不变量）。"
function _sync!(psi::MPS)
    c = scaling(psi.core)^length(psi.core)
    isone(c) && return psi
    psi.core[1] = psi.core[1] * c
    for k in 2:length(psi)
        s = psi.core.s[k]
        ismissing(s) || (psi.core.s[k] = c .* s)
    end
    setscaling!(psi.core, 1.0)
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

态 2-范数（双网络全收缩，不依赖规范形式）。复用 `FiniteMPSAlgorithms.norm`。
"""
LinearAlgebra.norm(psi::MPS) = norm(psi.core)

"""
    LinearAlgebra.dot(a::MPS, b::MPS)

重叠 `⟨a|b⟩`。复用 `FiniteMPSAlgorithms.dot`。
"""
LinearAlgebra.dot(a::MPS, b::MPS) = dot(a.core, b.core)

# ── 规范检查 ────────────────────────────────────────────────────────────────

"所有站点张量是否右正则（复用 `FiniteMPSAlgorithms.isrightcanonical`）。"
isrightcanonical(psi::MPS; kwargs...) = all(A -> isrightcanonical(A; kwargs...), psi.core.data)

"""
    iscanonical(psi::MPS; atol) -> Bool

psi 是否处于规范形式：所有张量右正则，且键谱与各键真实 Schmidt 谱
一致（前缀约化密度矩阵 = diag(Λ²)）。复用 `FiniteMPSAlgorithms.iscanonical`。
"""
iscanonical(psi::MPS; atol::Real=1e-8) = iscanonical(psi.core; atol)

# ── 精确规范变换与截断 ─────────────────────────────────────────────────────

"""
    canonicalize!(psi::MPS; trunc=NoTruncation(), normalize=true) -> Float64

把 psi 扫描为规范形式（张量右正则 + 键谱记录各键 Schmidt 谱），复用
`FiniteMPSAlgorithms.canonicalize!`：先自左向右 QR 等距化（精确），再自右向左
SVD 捕获谱（`trunc` 非空时逐键截断，兼作压缩）。返回最大单键截断误差。
`normalize=true` 归一到单位范数。
"""
function canonicalize!(psi::MPS; trunc::TruncationScheme=NoTruncation(),
                       normalize::Bool=true)
    _, err = canonicalize!(psi.core; alg=Orthogonalize(SVD(), trunc, false))
    _sync!(psi)
    if normalize
        n = norm(psi)
        n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
        _scale!(psi, inv(n))
    end
    return err
end

"""
    LinearAlgebra.normalize!(psi::MPS) -> psi

就地归一化到单位 2-范数（缩放首张量；键谱同步缩放，保持键谱 = 数据的真实谱）。
"""
function LinearAlgebra.normalize!(psi::MPS)
    n = norm(psi)
    n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
    return _scale!(psi, inv(n))
end
