# initializers.jl — 态初始化（返回 CAMPS，帧为空）
#
# 约定：模拟态统一使用复标量（`ComplexF64`；传入实类型时自动提升为复标量），
# 因此任何门（含实 Clifford 门）与帧操作都无需运行时类型转换。

"复标量类型：`T <: Complex` 保留浮点精度，否则回退 `ComplexF64`。"
_ctype(::Type{T}) where {T<:Complex} = complex(float(real(T)))
_ctype(::Type{T}) where {T<:Number} = ComplexF64
_ctype(::Type{T}) where {T} = ComplexF64

"乘积态 MPS：`amps[k]` 为站点 `k` 的局域振幅（长度 2）；键谱平凡（全 1）。"
function _product_mps(::Type{T}, amps::AbstractVector{<:AbstractVector}) where {T<:Number}
    L = length(amps)
    L >= 1 || throw(ArgumentError("empty state"))
    CT = _ctype(T)
    R = real(CT)
    data = [Array{CT,3}(undef, 1, 2, 1) for _ in 1:L]
    for (k, a) in enumerate(amps)
        length(a) == 2 || throw(ArgumentError("each local amplitude must have length 2"))
        data[k][1, :, 1] .= a
    end
    return MPS(data, [[one(R)] for _ in 1:(L+1)])
end

"计算基 MPS：`bits[k]` = qubit `k` 的取值（0/1）。"
function onehot_mps(::Type{T}, bits::AbstractVector{Int}) where {T<:Number}
    all(b -> b == 0 || b == 1, bits) || throw(ArgumentError("bits must be 0 or 1"))
    return _product_mps(T, [b == 0 ? [one(_ctype(T)), zero(_ctype(T))] : [zero(_ctype(T)), one(_ctype(T))] for b in bits])
end
onehot_mps(bits::AbstractVector{Int}) = onehot_mps(ComplexF64, bits)

"直积态 MPS：`amps[k] = cos(πθ/2)|0⟩ + sin(πθ/2)|1⟩`。"
function qubit_encoding_mps(::Type{T}, θs::AbstractVector{<:Real}) where {T<:Number}
    CT = _ctype(T)
    return _product_mps(T, [[CT(cos(s * pi / 2)), CT(sin(s * pi / 2))] for s in θs])
end
qubit_encoding_mps(θs::AbstractVector{<:Real}) = qubit_encoding_mps(ComplexF64, θs)

"""
    zero_state([T=ComplexF64,] n) -> CAMPS

`|0…0⟩`（帧为空）。
"""
zero_state(::Type{T}, n::Int) where {T<:Number} =
    CAMPS(_product_mps(T, [[one(_ctype(T)), zero(_ctype(T))] for _ in 1:n]),
          CliffordFrame{_ctype(T)}())
zero_state(n::Int) = zero_state(ComplexF64, n)

"""
    onehot_state([T=ComplexF64,] bits) -> CAMPS

计算基态（`bits[k]` = qubit `k` 取值）。
"""
onehot_state(::Type{T}, bits::AbstractVector{Int}) where {T<:Number} =
    CAMPS(onehot_mps(T, collect(Int, bits)), CliffordFrame{_ctype(T)}())
onehot_state(bits::AbstractVector{Int}) = onehot_state(ComplexF64, bits)

"""
    qubit_encoding_state([T=ComplexF64,] θs) -> CAMPS

直积态：`qubit k = cos(πθ/2)|0⟩ + sin(πθ/2)|1⟩`。
"""
qubit_encoding_state(::Type{T}, θs::AbstractVector{<:Real}) where {T<:Number} =
    CAMPS(qubit_encoding_mps(T, θs), CliffordFrame{_ctype(T)}())
qubit_encoding_state(θs::AbstractVector{<:Real}) = qubit_encoding_state(ComplexF64, θs)

"""
    rand_state([T=ComplexF64,] n; D=8, rng=Random.default_rng()) -> CAMPS

键维上限 `D` 的随机 CAMPS 态（复高斯张量 + 归一化，帧为空）。
"""
function rand_state(::Type{T}, n::Int; D::Int=8,
                    rng::AbstractRNG=Random.default_rng()) where {T<:Number}
    n >= 1 || throw(ArgumentError("number of qubits must be positive"))
    CT = _ctype(T)
    data = Vector{Array{CT,3}}(undef, n)
    for k in 1:n
        dl = (k == 1) ? 1 : min(D, 2^(k-1), 2^(n-k+1))
        dr = (k == n) ? 1 : min(D, 2^k, 2^(n-k))
        data[k] = convert(Array{CT,3}, randn(rng, ComplexF64, dl, 2, dr))
    end
    mps0 = MPS(data, nothing)
    canonicalize!(mps0; trunc=NoTruncation(), normalize=true)   # 生成键谱
    return CAMPS(mps0, CliffordFrame{CT}())
end
rand_state(n::Int; kwargs...) = rand_state(ComplexF64, n; kwargs...)
