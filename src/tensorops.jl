# tensorops.jl — 奇异值截断方案（类型名与语义对齐 MPSSimulator）

"""
    TruncationScheme

张量截断方案抽象类型：规定奇异值按维数、按误差或两者结合的方式截断。
"""
abstract type TruncationScheme end

"""
    NoTruncation()

不截断，保留全部奇异值。
"""
struct NoTruncation <: TruncationScheme end

"按维数截断：只保留最大的 `D` 个奇异值。"
struct TruncateDim <: TruncationScheme
    D::Int
end
TruncateDim(; D::Int) = TruncateDim(D)

"""
    truncdim(d::Int)
    truncdim(; D::Int)

按维数截断：只保留最大的 `d` 个奇异值。
"""
truncdim(d::Int) = TruncateDim(d)
truncdim(; D::Int) = truncdim(D)

"按相对误差截断：丢弃相对范数低于 `ϵ` 的全部奇异值。"
struct TruncateCutoff <: TruncationScheme
    ϵ::Float64
end
TruncateCutoff(; ϵ::Real) = TruncateCutoff(convert(Float64, ϵ))

"""
    trunccutoff(; ϵ::Real)

按相对误差截断：丢弃相对范数低于 `ϵ` 的奇异值。
"""
trunccutoff(; ϵ::Real) = TruncateCutoff(ϵ)

"""
    TruncationDimCutoff(D, ϵ, add_back=0)
    TruncationDimCutoff(; D, ϵ, add_back=0)

维数 + 误差组合截断：先按相对范数 `ϵ` 确定截断点，再封顶 `D` 个奇异值，
且至少保留 `add_back` 个。
"""
struct TruncationDimCutoff <: TruncationScheme
    D::Int
    ϵ::Float64
    add_back::Int
end
TruncationDimCutoff(; D::Int, ϵ::Real, add_back::Int=0) =
    TruncationDimCutoff(D, float(ϵ), min(add_back, D))

"""
    truncdimcutoff(D, ϵ, add_back=0)
    truncdimcutoff(; D, ϵ, add_back=0)

`TruncationDimCutoff` 的位置参数 / 关键字便捷构造。
"""
truncdimcutoff(D::Int, epsilon::Real; add_back::Int=0) =
    TruncationDimCutoff(D, epsilon, min(add_back, D))
truncdimcutoff(; D::Int, ϵ::Real, add_back::Int=0) =
    TruncationDimCutoff(D, float(ϵ), min(add_back, D))

# ── 截断执行（假定 v 已按奇异值降序）───────────────────────────────────────

"返回 (保留的奇异值数, 截断误差)。不改动向量。"
function _truncation_keep(v::AbstractVector{<:Real}, trunc::TruncationScheme)
    if trunc isa NoTruncation
        return length(v), 0.0
    elseif trunc isa TruncateDim
        r = min(length(v), trunc.D)
        e = sum(abs2, view(v, r+1:length(v)))
        return r, e
    elseif trunc isa TruncateCutoff
        sca = sum(abs2, v)
        r = something(findlast(x -> abs2(x) > sca * trunc.ϵ^2, v), 0)
        return r, sum(abs2, view(v, r+1:length(v)))
    elseif trunc isa TruncationDimCutoff
        sca = sum(abs2, v)
        r = something(findlast(x -> abs2(x) > sca * trunc.ϵ^2, v), 0)
        r = max(r, trunc.add_back)
        r = min(r, trunc.D)
        e = sum(abs2, view(v, r+1:length(v)))
        return r, e
    end
    throw(ArgumentError("unknown truncation scheme $(typeof(trunc))"))
end

"是否可能发生截断（用于跳过不必要的规范/搜索开销）。"
_possibly_truncating(::NoTruncation) = false
_possibly_truncating(::TruncationScheme) = true
