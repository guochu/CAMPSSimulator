# tensorops.jl — 张量操作层：re-export FiniteMPSAlgorithms
#
# 截断方案（`TruncationScheme` 及各具体方案）与正交 / 奇异值分解原语
# 直接复用 FiniteMPSAlgorithms（不使用其 experimental 部分）；
# 这里只保留 CAMPSSimulator 特有的截断辅助。

# 兼容别名：CAMPSSimulator 旧类型名 → FiniteMPSAlgorithms 的截断方案
const TruncateCutoff = TruncateRelError
const TruncationDimCutoff = TruncateDimCutoff
trunccutoff(args...; kwargs...) = truncrelerr(args...; kwargs...)

"""
    _truncation_keep(v, trunc) -> (r, err2)

返回保留的奇异值个数 `r` 与被丢弃权重的平方和 `err2`（不修改 `v`）。
基于 `FiniteMPSAlgorithms.truncate!`（`err2 = err^2`，`err` 为丢弃的
2-范数；组合方案 `TruncateDimCutoff` 的 `err` 为相对值）。
"""
function _truncation_keep(v::AbstractVector{<:Real}, trunc::TruncationScheme)
    v2, err = truncate!(collect(v), trunc)
    return length(v2), err^2
end

"是否可能发生截断（用于跳过不必要的规范/搜索开销）。"
_possibly_truncating(::NoTruncation) = false
_possibly_truncating(::TruncationScheme) = true
