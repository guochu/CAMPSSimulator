# quantumcircuits/classical.jl — IfOp 执行（经典寄存器运行时已上移至 QuantumCircuits）
#
# ClassicalStore / reg_value / set_clbit! / eval_cond 由
# QuantumCircuits 统一提供（2026-10 上移，各后端共享）；本文件只保留
# CAMPS 态上的 IfOp 分支执行。

"""
    apply!(state, op::IfOp, store::ClassicalStore; trunc=DefaultTruncation) -> state

执行经典条件分支：按 `store` 求值 `op.cond`，执行 `op.then` 或
`op.otherwise` 子线路。需直接调用时请使用 `simulate`。
"""
function apply!(s::CAMPS, op::IfOp, store::ClassicalStore,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    branch = eval_cond(op.cond, store) ? op.then : op.otherwise
    branch === nothing && return s
    for o in branch.ops
        s = apply!(s, o, table; trunc=trunc, search=search, wmax=wmax)
    end
    return s
end

function apply!(s::CAMPS, op::IfOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    throw(ArgumentError("`apply!` on IfOp requires classical register values; use `simulate` instead"))
end
