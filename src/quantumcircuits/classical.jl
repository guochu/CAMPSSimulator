# quantumcircuits/classical.jl — 经典寄存器运行时 + IfOp（与 MPSSimulator 一致）

"""
模拟期的经典寄存器状态（`CReg` → 整数值，位序与 `ClbitRef.index` 一致）。
"""
mutable struct ClassicalStore
    regs::Vector{CReg}
    values::Vector{Int}
end

ClassicalStore(c::Circuit) = ClassicalStore(copy(c.cregs), zeros(Int, length(c.cregs)))

function _reg_index(store::ClassicalStore, r::CReg)
    idx = findfirst(==(r), store.regs)
    idx === nothing && throw(ArgumentError("register $(r.name) is not part of this simulation"))
    return idx
end

_reg_value(store::ClassicalStore, r::CReg) = store.values[_reg_index(store, r)]

function _set_clbit!(store::ClassicalStore, cb::ClbitRef, b::Int)
    idx = _reg_index(store, cb.reg)
    if b == 0
        store.values[idx] &= ~(1 << (cb.index - 1))
    else
        store.values[idx] |= 1 << (cb.index - 1)
    end
    return store
end

function _eval_cond(cond::Cond, store::ClassicalStore)
    v = _reg_value(store, cond.reg)
    v = cond.bit === nothing ? v : (v >> (cond.bit - 1)) & 1
    op = cond.op
    return op === :(==) ? v == cond.value :
           op === :(≠) ? v != cond.value :
           op === :(≥) ? v >= cond.value :
           op === :(≤) ? v <= cond.value :
           throw(ArgumentError("unsupported condition operator $(op)"))
end

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
    branch = _eval_cond(op.cond, store) ? op.then : op.otherwise
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
