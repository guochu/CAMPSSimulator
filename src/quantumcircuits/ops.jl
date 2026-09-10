# quantumcircuits/ops.jl — IR 指令 → 核心原语（CAMPS）

function _check_gateop(op::Operation, n::Int)
    qs = qubits(op)
    length(unique(qs)) == length(qs) || throw(ArgumentError("duplicate qubits in $op"))
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("qubit index out of range [1, $n] in $op"))
    return qs
end

"""
    apply!(state::CAMPS, op, [table]; trunc, search, wmax) -> state

把 IR 指令 `op` 就地作用到 CAMPS 态上。

支持：`GateOp`（酉门）、`MeasOp`（坍缩，测量结果丢弃）、`ReinitOp`、
`BarrierOp`（无操作）、`BlockOp`（展开执行）。`IfOp` 需要经典寄存器
状态，请通过 `simulate` 执行。

**v1 限制**：`ChannelOp`（噪声信道）会转为混合态，本版本不支持并抛出
明确错误（纯态 Clifford 帧 + MPS 无法精确承载密度算符）。
"""
function apply!(s::CAMPS, op::GateOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    _check_gateop(op, length(s))
    return apply!(s, mat(op, table), Tuple(qubits(op)); trunc=trunc, search=search, wmax=wmax)
end

function apply!(s::CAMPS, op::ChannelOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    throw(ArgumentError("CAMPSSimulator v1 不支持 ChannelOp（噪声/混合态）：" *
                        "Clifford 帧 + 纯态 MPS 表示无法精确承载密度算符。"))
end

function apply!(s::CAMPS, op::ReinitOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    _check_gateop(op, length(s))
    for q in qubits(op)
        s = reset_qubit_zero!(s, q; trunc=trunc)
    end
    return s
end

apply!(s::CAMPS, ::BarrierOp, table::Union{Nothing,AbstractDict}=nothing;
       trunc::TruncationScheme=DefaultTruncation,
       search::Bool=DefaultSearchDisentangler,
       wmax::Int=DefaultConjWindow) = s

function apply!(s::CAMPS, op::MeasOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    _check_gateop(op, length(s))
    for q in qubits(op)
        measure!(s, q; trunc=trunc)
    end
    return s
end

function apply!(s::CAMPS, op::BlockOp, table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultTruncation,
                search::Bool=DefaultSearchDisentangler,
                wmax::Int=DefaultConjWindow)
    for mapped in unroll(op)
        s = apply!(s, mapped, table; trunc=trunc, search=search, wmax=wmax)
    end
    return s
end

"""
    reset_qubit_zero!(state::CAMPS, q; trunc=DefaultTruncation) -> state

把 qubit `q` 相干重置到 `|0⟩`（线性映射 `[[1,1],[0,0]]` + 归一化）。
"""
function reset_qubit_zero!(ψ::CAMPS, q::Int; trunc::TruncationScheme=DefaultTruncation)
    n = length(ψ)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    isempty(ψ.frame) || fold_frame!(ψ; trunc=trunc)
    m = eltype(ψ)[1 1; 0 0]
    _apply_single!(ψ.data, m, q)
    canonicalize!(ψ.data; trunc=NoTruncation(), normalize=true)
    return ψ
end
