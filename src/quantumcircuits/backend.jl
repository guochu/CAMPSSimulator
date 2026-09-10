# quantumcircuits/backend.jl — QuantumCircuits.Interface 后端（CAMPS）

using QuantumCircuits.Interface: Backend, SimResult
import QuantumCircuits.Interface: simulate, supports

"""
    CAMPSBackend(; dtype=ComplexF64, trunc=DefaultTruncation,
                 search=true, wmax=8) <: QuantumCircuits.Interface.Backend

Clifford 增广 MPS（CAMPS）后端：把 `Circuit` 演化到 [`CAMPS`](@ref) 态上。

* `shots = 0`：强模拟，`SimResult.state` 给出终态（`CAMPS`）；
* `shots > 0`：逐 shot 采样（每次从 `|0…0⟩` 重新演化，测量 / 经典反馈
  写回经典寄存器），`SimResult.counts` 按 `counts_key` 约定聚合；
* `trunc`：MPS 截断方案；`search`：是否启用两比特截断的 Clifford
  解纠缠搜索（CAMPS 增强）；`wmax`：帧共轭最大窗（超出先折叠）；
* 能力：`:statevector`、`:mid_measure`（**无** `:noise`——信道指令
  不被支持）。
"""
struct CAMPSBackend{T<:TruncationScheme} <: Backend
    dtype::DataType
    trunc::T
    search::Bool
    wmax::Int
end
CAMPSBackend(; dtype::DataType=ComplexF64,
             trunc::TruncationScheme=DefaultTruncation,
             search::Bool=DefaultSearchDisentangler,
             wmax::Int=DefaultConjWindow) =
    CAMPSBackend{typeof(trunc)}(dtype, trunc, search, wmax)

supports(::CAMPSBackend, cap::Symbol) = cap in (:statevector, :mid_measure)

# 线路中含 MeasOp 的经典寄存器
function _measured_cregs(c::Circuit)
    out = CReg[]
    for op in c.ops
        op isa MeasOp || continue
        for cb in op.clbits
            cb.reg in out || push!(out, cb.reg)
        end
    end
    return out
end

function simulate(c::Circuit, b::CAMPSBackend;
                  shots::Int=0,
                  seed::Union{Nothing,Integer}=nothing,
                  kwargs...)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    measured = _measured_cregs(c)
    if shots <= 0
        store = ClassicalStore(c)
        state = _simulate_bound!(c, zero_state(b.dtype, c.n), _ParamEnv(),
                                 b.trunc, b.search, b.wmax, store)
        return SimResult(nothing, state)
    end

    counts = Dict{String,Int}()
    seed === nothing || Random.seed!(Int(seed))            # 保证同种子同结果（measure! 用全局 RNG）
    for _ in 1:shots
        store = ClassicalStore(c)
        s = _simulate_bound!(c, zero_state(b.dtype, c.n), _ParamEnv(),
                             b.trunc, b.search, b.wmax, store)
        outcome = Dict{ClbitRef,Int}()
        for reg in measured
            v = _reg_value(store, reg)
            for i in 1:reg.n
                outcome[ClbitRef(reg, i)] = (v >> (i - 1)) & 1
            end
        end
        k = counts_key(c, outcome)
        counts[k] = get(counts, k, 0) + 1
    end
    return SimResult(; counts=counts,
                     metadata=Dict{Symbol,Any}(:shots => shots))
end
