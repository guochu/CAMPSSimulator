# quantumcircuits/simulate.jl — 线路演化（对齐 MPSSimulator，附加 search/wmax）

# ── 参数环境（沿用 MPSSimulator 语义）──────────────────────────────────────

"""
    _ParamEnv <: AbstractDict{Param,Float64}

模拟期参数环境：按 `parameters(c)` 顺序绑定值，提供 `Param → 序号` 索引。
"""
struct _ParamEnv <: AbstractDict{Param,Float64}
    vec::Vector{Float64}
    index::Dict{Param,Int}

    function _ParamEnv(vec::Vector{Float64}, index::Dict{Param,Int})
        length(vec) == length(index) ||
            throw(ArgumentError("parameter values and index length mismatch"))
        new(vec, index)
    end
end

_ParamEnv() = _ParamEnv(Float64[], Dict{Param,Int}())

Base.getindex(e::_ParamEnv, p::Param) = e.vec[e.index[p]]
Base.haskey(e::_ParamEnv, p::Param) = haskey(e.index, p)
Base.keys(e::_ParamEnv) = keys(e.index)
Base.length(e::_ParamEnv) = length(e.vec)
Base.iterate(e::_ParamEnv, state...) = iterate(pairs(e.index), state...)

function _param_env(c::Circuit, params::AbstractVector{<:Real})
    ps = parameters(c)
    length(params) == length(ps) ||
        throw(ArgumentError("circuit has $(length(ps)) symbolic parameter(s), got $(length(params)) values"))
    return _ParamEnv(Float64.(params), Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

function _param_env(c::Circuit, table::AbstractDict)
    ps = parameters(c)
    t = Dict{Param,Float64}()
    for (k, v) in table
        if k isa ParamVector
            for (i, vv) in enumerate(v)
                t[k[i]] = Float64(vv)
            end
        elseif k isa Param
            t[k] = Float64(v)
        elseif k isa Symbol
            t[Param(k)] = Float64(v)
        else
            throw(ArgumentError("invalid parameter key $(k)"))
        end
    end
    vec = Float64[get(t, p) do
                      throw(ArgumentError("parameter $(p.name) is not bound"))
                  end for p in ps]
    return _ParamEnv(vec, Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

_param_env(::Circuit, ::Nothing) = _ParamEnv()

# ── 演化 ─────────────────────────────────────────────────────────────────────

import QuantumCircuits.Interface: simulate, simulate!

"""
    simulate(c::Circuit, state::CAMPS; params=nothing, trunc=DefaultTruncation,
             search=true, wmax=8) -> state'

演化整条线路（非就地；`simulate!` 就地）。测量写回经典寄存器供 `IfOp` 使用。
"""
function simulate(c::Circuit, s::CAMPS; params=nothing,
                  trunc::TruncationScheme=DefaultTruncation,
                  search::Bool=DefaultSearchDisentangler,
                  wmax::Int=DefaultConjWindow)
    return _simulate_bound(c, copy(s), _param_env(c, params), trunc, search, wmax)
end

function simulate!(c::Circuit, s::CAMPS; params=nothing,
                   trunc::TruncationScheme=DefaultTruncation,
                   search::Bool=DefaultSearchDisentangler,
                   wmax::Int=DefaultConjWindow)
    return _simulate_bound!(c, s, _param_env(c, params), trunc, search, wmax)
end

function _simulate_bound(c::Circuit, s::CAMPS, env::_ParamEnv,
                         trunc::TruncationScheme, search::Bool, wmax::Int)
    return _simulate_bound!(c, copy(s), env, trunc, search, wmax, ClassicalStore(c))
end

function _simulate_bound!(c::Circuit, s::CAMPS, env::_ParamEnv,
                          trunc::TruncationScheme, search::Bool, wmax::Int)
    return _simulate_bound!(c, s, env, trunc, search, wmax, ClassicalStore(c))
end

function _simulate_bound!(c::Circuit, s::CAMPS, env::_ParamEnv,
                          trunc::TruncationScheme, search::Bool, wmax::Int,
                          store::ClassicalStore)
    for op in c.ops
        s = _apply_with_store!(s, op, store, env, trunc, search, wmax)
    end
    return s
end

function _apply_with_store!(s::CAMPS, op::Operation, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme, search::Bool, wmax::Int)
    return apply!(s, op, env; trunc=trunc, search=search, wmax=wmax)
end

function _apply_with_store!(s::CAMPS, op::MeasOp, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme, search::Bool, wmax::Int)
    for (q, cb) in zip(op.qubits, op.clbits)
        outcome = measure!(s, q; trunc=trunc)
        _set_clbit!(store, cb, outcome)
    end
    return s
end

function _apply_with_store!(s::CAMPS, op::IfOp, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme, search::Bool, wmax::Int)
    return apply!(s, op, store, env; trunc=trunc, search=search, wmax=wmax)
end
