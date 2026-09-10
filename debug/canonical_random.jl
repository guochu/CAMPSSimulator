# debug/canonical_random.jl
#
# 目的：随机生成量子线路（单比特 / 两比特门，Clifford 进帧、非 Clifford 经帧
# 共轭或折叠），演化结束后检查：
#   1. MPS 是否保持规范形式（`iscanonical`），并输出逐站点右正则偏差 / 逐键
#      transfer 偏差定位失败来源；
#   2. 态范数 ≈ 1、键维不超过截断上限；
#   3. 无截断时与精确态矢的保真度（验证演化本身正确）。
#
# 运行（沙箱配方）：
#   JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH="/tmp/jdepot:/home/guochu/.julia" \
#     julia --project=/tmp/vqc_env debug/canonical_random.jl

using CAMPSSimulator
using CAMPSSimulator: mps, iscanonical, truncdim, NoTruncation
using LinearAlgebra, Random

# ── 门库（QC 约定：positions[1] = 最高位）──────────────────────────────────
const _H = ComplexF64[1 1; 1 -1] ./ sqrt(2)
const _S = ComplexF64[1 0; 0 im]
const _T = ComplexF64[1 0; 0 exp(im * pi / 4)]
const _X = ComplexF64[0 1; 1 0]
const _CX = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
const _CZ = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
const _SWAP = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
_crz(th) = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 exp(-im * th / 2) 0; 0 0 0 exp(im * th / 2)]
_rx(th) = [cos(th / 2) -im * sin(th / 2); -im * sin(th / 2) cos(th / 2)]

# ── 随机线路生成 ───────────────────────────────────────────────────────────
function random_gates(rng, n, nop)
    gates = Vector{Any}()
    for _ in 1:nop
        r = rand(rng, 1:8)
        if r == 1
            push!(gates, (_H, [rand(rng, 1:n)]))
        elseif r == 2
            push!(gates, (_S, [rand(rng, 1:n)]))
        elseif r == 3
            push!(gates, (_T, [rand(rng, 1:n)]))
        elseif r == 4
            push!(gates, (_rx(rand(rng) * pi), [rand(rng, 1:n)]))
        elseif r == 5
            a, b = randperm(rng, n)[1:2]
            push!(gates, (_CX, [a, b]))
        elseif r == 6
            a, b = randperm(rng, n)[1:2]
            push!(gates, (_CZ, [a, b]))
        elseif r == 7
            a, b = randperm(rng, n)[1:2]
            push!(gates, (_crz(rand(rng) * pi), [a, b]))
        else
            a, b = randperm(rng, n)[1:2]
            push!(gates, (_SWAP, [a, b]))
        end
    end
    return gates
end

# ── 规范残差诊断 ───────────────────────────────────────────────────────────
function diagnostics(ψ::CAMPS)
    M = mps(ψ)
    n = length(ψ)
    rowdev = 0.0
    rowsite = 0
    for k in 1:n
        A = M[k]
        Dl, d, Dr = size(A)
        X = reshape(A, Dl, d * Dr)
        dev = maximum(abs.(X * X' - Matrix{Float64}(I, Dl, Dl)))
        if dev > rowdev
            rowdev = dev
            rowsite = k
        end
    end
    trdev = 0.0
    trbond = 0
    E = fill(1.0, 1, 1)
    for k in 1:(n-1)
        E = CAMPSSimulator._transfer(E, M[k], M[k])
        D2 = M.svectors[k+1] .^ 2
        dev = size(E, 1) == length(D2) ? maximum(abs.(E - Diagonal(D2))) : Inf
        if dev > trdev
            trdev = dev
            trbond = k
        end
    end
    return rowdev, rowsite, trdev, trbond
end

function check_canonical(ψ::CAMPS; atol::Real=1e-9, label::String="")
    M = mps(ψ)
    normv = norm(CAMPSSimulator._statevector_from_mps(copy(M)))
    ok = iscanonical(M; atol=atol)
    rowdev, rowsite, trdev, trbond = diagnostics(ψ)
    println(rpad("[$label]", 26),
            " norm=", round(normv, digits=8),
            " maxbond=", maximum(CAMPSSimulator.bond_dimensions(M)),
            " frame=", length(CAMPSSimulator.frame(ψ).gates),
            " canonical(atol=$atol)=", ok,
            ok ? "" : "  [行残差=$rowdev@site$rowsite, transfer残差=$trdev@bond$trbond]")
    return ok
end

# ── 精确态矢参照（无截断时用）──────────────────────────────────────────────
function qc_apply_vec(v, m, positions)
    n = round(Int, log2(length(v)))
    k = length(positions)
    dim = 1 << n
    positions = collect(Int, positions)
    mbit = [k - j for j in 1:k]
    vbit = [p - 1 for p in positions]
    out = zeros(ComplexF64, dim)
    for i in 0:(dim-1)
        ai = 0
        for t in 1:k
            ai |= ((i >> vbit[t]) & 1) << mbit[t]
        end
        acc = 0.0im
        for jj in 0:(dim-1)
            same = true
            for b in 0:(n-1)
                b in vbit && continue
                (((jj >> b) & 1) != ((i >> b) & 1)) && (same = false; break)
            end
            same || continue
            aj = 0
            for t in 1:k
                aj |= ((jj >> vbit[t]) & 1) << mbit[t]
            end
            acc += m[ai+1, aj+1] * v[jj+1]
        end
        out[i+1] = acc
    end
    return out
end

exact_sim(n, gates) = (v = zeros(ComplexF64, 1 << n); v[1] = 1;
                       foreach(g -> (v = qc_apply_vec(v, g[1], g[2])), gates); v)

function main(; nseeds::Int=6, n::Int=8, nop::Int=80)
    allok = true
    for seed in 1:nseeds
        rng = MersenneTwister(seed)
        gates = random_gates(rng, n, nop)
        # —— 模式 A：无截断（应完全精确 + 规范）——
        ψA = CAMPSSimulator.zero_state(n)
        for (m, ps) in gates
            CAMPSSimulator.apply!(ψA, m, Tuple(ps); trunc=NoTruncation())
        end
        okA = check_canonical(ψA; atol=1e-9, label="seed$seed · 无截断")
        v = exact_sim(n, gates)
        vs = CAMPSSimulator.statevector(ψA)
        fidv = abs(dot(v, vs))^2 / (norm(v)^2 * norm(vs)^2)
        println("     精确保真度 = ", fidv)
        allok &= okA && (fidv > 1 - 1e-8)

        # —— 模式 B：截断（应规范、键维封顶）——
        D = 4
        ψB = CAMPSSimulator.zero_state(n)
        for (m, ps) in gates
            CAMPSSimulator.apply!(ψB, m, Tuple(ps); trunc=truncdim(D))
        end
        okB = check_canonical(ψB; atol=1e-8, label="seed$seed · 截断 D=$D")
        nb = maximum(CAMPSSimulator.bond_dimensions(mps(ψB)))
        allok &= okB && nb <= D + 1
    end
    println(allok ? "\n✅ 全部通过：演化结束后 MPS 均保持规范形式" :
                    "\n❌ 存在失败，请查看上方残差定位")
    return allok
end

allok = main()
exit(allok ? 0 : 1)
