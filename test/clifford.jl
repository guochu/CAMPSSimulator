# clifford.jl — Clifford 主导电路：纯 Clifford 电路零截断、GHZ 构造
using CAMPSSimulator
using CAMPSSimulator: truncdim
using Random, Test

@testset "clifford frame" begin
    rng = MersenneTwister(42)

    @testset "纯 Clifford 线路精确（MPS 完全不截断）" begin
        n = 8
        ψ = CAMPSSimulator.zero_state(n)
        gates = Vector{Any}()
        # 随机 Clifford 电路（H / S / CX / CZ / SWAP）
        for _ in 1:60
            g = rand(rng, 1:5)
            if g == 1
                q = rand(rng, 1:n)
                push!(gates, (_H, [q])); CAMPSSimulator.apply!(ψ, _H, (q,))
            elseif g == 2
                q = rand(rng, 1:n)
                push!(gates, (_S, [q])); CAMPSSimulator.apply!(ψ, _S, (q,))
            elseif g == 3
                a, b = randperm(rng, n)[1:2]
                push!(gates, (_CX, [a, b])); CAMPSSimulator.apply!(ψ, _CX, (a, b))
            elseif g == 4
                a, b = randperm(rng, n)[1:2]
                push!(gates, (_CZ, [a, b])); CAMPSSimulator.apply!(ψ, _CZ, (a, b))
            else
                a, b = randperm(rng, n)[1:2]
                sw = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
                push!(gates, (sw, [a, b])); CAMPSSimulator.apply!(ψ, sw, (a, b))
            end
        end
        # 极端小键维也不影响：Clifford 门全在帧里，MPS 从未截断
        @test CAMPSSimulator.max_bond_dimension(CAMPSSimulator.mps(ψ)) == 1
        v = exact_simulate(n, gates)
        vs = CAMPSSimulator.statevector(ψ)
        @test fid(v, vs) ≈ 1.0 atol = 1e-9
        @test !isempty(CAMPSSimulator.frame(ψ).gates)
    end

    @testset "GHZ 态构造（幅度 + 期望）" begin
        n = 6
        ψ = CAMPSSimulator.zero_state(n)
        CAMPSSimulator.apply!(ψ, _H, (1,))
        for k in 2:n
            CAMPSSimulator.apply!(ψ, _CX, (1, k))   # 控制=q1
        end
        @test CAMPSSimulator.max_bond_dimension(CAMPSSimulator.mps(ψ)) == 1
        # 幅度：只有 |00..0⟩ 与 |11..1⟩
        a0 = CAMPSSimulator.amplitude(ψ, zeros(Int, n))
        a1 = CAMPSSimulator.amplitude(ψ, ones(Int, n))
        @test abs(a0) ≈ 1 / sqrt(2) atol = 1e-9
        @test abs(a1) ≈ 1 / sqrt(2) atol = 1e-9
        # 单比特观测量
        @test CAMPSSimulator.expectation(ψ, _Z, 1) ≈ 0.0 atol = 1e-9
        @test CAMPSSimulator.expectation(ψ, _Z, n) ≈ 0.0 atol = 1e-9
        # 两比特 <Z1 Zn> = 1（帧内多观测量张量积期望）
        @test CAMPSSimulator.expectation(ψ, [1 => _Z, n => _Z]) ≈ 1.0 atol = 1e-8
    end

    @testset "Clifford 判定" begin
        @test CAMPSSimulator.is_clifford(_H)
        @test CAMPSSimulator.is_clifford(_S)
        @test CAMPSSimulator.is_clifford(_X)
        @test CAMPSSimulator.is_clifford(_CX)
        @test CAMPSSimulator.is_clifford(_CZ)
        @test !CAMPSSimulator.is_clifford(_T)
        # 任意相位
        @test CAMPSSimulator.is_clifford(exp(im * 0.3) * _CX)
    end
end
