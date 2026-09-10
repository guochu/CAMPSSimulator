# measure.jl — 测量 / 坍缩 / 边缘概率
using CAMPSSimulator
using Random, Test

@testset "measure" begin
    rng = MersenneTwister(11)

    @testset "Bell 态单比特边缘概率" begin
        ψ = CAMPSSimulator.zero_state(2)
        CAMPSSimulator.apply!(ψ, _H, (1,))
        CAMPSSimulator.apply!(ψ, _CX, (1, 2))
        p = CAMPSSimulator.marginal_probabilities(ψ, 1)
        @test p[1] ≈ 0.5 atol = 1e-9
        @test p[2] ≈ 0.5 atol = 1e-9
    end

    @testset "Bell 测量坍缩：关联性" begin
        agree = 0
        total = 200
        Random.seed!(5)
        for _ in 1:total
            ψ = CAMPSSimulator.zero_state(2)
            CAMPSSimulator.apply!(ψ, _H, (1,))
            CAMPSSimulator.apply!(ψ, _CX, (1, 2))
            b = CAMPSSimulator.measure!(ψ, 2)
            c = CAMPSSimulator.measure!(ψ, 1)
            agree += (b == c)
        end
        @test agree / total > 0.95
    end

    @testset "GHZ 中途测量" begin
        n = 4
        # 精确分布参照：p(0000)=p(1111)=1/2
        counts = Dict{Int,Int}()
        Random.seed!(9)
        for _ in 1:400
            ψ = CAMPSSimulator.zero_state(n)
            CAMPSSimulator.apply!(ψ, _H, (1,))
            for k in 2:n
                CAMPSSimulator.apply!(ψ, _CX, (1, k))
            end
            o = CAMPSSimulator.measure!(ψ)          # 全比特
            counts[o] = get(counts, o, 0) + 1
        end
        @test get(counts, 0, 0) + get(counts, (1 << n) - 1, 0) > 380
    end

    @testset "measure 非就地协议" begin
        ψ = CAMPSSimulator.zero_state(1)
        CAMPSSimulator.apply!(ψ, _H, (1,))
        ψ2, b, p = CAMPSSimulator.measure(ψ, 1)
        @test p ≈ 0.5 atol = 1e-9
        @test ψ2 isa CAMPSSimulator.CAMPS
        # 原态未被就地坍缩（复制语义）
        @test CAMPSSimulator.marginal_probabilities(ψ, 1) ≈ [0.5, 0.5] atol = 1e-9
    end
end
