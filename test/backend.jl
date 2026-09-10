# backend.jl — QuantumCircuits.Interface 后端集成
using CAMPSSimulator
using QuantumCircuits
using QuantumCircuits: ClbitRef, CReg
using CAMPSSimulator: CAMPSBackend
using Random, Test

@testset "backend" begin
    @testset "强模拟：state 为 CAMPS 且与手动一致" begin
        c = Circuit(3)
        push!(c, H(1))
        push!(c, CX(1, 2))
        push!(c, RZ(0.7, 2))
        res = simulate(c, CAMPSBackend(; trunc=truncdim(8)))
        @test res.state isa CAMPSSimulator.CAMPS
        ψ = CAMPSSimulator.zero_state(3)
        for op in c.ops
            CAMPSSimulator.apply!(ψ, op; trunc=truncdim(8))
        end
        @test fid(CAMPSSimulator.statevector(res.state), CAMPSSimulator.statevector(ψ)) > 1 - 1e-9
    end

    @testset "弱模拟：Bell 测量计数" begin
        c = Circuit(2)
        push!(c, H(1))
        push!(c, CX(1, 2))
        push!(c, measure(1, c.cregs[1][1]))
        push!(c, measure(2, c.cregs[1][2]))
        res = simulate(c, CAMPSBackend(); shots=2000, seed=2024)
        @test res.counts !== nothing
        @test sum(values(res.counts)) == 2000
        # 只应有 00 或 11
        @test all(k -> k == "c:00" || k == "c:11", keys(res.counts))
        # 同一 seed 可复现
        res2 = simulate(c, CAMPSBackend(); shots=2000, seed=2024)
        @test res2.counts == res.counts
    end

    @testset "能力查询" begin
        b = CAMPSBackend()
        @test supports(b, :statevector)
        @test supports(b, :mid_measure)
        @test !supports(b, :noise)
    end

    @testset "信道指令明确报错" begin
        c = Circuit(1)
        push!(c, Depolarizing(1, 0.1))
        @test_throws ArgumentError simulate(c, CAMPSBackend())
    end
end
