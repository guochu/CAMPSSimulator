# gates.jl — 非 Clifford 门 / 截断语义
using CAMPSSimulator
using CAMPSSimulator: truncdim, NoTruncation
using Random, Test

@testset "非 Clifford 门" begin
    rng = MersenneTwister(7)

    @testset "无截断精确（Clifford + T/RX/CRZ 混合）" begin
        n = 6
        gates = Vector{Any}()
        for _ in 1:30
            r = rand(rng, 1:6)
            if r <= 2
                q = rand(rng, 1:n)
                g = r == 1 ? _H : (rand(rng, Bool) ? _T : _S)
                push!(gates, (g, [q]))
            elseif r == 3
                q = rand(rng, 1:n)
                th = rand(rng) * pi
                rx = [cos(th/2) -im*sin(th/2); -im*sin(th/2) cos(th/2)]
                push!(gates, (rx, [q]))
            elseif r == 4
                a, b = randperm(rng, n)[1:2]
                push!(gates, (_CX, [a, b]))
            elseif r == 5
                a, b = randperm(rng, n)[1:2]
                th = rand(rng) * pi
                crz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 exp(-im*th/2) 0; 0 0 0 exp(im*th/2)]
                push!(gates, (crz, [a, b]))        # 受控 Rz（控制=a，最高位）
            else
                q = rand(rng, 1:n)
                push!(gates, (_Z, [q]))
            end
        end
        ψ = CAMPSSimulator.zero_state(n)
        for (m, ps) in gates
            CAMPSSimulator.apply!(ψ, m, Tuple(ps); trunc=NoTruncation())
        end
        v = exact_simulate(n, gates)
        vs = CAMPSSimulator.statevector(ψ)
        @test fid(v, vs) > 1 - 1e-9
    end

    @testset "远距两比特门（SWAP 聚集）" begin
        n = 6
        ψ = CAMPSSimulator.zero_state(n)
        CAMPSSimulator.apply!(ψ, _H, (1,))
        CAMPSSimulator.apply!(ψ, _CX, (1, 6); trunc=NoTruncation())
        CAMPSSimulator.apply!(ψ, _CX, (6, 2); trunc=NoTruncation())
        # 手工构造参照（1↔6 纠缠后 6、2 纠缠）
        v = exact_simulate(n, [(_H, [1]), (_CX, [1, 6]), (_CX, [6, 2])])
        vs = CAMPSSimulator.statevector(ψ)
        @test fid(v, vs) > 1 - 1e-9
    end

    @testset "CAMPS 解纠缠搜索：截断误差不劣于普通截断" begin
        n = 6
        D = 2
        worse = 0.0
        for seed in 1:3
            rng = MersenneTwister(seed + 100)
            gates = Vector{Any}()
            for _ in 1:14
                r = rand(rng, 1:4)
                if r <= 2
                    a, b = randperm(rng, n)[1:2]
                    push!(gates, (_CX, [a, b]))
                elseif r == 3
                    a, b = randperm(rng, n)[1:2]
                    th = rand(rng) * pi
                    crz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 exp(-im*th/2) 0; 0 0 0 exp(im*th/2)]
                    push!(gates, (crz, [a, b]))
                else
                    q = rand(rng, 1:n)
                    push!(gates, (rand(rng, Bool) ? _T : _H, [q]))
                end
            end
            v = exact_simulate(n, gates)
            ψ0 = CAMPSSimulator.zero_state(n)
            for (m, ps) in gates
                CAMPSSimulator.apply!(ψ0, m, Tuple(ps); trunc=truncdim(D), search=false)
            end
            ψ1 = CAMPSSimulator.zero_state(n)
            for (m, ps) in gates
                CAMPSSimulator.apply!(ψ1, m, Tuple(ps); trunc=truncdim(D), search=true)
            end
            f0 = fid(v, CAMPSSimulator.statevector(ψ0))
            f1 = fid(v, CAMPSSimulator.statevector(ψ1))
            worse = max(worse, f0 - f1)   # 期望 ≤ 0（搜索至少不劣）
        end
        @test worse <= 1e-8
    end
end

@testset "一般多比特门（与 MPSSimulator 一致）" begin
    n = 5
    # Toffoli：QC 约定 positions[1]=最高位（第一个列为控制）。索引编码：
    # idx = ((a*2)+b)*2 + t（a=positions[1] 为最高位）。
    function ccx_msb()
        M = zeros(ComplexF64, 8, 8)
        for a in 0:1, b in 0:1, t in 0:1
            i = ((a * 2) + b) * 2 + t + 1
            j = ((a * 2) + b) * 2 + (t ⊻ (a & b)) + 1
            M[i, j] = 1
        end
        return M
    end
    CCX = ccx_msb()

    @testset "三比特门（相邻与非相邻）无截断精确" begin
        for ps in ([1, 2, 3], [1, 3, 5], [5, 2, 4])
            ψ = CAMPSSimulator.zero_state(n)
            CAMPSSimulator.apply!(ψ, _H, (ps[1],))
            CAMPSSimulator.apply!(ψ, CCX, Tuple(ps); trunc=NoTruncation())
            v = exact_simulate(n, [(_H, [ps[1]]), (CCX, ps)])
            vs = CAMPSSimulator.statevector(ψ)
            @test fid(v, vs) > 1 - 1e-8
            @test CAMPSSimulator.iscanonical(mps(ψ); atol=1e-8)
        end
    end

    @testset "截断下多比特门：键维封顶 + 规范 + 谱正确" begin
        D = 4
        rng = MersenneTwister(31)
        for _ in 1:10
            a, b, c = randperm(rng, n)[1:3]
            ψ = CAMPSSimulator.zero_state(n)
            CAMPSSimulator.apply!(ψ, _H, (a,))
            CAMPSSimulator.apply!(ψ, CCX, (a, b, c); trunc=truncdim(D))
            CAMPSSimulator.apply!(ψ, _CX, (b, c); trunc=truncdim(D))
            CAMPSSimulator.apply!(ψ, CCX, (c, a, b); trunc=truncdim(D))
            M = mps(ψ)
            @test maximum(CAMPSSimulator.bond_dimensions(M)) <= D + 1
            # MPS（魔法部分 φ）谱与真实 Schmidt 一致
            v2 = CAMPSSimulator._statevector_from_mps(copy(M))
            for bb in 1:(n-1)
                rt = filter(x -> x > 1e-12, svdvals(reshape(v2, 1 << bb, 1 << (n - bb))))
                st = filter(x -> x > 1e-9, M.svectors[bb+1])
                @test length(st) == length(rt)
                @test st ≈ rt atol = 1e-5 rtol = 1e-3
            end
        end
    end
end
