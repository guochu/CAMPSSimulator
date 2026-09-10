# draw.jl — 帧可视化 / 命名 / 统计 单元测试
# 覆盖 src/cliffframe.jl 的 draw_frame、frame_gate_name、_cnot_ctl、frame_stats，
# 固化 QuantumCircuits.draw 风格的画图行为（含 CNOT 的 ●/X 与纵向连接棒 │）。
using CAMPSSimulator
using CAMPSSimulator: LocalOp, CliffordFrame, draw_frame, frame_gate_name, frame_stats, _cnot_ctl
using Test

# 常用矩阵（内部约定：LocalOp.pos 升序 + 小端序，bit0 = 最小站点）
const _CX_LOW = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]   # CNOT 控制=bit0
# 控制=bit1 的 CNOT 即 util.jl 的 _CX（复用）
const _SWAP_M  = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
const _ISWAP_M = ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]
const _SQRTX_M = (ComplexF64[1 0; 0 1] - im * ComplexF64[0 1; 1 0]) / sqrt(2)

"渲染帧并切行（供结构断言使用）。"
function render(fr::CliffordFrame, n::Int)
    io = IOBuffer()
    draw_frame(io, fr, n)
    split(String(take!(io)), '\n'; keepempty=false)
end

"符号所在行的字符位置（1-based，非字节索引——行内含多字节 Unicode 字符）。"
function cpos(s::AbstractString, ch::Char)
    i = findfirst(isequal(ch), s)
    i === nothing && return nothing
    return length(String(s[1:prevind(s, i)])) + 1
end

@testset "frame naming / stats" begin
    @testset "frame_gate_name 识别（含任意控制端朝向）" begin
        @test frame_gate_name(LocalOp([2], _H)) == "H"
        @test frame_gate_name(LocalOp([1], _S)) == "S"
        @test frame_gate_name(LocalOp([3], _SQRTX_M)) == "√X"
        @test frame_gate_name(LocalOp([2, 5], _CX_LOW)) == "CX"   # 控制=bit0
        @test frame_gate_name(LocalOp([2, 5], _CX)) == "CX"       # 控制=bit1
        @test frame_gate_name(LocalOp([1, 2], _CZ)) == "CZ"
        @test frame_gate_name(LocalOp([1, 2], _SWAP_M)) == "SWAP"
        @test frame_gate_name(LocalOp([1, 2], _ISWAP_M)) == "iSWAP"
        @test frame_gate_name(LocalOp([1, 2, 3], Matrix{ComplexF64}(I, 8, 8))) == "?3"
    end

    @testset "_cnot_ctl 控制端判定" begin
        @test _cnot_ctl(_CX_LOW) == 1
        @test _cnot_ctl(_CX) == 2
        @test _cnot_ctl(_CZ) == 0
        @test _cnot_ctl(_SWAP_M) == 0
        @test _cnot_ctl(_ISWAP_M) == 0
    end

    @testset "frame_stats 计数" begin
        fr = CliffordFrame([
            LocalOp([1], _H),
            LocalOp([2], _S),
            LocalOp([3], _SQRTX_M),
            LocalOp([1, 2], _CX_LOW),
            LocalOp([3, 4], _CX),      # 反向控制端也计入 CX
        ])
        st = frame_stats(fr)
        @test st["total"] == 5
        @test st["H"] == 1
        @test st["S"] == 1
        @test st["√X"] == 1
        @test st["CX"] == 2
    end
end

@testset "draw_frame（QC 风格：含跨线连接行）" begin
    @testset "CNOT 控制=bit0：●/│/X 同列对齐，且上下相邻" begin
        lines = render(CliffordFrame([LocalOp([1, 2], _CX_LOW)]), 2)
        @test length(lines) == 3                 # 2 根线 + 1 行中间连接
        i1 = findfirst(l -> startswith(l, "q[1]:"), lines)
        i2 = findfirst(l -> startswith(l, "q[2]:"), lines)
        @test i1 !== nothing && i2 == i1 + 2      # q[2] 行紧跟连接行之后
        @test occursin("●", lines[i1])            # 控制端在上
        @test occursin("X", lines[i2])            # 目标端在下
        @test occursin("│", lines[i1+1])          # 控制与目标之间确有连线
        # 三者对齐在同一可视列
        @test cpos(lines[i1], '●') == cpos(lines[i1+1], '│') == cpos(lines[i2], 'X')
    end

    @testset "CNOT 控制=bit1：X 在上、● 在下" begin
        lines = render(CliffordFrame([LocalOp([1, 2], _CX)]), 2)
        i1 = findfirst(l -> startswith(l, "q[1]:"), lines)
        @test occursin("X", lines[i1])
        @test occursin("●", lines[i1+2])
        @test cpos(lines[i1], 'X') == cpos(lines[i1+1], '│') == cpos(lines[i1+2], '●')
    end

    @testset "跨线 CNOT（q1→q3）：中间线贯通两根连接棒" begin
        lines = render(CliffordFrame([LocalOp([1, 3], _CX_LOW)]), 3)
        @test length(lines) == 5                 # 3 根线 + 2 行连接
        i1 = findfirst(l -> startswith(l, "q[1]:"), lines)
        i2 = findfirst(l -> startswith(l, "q[2]:"), lines)
        i3 = findfirst(l -> startswith(l, "q[3]:"), lines)
        @test occursin("●", lines[i1])
        @test occursin("X", lines[i3])
        @test !occursin("●", lines[i2]) && !occursin("X", lines[i2])  # 中间线无门
        @test occursin("│", lines[i1+1]) && occursin("│", lines[i2+1])
        col = cpos(lines[i1], '●')
        @test cpos(lines[i1+1], '│') == col
        @test cpos(lines[i2+1], '│') == col
        @test cpos(lines[i3], 'X') == col
    end

    @testset "单比特门只占一行；SWAP/CZ 记号" begin
        # H@q2：仅 q[2] 行出现 H，其余行全为水平线
        lines = render(CliffordFrame([LocalOp([2], _H)]), 3)
        @test length(lines) == 3
        i2 = findfirst(l -> startswith(l, "q[2]:"), lines)
        @test occursin("H", lines[i2])
        @test !any(occursin("H", l) for l in lines if !startswith(l, "q[2]:"))
        # SWAP：两端 ✕ + 连接棒
        sw = render(CliffordFrame([LocalOp([1, 2], _SWAP_M)]), 2)
        @test count(l -> occursin("✕", l), sw) == 2
        @test any(l -> occursin("│", l), sw)
        # CZ：●/Z
        cz = render(CliffordFrame([LocalOp([1, 2], _CZ)]), 2)
        @test any(l -> occursin("●", l), cz)
        @test any(l -> occursin("Z", l), cz)
    end

    @testset "golden：H@q1; CX(q1→q2); S@q2 三列排布（逐字锁定）" begin
        fr = CliffordFrame([LocalOp([1], _H), LocalOp([1, 2], _CX_LOW), LocalOp([2], _S)])
        lines = render(fr, 2)
        @test lines == [
            "q[1]: ─H──●────",   # H 列 | CX 列(●) | S 列(空)
            "          │    ",   # 连接行：│ 对齐于 ●/X 列（画风同 QuantumCircuits）
            "q[2]: ────X──S─",   # CX 列(X) 与 S 分列
        ]
        # 强调：q[2] 行 X 与 q[1] 行 ● 位于同一可视列
        @test cpos(lines[3], 'X') == cpos(lines[1], '●')
        @test cpos(lines[3], 'S') > cpos(lines[3], 'X')   # S 画在 CX 之后的分列
    end
end
