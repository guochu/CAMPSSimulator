using Test
include("util.jl")

@testset "CAMPSSimulator" begin
    include("clifford.jl")
    include("draw.jl")
    include("gates.jl")
    include("stability.jl")
    include("measure.jl")
    include("backend.jl")
end
