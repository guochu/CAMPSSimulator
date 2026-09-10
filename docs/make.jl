using Documenter
using CAMPSSimulator

makedocs(
	sitename = "CAMPSSimulator.jl",
	authors = "Guo Chu",
	pages = ["Home" => "index.md",
	         "states.md",
	         "policies.md",
	         "gates.md",
	         "measure.md",
	         "backend.md",
	         "tests.md"],
	format = Documenter.HTML(
	    prettyurls = get(ENV, "CI", nothing) == "true"
	)
)
