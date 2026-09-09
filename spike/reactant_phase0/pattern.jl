# Direct evidence: what does the bisect's monomials divergence actually LOOK like,
# and does composition (stage_u -> monomials) break something that each stage
# does not break alone?
using ACEpotentials, LinearAlgebra, Random, StaticArrays, Printf
import Polynomials4ML as P4ML
using Reactant
Base.Float64(x::Reactant.TracedRNumber{Bool}) = ifelse(x, 1.0, 0.0)
Reactant.set_default_backend("cpu")
M = ACEpotentials.Models
const NGRID = 32768; const MAX_ATOMS = 64; const MAX_EDGES = 2048
segment_sum(rows, idx, n) = permutedims(idx .== permutedims(1:n)) * rows
monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])
function hermite(r, grid0, h, n, values, derivs)
    x = clamp.((r .- grid0) ./ h, 0.0, n - 1.0 - 1e-9); i = Int32.(floor.(x)); t = x .- i
    y0, y1 = values[i .+ 1, :], values[i .+ 2, :]
    d0, d1 = derivs[i .+ 1, :] .* h, derivs[i .+ 2, :] .* h
    h00 = @. (1+2t)*(1-t)^2; h10 = @. t*(1-t)^2; h01 = @. t^2*(3-2t); h11 = @. t^2*(t-1)
    @. h00*y0 + h10*d0 + h01*y1 + h11*d1
end
include(joinpath(@__DIR__, "_ace_setup.jl"))

function stage_u(p, c, n, em)
    v = p[:, n] .- p[:, c]
    em_r = permutedims(Float64.(em))
    v = v .* em_r .+ [1.0, 0.0, 0.0] .* (1.0 .- em_r)
    l = sqrt.(sum(abs2, v; dims=1))[1, :]
    permutedims(v ./ permutedims(l))
end
stage_mono(p, c, n, em) = monomials(stage_u(p, c, n, em), EXPONENTS[LMAX+1])

rargs = (Reactant.to_rarray(pos_pad), Reactant.to_rarray(centers),
         Reactant.to_rarray(neighbors), Reactant.to_rarray(edge_mask))
eargs = (pos_pad, centers, neighbors, edge_mask)

u_e = stage_u(eargs...)
u_c = Array((@compile stage_u(rargs...))(rargs...))
@printf("stage_u   max|c-e| = %.3e\n", maximum(abs.(u_c .- u_e)))

m_e = stage_mono(eargs...)
m_c = Array((@compile stage_mono(rargs...))(rargs...))
@printf("stage_mono max|c-e| = %.3e\n\n", maximum(abs.(m_c .- m_e)))

# feed the CONCRETE compiled u into a compiled monomials: does it still break?
g(u) = monomials(u, EXPONENTS[LMAX+1])
g_c = Array((@compile g(Reactant.to_rarray(u_c)))(Reactant.to_rarray(u_c)))
@printf("monomials(concrete u)  max|c-e| = %.3e\n\n", maximum(abs.(g_c .- monomials(u_c, EXPONENTS[LMAX+1]))))

println("exponents: ", EXPONENTS[LMAX+1])
println("\nfirst 4 live edges, eager:");    display(round.(m_e[1:4, :], digits=4))
println("\nfirst 4 live edges, compiled:"); display(round.(m_c[1:4, :], digits=4))
println("\nu for those edges:");            display(round.(u_e[1:4, :], digits=4))
