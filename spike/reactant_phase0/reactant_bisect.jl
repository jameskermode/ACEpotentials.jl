# Bisect the Reactant compiled-vs-eager divergence in the ACE export.
#
# Two hypotheses already eliminated (see mask_path_test.jl, ylm_conditioning.jl):
#   - the monomial->Ylm least-squares map is well conditioned (~1e-15)
#   - the masked and unmasked code paths agree EXACTLY in eager Julia
# So the divergence is introduced by compilation. This compiles each
# intermediate separately and reports where compiled first departs from eager.
#
# SPIKE CODE.

using ACEpotentials, LinearAlgebra, Random, StaticArrays, Printf
import Polynomials4ML as P4ML
using Reactant
Base.Float64(x::Reactant.TracedRNumber{Bool}) = ifelse(x, 1.0, 0.0)
Reactant.set_default_backend("cpu")
M = ACEpotentials.Models
println("Reactant ", pkgversion(Reactant), "  julia ", VERSION)

const NGRID = 32768
const MAX_ATOMS = 64
const MAX_EDGES = 2048

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
    x = clamp.((r .- grid0) ./ h, 0.0, n - 1.0 - 1e-9)
    i = Int32.(floor.(x)); t = x .- i
    y0, y1 = values[i .+ 1, :], values[i .+ 2, :]
    d0, d1 = derivs[i .+ 1, :] .* h, derivs[i .+ 2, :] .* h
    h00 = @. (1 + 2t) * (1 - t)^2; h10 = @. t * (1 - t)^2
    h01 = @. t^2 * (3 - 2t);       h11 = @. t^2 * (t - 1)
    @. h00 * y0 + h10 * d0 + h01 * y1 + h11 * d1
end

include(joinpath(@__DIR__, "_ace_setup.jl"))    # builds model, TABLES, cluster

# ---- stagewise functions of (positions, centers, neighbors, em) --------------
function stage_vectors(positions, centers, neighbors, em)
    v = positions[:, neighbors] .- positions[:, centers]
    em_r = permutedims(Float64.(em))
    v .* em_r .+ [1.0, 0.0, 0.0] .* (1.0 .- em_r)
end
stage_lengths(p,c,n,em) = sqrt.(sum(abs2, stage_vectors(p,c,n,em); dims=1))[1,:]
function stage_u(p,c,n,em)
    v = stage_vectors(p,c,n,em); l = sqrt.(sum(abs2, v; dims=1))[1,:]
    permutedims(v ./ permutedims(l))
end
stage_mono(p,c,n,em) = monomials(stage_u(p,c,n,em), EXPONENTS[LMAX+1])
function stage_ylm(p,c,n,em)
    u = stage_u(p,c,n,em)
    reduce(hcat, [monomials(u, EXPONENTS[l+1]) * Reactant.Ops.constant(YLM_MAPS[l+1])
                  for l in 0:LMAX])
end
stage_rnl(p,c,n,em) = hermite(stage_lengths(p,c,n,em), HR[1], HR[2], HR[3],
                              Reactant.Ops.constant(TABLES.rnl),
                              Reactant.Ops.constant(TABLES.drnl))
function stage_edge_a(p,c,n,em)
    ea = stage_rnl(p,c,n,em)[:, A_NL] .* stage_ylm(p,c,n,em)[:, A_LM]
    Float64.(em) .* ea
end
stage_A(p,c,n,em) = segment_sum(stage_edge_a(p,c,n,em), c, MAX_ATOMS)
function stage_site(p,c,n,em)
    a = stage_A(p,c,n,em)
    aa = reduce(hcat, [reduce(.*, [a[:, spec[:,k]] for k in 1:size(spec,2)])
                       for spec in AA_SPECS])
    (aa * Reactant.Ops.constant(TABLES.a2b)') * Reactant.Ops.constant(TABLES.wb)
end

# eager versions use plain arrays for the constants
const _C = Dict(:ylm => YLM_MAPS, :rnl => TABLES.rnl)
eager(f, args...) = f(args...)

STAGES = [("vectors", stage_vectors), ("lengths", stage_lengths), ("u", stage_u),
          ("monomials(lmax)", stage_mono), ("ylm", stage_ylm), ("rnl", stage_rnl),
          ("edge_a", stage_edge_a), ("A (segment_sum)", stage_A),
          ("site energies", stage_site)]

rargs = (Reactant.to_rarray(pos_pad), Reactant.to_rarray(centers),
         Reactant.to_rarray(neighbors), Reactant.to_rarray(edge_mask))
eargs = (pos_pad, centers, neighbors, edge_mask)

@printf("\n%-18s %14s %14s   %s\n", "stage", "max|eager|", "max|c-e|", "rel")
for (name, f) in STAGES
    local ev, cv
    try
        ev = Array(f(eargs...))
    catch err
        @printf("%-18s  EAGER FAILED: %s\n", name, sprint(showerror, err)[1:min(70,end)])
        continue
    end
    try
        cf = @compile f(rargs...)
        cv = Array(cf(rargs...))
    catch err
        @printf("%-18s  COMPILE FAILED: %s\n", name, sprint(showerror, err)[1:min(70,end)])
        continue
    end
    d = maximum(abs.(cv .- ev)); den = max(maximum(abs.(ev)), eps())
    @printf("%-18s %14.6e %14.6e   %.3e%s\n", name, den, d, d/den,
            d/den > 1e-10 ? "   <<< DIVERGES" : "")
end
