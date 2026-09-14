# Why does the Reactant ACE export diverge on the ylm intermediate?
#
# lammps-jax dev/julia_export examples/julia/ace_export.jl cannot trace
# SpheriCart, so it replaces Ylm with monomials of the unit vector mapped onto
# SpheriCart values by a least-squares fit at 64 random probes:
#
#     YLM_MAPS[l+1] = monomials(probes, EXPONENTS[l+1]) \ y_ref[:, l^2+1:(l+1)^2]
#     ylm           = monomials(u, EXPONENTS[l+1]) * YLM_MAPS[l+1]
#
# This measures the conditioning of that construction. SPIKE CODE.

using ACEpotentials, LinearAlgebra, Random, StaticArrays
import Polynomials4ML as P4ML
M = ACEpotentials.Models

monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])

TD = parse(Int, get(ENV, "TD", "8"))
model = ace1_model(elements = [:Al], order = 3, totaldegree = TD)
m = model.model
A_LM = [t[2] for t in m.tensor.abasis.spec]
LMAX = isqrt(maximum(A_LM) - 1)
EXPONENTS = [[(a,b,c) for a in 0:l for b in 0:l for c in 0:l if a+b+c == l] for l in 0:LMAX]
println("model: ace1_model(:Al, order=3, totaldegree=$TD)   LMAX = $LMAX")

rng = MersenneTwister(7)   # same seed as ace_export.jl
probes = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:64]
probe_mat = permutedims(reduce(hcat, [collect(v) for v in probes]))
y_ref = Matrix(P4ML.evaluate(m.ybasis, probes))

println("\n l  n_mono  cond(feats)   ||m_l||     fit resid (probes)   TEST-SET err")
maps = []
for l in 0:LMAX
    feats = monomials(probe_mat, EXPONENTS[l+1])
    block = y_ref[:, l*l+1:(l+1)*(l+1)]
    m_l = feats \ block
    push!(maps, m_l)
    resid = maximum(abs.(feats * m_l - block))
    # held-out directions: the fit is only constrained on the 64 probes
    test = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:2000]
    tmat = permutedims(reduce(hcat, [collect(v) for v in test]))
    yt = Matrix(P4ML.evaluate(m.ybasis, test))[:, l*l+1:(l+1)*(l+1)]
    terr = maximum(abs.(monomials(tmat, EXPONENTS[l+1]) * m_l - yt))
    println(lpad(l,2), lpad(length(EXPONENTS[l+1]),8),
            lpad(round(cond(feats), sigdigits=4),13),
            lpad(round(norm(m_l), sigdigits=5),11),
            lpad(round(resid, sigdigits=4),21),
            lpad(round(terr, sigdigits=4),16))
end

# Amplification: how much does a 1-ulp perturbation of the monomials move ylm?
println("\nsensitivity: relative ylm error from a 1e-16 relative monomial perturbation")
for l in 0:LMAX
    feats = monomials(probe_mat, EXPONENTS[l+1])
    yl = feats * maps[l+1]
    pert = feats .* (1 .+ 1e-16 .* randn(rng, size(feats)))
    d = maximum(abs.(pert * maps[l+1] - yl)) / max(maximum(abs.(yl)), eps())
    println("  l=$l  amplified relative error = ", round(d, sigdigits=4))
end
