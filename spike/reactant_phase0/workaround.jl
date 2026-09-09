using Reactant, Printf, Random
Reactant.set_default_backend("cpu")

# as in lammps-jax ace_export.jl
monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
# workaround: build the factor list first, no nothing-accumulator
monomial_col_fix(u, e) = begin
    f = [u[:, k] for k in 1:3 for _ in 1:e[k]]
    isempty(f) ? one.(u[:, 1]) : reduce(.*, f)
end
mono(cf) = (u, exps) -> reduce(hcat, [cf(u, e) for e in exps])

rng = MersenneTwister(1)
for N in (2, 4, 64, 2048), L in (1, 2, 3)
    EX = [(a,b,c) for a in 0:L for b in 0:L for c in 0:L if a+b+c == L]
    u = randn(rng, N, 3); ru = Reactant.to_rarray(u)
    want = mono(monomial_col)(u, EX)
    f_bad = u -> mono(monomial_col)(u, EX)
    f_fix = u -> mono(monomial_col_fix)(u, EX)
    b = Array((@compile f_bad(ru))(ru)); g = Array((@compile f_fix(ru))(ru))
    @printf("N=%-5d l=%d  n_mono=%-3d  original: %-9s  workaround: %s\n", N, L, length(EX),
            maximum(abs.(b .- want)) < 1e-12 ? "OK" : "CORRUPT",
            maximum(abs.(g .- want)) < 1e-12 ? "OK" : "CORRUPT")
end
