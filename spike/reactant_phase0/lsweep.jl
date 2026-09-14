using Reactant, Printf, Random
Reactant.set_default_backend("cpu")
monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])
rng = MersenneTwister(1); u = randn(rng, 64, 3); ru = Reactant.to_rarray(u)
println(" l  n_mono   status    n_wrong_cols   which exponents are wrong")
for L in 0:5
    EX = [(a,b,c) for a in 0:L for b in 0:L for c in 0:L if a+b+c == L]
    f = u -> monomials(u, EX)
    w = f(u); c = Array((@compile f(ru))(ru))
    bad = [j for j in 1:size(w,2) if maximum(abs.(c[:,j] .- w[:,j])) > 1e-12]
    @printf("%2d %6d   %-8s %10d   %s\n", L, length(EX),
            isempty(bad) ? "OK" : "CORRUPT", length(bad),
            isempty(bad) ? "-" : string([EX[j] for j in bad]))
end
# also: is a bare 2-factor product of distinct columns enough?
println("\nminimal: single mixed monomial per compiled function")
for e in [(1,1,0), (1,0,1), (0,1,1), (1,1,1), (2,1,0)]
    f = u -> monomial_col(u, e)
    w = f(u); c = Array((@compile f(ru))(ru))
    @printf("  e=%-10s %s\n", string(e), maximum(abs.(c .- w)) < 1e-12 ? "OK" : "CORRUPT")
end
