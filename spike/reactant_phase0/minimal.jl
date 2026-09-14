using Reactant, Printf, Random, Combinatorics
Reactant.set_default_backend("cpu")
monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])
rng = MersenneTwister(1); u = randn(rng, 8, 3); ru = Reactant.to_rarray(u)
ALL = [(0,0,2),(0,1,1),(0,2,0),(1,0,1),(1,1,0),(2,0,0)]
bad(EX) = begin
    f = u -> monomials(u, EX); w = f(u); c = Array((@compile f(ru))(ru))
    maximum(abs.(c .- w)) > 1e-12
end
println("searching for the smallest corrupting exponent set...")
found = nothing
for n in 2:length(ALL), sub in combinations(ALL, n)
    if bad(sub); global found = sub; break; end
    found !== nothing && break
end
println("\nminimal corrupting set (size $(length(found))): ", found)
f = u -> monomials(u, found); w = f(u); c = Array((@compile f(ru))(ru))
println("\neager:");    display(round.(w[1:3,:], digits=4))
println("compiled:");   display(round.(c[1:3,:], digits=4))
println("u:");          display(round.(u[1:3,:], digits=4))
