# Does monomials break when u is a permutedims of a traced array?
using Reactant, Printf
Reactant.set_default_backend("cpu")
monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])
EX = [(0,1,1), (1,0,1), (1,1,0), (2,0,0)]

v = [1.0 4.0 7.0 10.0; 2.0 5.0 8.0 11.0; 3.0 6.0 9.0 12.0]   # 3 x N
u_direct = permutedims(v)                                     # N x 3
rv = Reactant.to_rarray(v); ru = Reactant.to_rarray(u_direct)

tests = [
 ("u passed directly (N x 3)",        ru, u_direct, (x -> monomials(x, EX))),
 ("permutedims inside fn",            rv, v,        (x -> monomials(permutedims(x), EX))),
 ("permutedims + copy",               rv, v,        (x -> monomials(copy(permutedims(x)), EX))),
 ("full stage_u shape (norm+div)",    rv, v,
    (x -> begin l = sqrt.(sum(abs2, x; dims=1))[1,:]
                monomials(permutedims(x ./ permutedims(l)), EX) end)),
 ("stage_u + copy",                   rv, v,
    (x -> begin l = sqrt.(sum(abs2, x; dims=1))[1,:]
                monomials(copy(permutedims(x ./ permutedims(l))), EX) end)),
]
for (name, ra, ea, f) in tests
    w = f(ea); c = Array((@compile f(ra))(ra))
    ok = maximum(abs.(c .- w)) < 1e-12
    @printf("%-32s %s   max|c-e|=%.3e\n", name, ok ? "OK " : "<<< WRONG", maximum(abs.(c .- w)))
    if !ok
        println("   eager:");    display(round.(w, digits=3))
        println("   compiled:"); display(round.(c, digits=3))
    end
end
