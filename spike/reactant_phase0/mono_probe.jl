# Localise the compiled-vs-eager divergence in `monomials`. SPIKE CODE.
using Reactant, Printf, LinearAlgebra
Reactant.set_default_backend("cpu")

monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])

L = 2
EXPS = [(a,b,c) for a in 0:L for b in 0:L for c in 0:L if a+b+c == L]
println("exponents (degree $L): ", EXPS)

u = [0.1 0.2 0.3; 0.4 0.5 0.6; -0.7 0.8 -0.9; 1.0 -1.1 1.2]
ru = Reactant.to_rarray(u)

E = monomials(u, EXPS)
C = Array((@compile monomials(ru, EXPS))(ru, EXPS))
@printf("\neager  size %s\ncompiled size %s\n", size(E), size(C))
println("\neager:");    display(round.(E, digits=4))
println("\ncompiled:"); display(round.(C, digits=4))
@printf("\nmax|c-e| = %.3e\n", maximum(abs.(C .- E)))

# is compiled a column permutation of eager?
println("\ncolumn matching (compiled col -> eager col):")
for j in 1:size(C,2)
    best, bd = 0, Inf
    for i in 1:size(E,2)
        d = maximum(abs.(C[:,j] .- E[:,i]))
        d < bd && ((best, bd) = (i, d))
    end
    @printf("  compiled[:,%d] (exp %s) == eager[:,%d] (exp %s)  d=%.2e\n",
            j, EXPS[j], best, EXPS[best], bd)
end

# per-column, single exponent at a time
println("\nper-exponent, compiled vs eager:")
for e in EXPS
    ce = monomial_col(u, e)
    cc = Array((@compile monomial_col(ru, e))(ru, e))
    @printf("  e=%s  max|c-e| = %.3e\n", e, maximum(abs.(cc .- ce)))
end
