# The failing construct is reduce(hcat, [monomial_col(u,e) for e in exps]).
# Each monomial_col is correct alone; combining them corrupts the mixed ones.
using Reactant, Printf
Reactant.set_default_backend("cpu")
u  = [1.0 2.0 3.0; 4.0 5.0 6.0]
ru = Reactant.to_rarray(u)

monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end

EX = [(0,1,1), (1,0,1), (1,1,0), (2,0,0)]   # want yz, xz, xy, x^2

variants = Dict(
  "reduce(hcat, comprehension)" => (u -> reduce(hcat, [monomial_col(u,e) for e in EX])),
  "hcat(splat)"                 => (u -> hcat([monomial_col(u,e) for e in EX]...)),
  "reduce+copy each col"        => (u -> reduce(hcat, [copy(monomial_col(u,e)) for e in EX])),
  "cat(dims=2)"                 => (u -> cat([monomial_col(u,e) for e in EX]...; dims=2)),
  "stack"                       => (u -> reduce(hcat, map(e -> monomial_col(u,e), EX))),
)
want = reduce(hcat, [monomial_col(u,e) for e in EX])
println("eager (want):"); display(round.(want, digits=3))
for name in sort(collect(keys(variants)))
    f = variants[name]
    c = Array((@compile f(ru))(ru))
    ok = maximum(abs.(c .- want)) < 1e-12
    @printf("\n%-28s %s\n", name, ok ? "OK" : "<<< WRONG")
    ok || display(round.(c, digits=3))
end

# does the number of columns matter? two at a time
println("\npairwise: which combinations corrupt?")
for i in 1:length(EX), j in i+1:length(EX)
    g = u -> reduce(hcat, [monomial_col(u,EX[i]), monomial_col(u,EX[j])])
    w = reduce(hcat, [monomial_col(u,EX[i]), monomial_col(u,EX[j])])
    c = Array((@compile g(ru))(ru))
    @printf("  %s + %s : %s\n", EX[i], EX[j],
            maximum(abs.(c .- w)) < 1e-12 ? "OK" : "WRONG")
end
