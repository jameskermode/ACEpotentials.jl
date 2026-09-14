# Minimal reproducer: Reactant mis-traces an accumulator built from column
# views taken at different loop indices. Reactant 0.2.222, CPU.
using Reactant, Printf
Reactant.set_default_backend("cpu")
u  = [1.0 2.0 3.0; 4.0 5.0 6.0]          # x=col1, y=col2, z=col3
ru = Reactant.to_rarray(u)

# 1. straight-line, no loop
f_straight(u) = u[:, 1] .* u[:, 2]                     # want x.*y
# 2. same product, accumulated across a loop over k
function f_loop(u)
    col = nothing
    for k in 1:2
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col
end
# 3. loop, but each view materialised via copy
function f_loop_copy(u)
    col = nothing
    for k in 1:2
        v = copy(u[:, k])
        col = col === nothing ? v : col .* v
    end
    col
end
# 4. loop over an explicit list of columns instead of an index
function f_loop_cols(u)
    cols = [u[:, k] for k in 1:2]
    reduce(.*, cols)
end

for (name, f) in (("straight  x.*y", f_straight), ("loop      x.*y", f_loop),
                  ("loop+copy x.*y", f_loop_copy), ("loop cols x.*y", f_loop_cols))
    e = f(u); c = Array((@compile f(ru))(ru))
    @printf("%-16s eager=%s compiled=%s  %s\n", name, string(round.(e,digits=3)),
            string(round.(c,digits=3)), maximum(abs.(c.-e)) < 1e-12 ? "OK" : "<<< WRONG")
end
