using Reactant, KernelAbstractions, Printf
Reactant.set_default_backend("cpu")
println("Reactant ", pkgversion(Reactant))
function attempt(f, name)
    print(rpad(name, 50))
    try println("OK   ", f()) catch err
        println("FAIL: ", first(replace(first(split(sprint(showerror,err),"\n")), r"\s+"=>" "), 95))
    end
end

@kernel function k1d!(A, X)            # 1-D ndrange, no scalar host index
    i = @index(Global)
    A[i] = 2 * X[i]
end
@kernel function k2d!(A, X)            # 2-D ndrange
    i, j = @index(Global, NTuple)
    A[i, j] = 2 * X[i, j]
end
@kernel function k2d_loop!(B, P, W, idx)   # 2-D + inner loop + traced gather
    i, j = @index(Global, NTuple)
    acc = zero(eltype(B))
    for k = 1:size(P, 2)
        acc += W[j, k, idx[i]] * P[i, k]
    end
    B[i, j] = acc
end

x1 = randn(32); x2 = randn(8, 4)
attempt("1-D ndrange, traced in/out") do
    f(x) = (A = similar(x); k1d!(KernelAbstractions.get_backend(x))(A, x; ndrange=length(x)); A)
    rx = Reactant.to_rarray(x1)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rx))(rx)) .- 2 .* x1)))
end
attempt("2-D ndrange, traced in/out") do
    f(x) = (A = similar(x); k2d!(KernelAbstractions.get_backend(x))(A, x; ndrange=size(x)); A)
    rx = Reactant.to_rarray(x2)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rx))(rx)) .- 2 .* x2)))
end
P = randn(8, 5); W = randn(4, 5, 2); idx = rand(1:2, 8)
attempt("2-D + loop + host idx array (as ET has it)") do
    f(p, w) = (B = similar(p, size(p,1), size(w,1));
               k2d_loop!(KernelAbstractions.get_backend(p))(B, p, w, idx; ndrange=size(B)); B)
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    Array((@compile f(rp,rw))(rp,rw)); "traced"
end
attempt("2-D + loop + TRACED idx array") do
    f(p, w, ix) = (B = similar(p, size(p,1), size(w,1));
               k2d_loop!(KernelAbstractions.get_backend(p))(B, p, w, ix; ndrange=size(B)); B)
    rp, rw, ri = Reactant.to_rarray(P), Reactant.to_rarray(W), Reactant.to_rarray(idx)
    Array((@compile f(rp,rw,ri))(rp,rw,ri)); "traced"
end
# and the array-op equivalent of SelectLinL, no kernel at all
attempt("SelectLinL as pure array ops (no KA)") do
    ref = reduce(hcat, [ [sum(W[j,k,idx[i]]*P[i,k] for k in 1:5) for i in 1:8] for j in 1:4 ])
    g(p, w) = begin
        Wsel = w[:, :, idx]                       # (4, 5, 8) gather
        dropdims(sum(permutedims(Wsel, (3,1,2)) .* reshape(p, size(p,1), 1, size(p,2)); dims=3), dims=3)
    end
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- ref)))
end
