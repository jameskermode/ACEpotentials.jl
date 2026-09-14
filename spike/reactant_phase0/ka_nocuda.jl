# Run ONE case of the KA ladder in its own process, so a hard crash is
# attributable and cannot swallow later results.
#   julia --project=. ka_one.jl <case> <cpu|gpu>
const CASE = ARGS[1]; const BACKEND = length(ARGS) >= 2 ? ARGS[2] : "cpu"
using Reactant, KernelAbstractions, StaticArrays, Printf  # NOTE: CUDA deliberately NOT loaded
Reactant.set_default_backend(BACKEND)
say(args...) = (println(args...); flush(stdout))
say("## case=", CASE, " backend=", BACKEND, " Julia ", VERSION,
    " Reactant ", pkgversion(Reactant), " CUDA NOT LOADED")

say("## ka_with_reactant defined in Reactant? ",
    isdefined(Reactant, :ka_with_reactant) ?
      string(length(methods(getfield(Reactant, :ka_with_reactant))), " method(s)") : "NOT DEFINED")
say("## loaded Reactant extensions: ", join(sort([string(k.name) for k in keys(Base.loaded_modules)
     if startswith(string(k.name), "Reactant") && string(k.name) != "Reactant"]), ", "))

@kernel function k1d!(A, X)
    i = @index(Global); A[i] = 2 * X[i]
end
@kernel function k2d!(A, X)
    i, j = @index(Global, NTuple); A[i, j] = 2 * X[i, j]
end
@kernel function k2d_loop!(B, P, W, idx)
    i, j = @index(Global, NTuple)
    acc = zero(eltype(B))
    for k = 1:size(P, 2); acc += W[j, k, idx[i]] * P[i, k]; end
    B[i, j] = acc
end
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
    iB, jB = @index(Global, NTuple)
    i_x = selector(X[iB])
    B[iB, jB] = 0
    for k = 1:size(P, 2); B[iB, jB] += W[jB, k, i_x] * P[iB, k]; end
    nothing
end
function selectlinl(P, X, W, selector)
    B = similar(P, size(P,1), size(W,1))
    _ka_apply_selectlinl!(KernelAbstractions.get_backend(P))(B, P, X, W, selector; ndrange=size(B))
    return B
end
@kernel function _ka_pooled!(A, R, Y, spec_r, spec_y, nX)
    i, k = @index(Global, NTuple)
    a = zero(eltype(A))
    for j = 1:nX; a += R[j, i, spec_r[k]] * Y[j, i, spec_y[k]]; end
    A[i, k] = a
end
function pooled(R, Y, sr, sy)
    A = similar(R, size(R,2), length(sr))
    _ka_pooled!(KernelAbstractions.get_backend(R))(A, R, Y, sr, sy, size(R,1); ndrange=size(A))
    return A
end
struct Edge; z0::Int; z1::Int; end

using Random; Random.seed!(7)
x1 = randn(32); x2 = randn(8, 4)
P = randn(8, 5); W = randn(4, 5, 2); idx = rand(1:2, 8)
refloop = [sum(W[j,k,idx[i]]*P[i,k] for k in 1:5) for i in 1:8, j in 1:4]
P2 = randn(16, 5); Xi = rand(1:2, 16); W2 = randn(4, 5, 2)
Xs = [Edge(Xi[i], Xi[i]) for i in 1:16]
R = randn(6, 10, 7); Y = randn(6, 10, 9); sr = rand(1:7, 12); sy = rand(1:9, 12)

function run(case)
    if case == "1"
        f1(x) = (A = similar(x); k1d!(KernelAbstractions.get_backend(x))(A, x; ndrange=length(x)); A)
        rx = Reactant.to_rarray(x1)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f1(rx))(rx)) .- 2 .* x1)))
    elseif case == "2"
        f2(x) = (A = similar(x); k2d!(KernelAbstractions.get_backend(x))(A, x; ndrange=size(x)); A)
        rx = Reactant.to_rarray(x2)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f2(rx))(rx)) .- 2 .* x2)))
    elseif case == "3"
        f3(p,w) = (B = similar(p, size(p,1), size(w,1)); k2d_loop!(KernelAbstractions.get_backend(p))(B,p,w,idx; ndrange=size(B)); B)
        rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f3(rp,rw))(rp,rw)) .- refloop)))
    elseif case == "4"
        f4(p,w,ix) = (B = similar(p, size(p,1), size(w,1)); k2d_loop!(KernelAbstractions.get_backend(p))(B,p,w,ix; ndrange=size(B)); B)
        rp, rw, ri = Reactant.to_rarray(P), Reactant.to_rarray(W), Reactant.to_rarray(idx)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f4(rp,rw,ri))(rp,rw,ri)) .- refloop)))
    elseif case == "B1"
        ref = selectlinl(P2, Xi, W2, identity)
        gB1(p,w) = selectlinl(p, Xi, w, identity)
        rp, rw = Reactant.to_rarray(P2), Reactant.to_rarray(W2)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile gB1(rp,rw))(rp,rw)) .- ref)))
    elseif case == "B1b"
        sel = e -> e.z0
        ref = selectlinl(P2, Xs, W2, sel)
        gB1b(p,w) = selectlinl(p, Xs, w, sel)
        rp, rw = Reactant.to_rarray(P2), Reactant.to_rarray(W2)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile gB1b(rp,rw))(rp,rw)) .- ref)))
    elseif case == "B2"
        ref = pooled(R, Y, sr, sy)
        gB2(r,y) = pooled(r, y, sr, sy)
        rr, ry = Reactant.to_rarray(R), Reactant.to_rarray(Y)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile gB2(rr,ry))(rr,ry)) .- ref)))
    elseif case == "C"
        gC(p,w) = (Wsel = w[:,:,idx];
                  dropdims(sum(permutedims(Wsel,(3,1,2)) .* reshape(p,size(p,1),1,size(p,2)); dims=3), dims=3))
        rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
        return @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile gC(rp,rw))(rp,rw)) .- refloop)))
    elseif case == "D"
        h(u) = hcat(u[:,3] .* u[:,3], u[:,2] .* u[:,3])
        u = [1.0 2.0 3.0; 4.0 5.0 6.0]; ru = Reactant.to_rarray(u)
        c = Array((@compile h(ru))(ru)); e = h(u); d = maximum(abs.(c .- e))
        return @sprintf("max|c-e| = %.3e %s", d, d < 1e-12 ? "(#3267 NOT present)" : "(#3267 PRESENT $(c))")
    end
    error("unknown case $case")
end

try
    say("RESULT ", CASE, " ", BACKEND, "  TRACES  ", run(CASE))
catch err
    s = replace(first(split(sprint(showerror, err), "\n")), r"\s+" => " ")
    say("RESULT ", CASE, " ", BACKEND, "  FAIL  ", first(s, 200))
end
