# Do the two ETACE tracing blockers still fail on the LATEST Reactant?
#
# EquivariantTensors cannot be installed alongside Reactant > 0.2.222
# (ET -> WignerD -> StructArrays <= 0.6.21 vs Reactant -> StructArrays >= 0.7.2),
# so the two blocking kernels are replicated standalone here.
using Reactant, KernelAbstractions, StaticArrays, Printf
Reactant.set_default_backend("cpu")
println("Reactant ", pkgversion(Reactant), "  KernelAbstractions ", pkgversion(KernelAbstractions))

# ---- Blocker 1: ET.SelectLinL  (utils/selectlinl.jl:56-68) ------------------
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
    iB, jB = @index(Global, NTuple)
    i_x = selector(X[iB])                 # scalar index into a struct array
    B[iB, jB] = 0
    for k = 1:size(P, 2)
        B[iB, jB] += W[jB, k, i_x] * P[iB, k]
    end
    nothing
end
function selectlinl(P, X, W, selector)
    B = similar(P, size(P,1), size(W,1))
    kernel! = _ka_apply_selectlinl!(KernelAbstractions.get_backend(X))
    kernel!(B, P, X, W, selector; ndrange = size(B))
    return B
end

# ---- Blocker 2: PooledSparseProduct  (ace/sparseprodpool_ka.jl) -------------
@kernel function _ka_pooled!(A, R, Y, spec_r, spec_y, nX)
    i, k = @index(Global, NTuple)
    a = zero(eltype(A))
    for j = 1:nX
        a += R[j, i, spec_r[k]] * Y[j, i, spec_y[k]]
    end
    A[i, k] = a
end
function pooled(R, Y, sr, sy)
    A = similar(R, size(R,2), length(sr))
    kernel! = _ka_pooled!(KernelAbstractions.get_backend(R))
    kernel!(A, R, Y, sr, sy, size(R,1); ndrange = size(A))
    return A
end

function attempt(f, name)
    print(rpad(name, 44))
    try
        println("OK   ", f())
    catch err
        s = replace(first(split(sprint(showerror, err), "\n")), r"\s+" => " ")
        println("FAIL: ", first(s, 110))
    end
end

# blocker 1: X as a plain Int vector (species index already resolved)
P = randn(16, 5); Xi = rand(1:2, 16); W = randn(4, 5, 2)
attempt("B1 SelectLinL, X::Vector{Int}") do
    ref = selectlinl(P, Xi, W, identity)
    g(p, w) = selectlinl(p, Xi, w, identity)
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- ref)))
end
# blocker 1 as ET actually has it: X is an array of structs, selector extracts
struct Edge; z0::Int; z1::Int; end
Xs = [Edge(rand(1:2), rand(1:2)) for _ in 1:16]
attempt("B1 SelectLinL, X::Vector{Edge}") do
    sel = e -> e.z0
    ref = selectlinl(P, Xs, W, sel)
    g(p, w) = selectlinl(p, Xs, w, sel)
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- ref)))
end
# blocker 2
R = randn(6, 10, 7); Y = randn(6, 10, 9); sr = rand(1:7, 12); sy = rand(1:9, 12)
attempt("B2 PooledSparseProduct kernel") do
    ref = pooled(R, Y, sr, sy)
    g(r, y) = pooled(r, y, sr, sy)
    rr, ry = Reactant.to_rarray(R), Reactant.to_rarray(Y)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rr,ry))(rr,ry)) .- ref)))
end
