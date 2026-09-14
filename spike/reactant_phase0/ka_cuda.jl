# Does a KernelAbstractions kernel trace under Reactant on a CUDA HOST?
#
# The earlier spike (no CUDA on the box) found that NO KA kernel traced, not
# even `A[i] = 2*X[i]`, with a `ka_with_reactant` MethodError.  Reactant's KA
# bridge may live in its CUDA extension (cf. Reactant issue #3038, whose
# environment lists "CUDA ... CPU backend (CUDA loaded for ReactantCUDAExt)").
# This script re-runs the ladder from ka_triggers.jl / ka_blockers.jl with CUDA
# loaded, on whichever Reactant backend is named in ARGS[1] ("cpu" | "gpu").
#
#   julia --project=. ka_cuda.jl cpu
#   julia --project=. ka_cuda.jl gpu

const BACKEND = length(ARGS) >= 1 ? ARGS[1] : "cpu"

using Reactant, CUDA, KernelAbstractions, StaticArrays, Printf

println("="^78)
println("Julia ", VERSION, "   backend requested: ", BACKEND)
println("Reactant ", pkgversion(Reactant),
        "  KernelAbstractions ", pkgversion(KernelAbstractions),
        "  CUDA ", pkgversion(CUDA))
println("CUDA.functional() = ", CUDA.functional())
try
    println("CUDA device: ", CUDA.name(CUDA.device()))
catch e
    println("CUDA device: unavailable (", first(split(sprint(showerror, e), "\n")), ")")
end

# which Reactant extensions actually loaded?
exts = [string(k.name) for k in keys(Base.loaded_modules)
        if startswith(string(k.name), "Reactant") && string(k.name) != "Reactant"]
println("loaded Reactant extensions: ", isempty(exts) ? "(none)" : join(sort(exts), ", "))

# is the KA bridge method present at all?
for sym in (:ka_with_reactant,)
    if isdefined(Reactant, sym)
        ms = methods(getfield(Reactant, sym))
        println("Reactant.", sym, ": ", length(ms), " method(s)")
        for m in ms; println("    ", m); end
    else
        println("Reactant.", sym, ": NOT DEFINED in Reactant")
    end
end

Reactant.set_default_backend(BACKEND)
println("Reactant devices: ", try string(Reactant.devices()) catch e; "?" end)
println("="^78)

results = Tuple{String,String,String}[]   # (name, verdict, detail)
function attempt(f, name)
    print(rpad(name, 46))
    local verdict, detail
    try
        detail = f()
        verdict = "TRACES"
        println("TRACES   ", detail)
    catch err
        s = replace(first(split(sprint(showerror, err), "\n")), r"\s+" => " ")
        detail = first(s, 130)
        verdict = occursin("ka_with_reactant", s)      ? "NO (ka_with_reactant)" :
                  occursin("scalar indexing", s)       ? "NO (scalar indexing)"  :
                  occursin("MethodError", s)           ? "NO (MethodError)"      : "NO"
        println("FAIL   ", detail)
    end
    push!(results, (name, verdict, detail))
    return nothing
end

# --------------------------------------------------------------------------
# The ladder (ka_triggers.jl)
# --------------------------------------------------------------------------
@kernel function k1d!(A, X)                  # trivial, 1-D ndrange, all traced
    i = @index(Global)
    A[i] = 2 * X[i]
end
@kernel function k2d!(A, X)                  # 2-D ndrange
    i, j = @index(Global, NTuple)
    A[i, j] = 2 * X[i, j]
end
@kernel function k2d_loop!(B, P, W, idx)     # 2-D + inner loop + gather
    i, j = @index(Global, NTuple)
    acc = zero(eltype(B))
    for k = 1:size(P, 2)
        acc += W[j, k, idx[i]] * P[i, k]
    end
    B[i, j] = acc
end

x1 = randn(32); x2 = randn(8, 4)
P = randn(8, 5); W = randn(4, 5, 2); idx = rand(1:2, 8)
refloop = [sum(W[j, k, idx[i]] * P[i, k] for k in 1:5) for i in 1:8, j in 1:4]

# control: does KA itself work here, outside Reactant?
attempt("0a. KA on plain CPU arrays (control)") do
    A = similar(x1); k1d!(KernelAbstractions.get_backend(x1))(A, x1; ndrange=length(x1))
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(x1))
    @sprintf("max|k-e| = %.3e", maximum(abs.(A .- 2 .* x1)))
end
attempt("0b. KA on CuArray (control)") do
    cx = CUDA.CuArray(x1); A = similar(cx)
    k1d!(KernelAbstractions.get_backend(cx))(A, cx; ndrange=length(cx))
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(cx))
    @sprintf("max|k-e| = %.3e", maximum(abs.(Array(A) .- 2 .* x1)))
end

attempt("1. trivial KA, 1-D ndrange, all traced") do
    f(x) = (A = similar(x); k1d!(KernelAbstractions.get_backend(x))(A, x; ndrange=length(x)); A)
    rx = Reactant.to_rarray(x1)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rx))(rx)) .- 2 .* x1)))
end
attempt("2. 2-D ndrange, traced in/out") do
    f(x) = (A = similar(x); k2d!(KernelAbstractions.get_backend(x))(A, x; ndrange=size(x)); A)
    rx = Reactant.to_rarray(x2)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rx))(rx)) .- 2 .* x2)))
end
attempt("3. 2-D + inner loop + HOST idx array") do
    f(p, w) = (B = similar(p, size(p,1), size(w,1));
               k2d_loop!(KernelAbstractions.get_backend(p))(B, p, w, idx; ndrange=size(B)); B)
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rp,rw))(rp,rw)) .- refloop)))
end
attempt("4. 2-D + inner loop + TRACED idx array") do
    f(p, w, ix) = (B = similar(p, size(p,1), size(w,1));
               k2d_loop!(KernelAbstractions.get_backend(p))(B, p, w, ix; ndrange=size(B)); B)
    rp, rw, ri = Reactant.to_rarray(P), Reactant.to_rarray(W), Reactant.to_rarray(idx)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile f(rp,rw,ri))(rp,rw,ri)) .- refloop)))
end

# --------------------------------------------------------------------------
# The two real ET blockers, replicated standalone (ka_blockers.jl)
# --------------------------------------------------------------------------
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)   # ET utils/selectlinl.jl:56
    iB, jB = @index(Global, NTuple)
    i_x = selector(X[iB])
    B[iB, jB] = 0
    for k = 1:size(P, 2)
        B[iB, jB] += W[jB, k, i_x] * P[iB, k]
    end
    nothing
end
function selectlinl(P, X, W, selector)
    B = similar(P, size(P,1), size(W,1))
    kernel! = _ka_apply_selectlinl!(KernelAbstractions.get_backend(P))
    kernel!(B, P, X, W, selector; ndrange = size(B))
    return B
end
@kernel function _ka_pooled!(A, R, Y, spec_r, spec_y, nX)      # ET ace/sparseprodpool_ka.jl
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

P2 = randn(16, 5); Xi = rand(1:2, 16); W2 = randn(4, 5, 2)
attempt("B1. SelectLinL kernel, X::Vector{Int}") do
    ref = selectlinl(P2, Xi, W2, identity)
    g(p, w) = selectlinl(p, Xi, w, identity)
    rp, rw = Reactant.to_rarray(P2), Reactant.to_rarray(W2)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- ref)))
end
struct Edge; z0::Int; z1::Int; end
Xs = [Edge(rand(1:2), rand(1:2)) for _ in 1:16]
attempt("B1b. SelectLinL kernel, X::Vector{Edge}") do
    sel = e -> e.z0
    ref = selectlinl(P2, Xs, W2, sel)
    g(p, w) = selectlinl(p, Xs, w, sel)
    rp, rw = Reactant.to_rarray(P2), Reactant.to_rarray(W2)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- ref)))
end
R = randn(6, 10, 7); Y = randn(6, 10, 9); sr = rand(1:7, 12); sy = rand(1:9, 12)
attempt("B2. PooledSparseProduct kernel (abasis)") do
    ref = pooled(R, Y, sr, sy)
    g(r, y) = pooled(r, y, sr, sy)
    rr, ry = Reactant.to_rarray(R), Reactant.to_rarray(Y)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rr,ry))(rr,ry)) .- ref)))
end

# --------------------------------------------------------------------------
# the array-op (no-KA) formulation of SelectLinL, for contrast
# --------------------------------------------------------------------------
attempt("C. SelectLinL as pure array ops (no KA)") do
    g(p, w) = begin
        Wsel = w[:, :, idx]
        dropdims(sum(permutedims(Wsel, (3,1,2)) .* reshape(p, size(p,1), 1, size(p,2)); dims=3), dims=3)
    end
    rp, rw = Reactant.to_rarray(P), Reactant.to_rarray(W)
    @sprintf("max|c-e| = %.3e", maximum(abs.(Array((@compile g(rp,rw))(rp,rw)) .- refloop)))
end

# --------------------------------------------------------------------------
# and the known miscompilation (#3267) on this backend, so a wrong VALUE
# elsewhere is not mistaken for a tracing failure
# --------------------------------------------------------------------------
attempt("D. Reactant #3267 miscompilation check") do
    h(u) = hcat(u[:, 3] .* u[:, 3], u[:, 2] .* u[:, 3])
    u = [1.0 2.0 3.0; 4.0 5.0 6.0]; ru = Reactant.to_rarray(u)
    c = Array((@compile h(ru))(ru)); e = h(u)
    d = maximum(abs.(c .- e))
    @sprintf("max|c-e| = %.3e  %s", d, d < 1e-12 ? "(bug NOT present)" : "(#3267 PRESENT: $(c))")
end

println("\n", "="^78)
println("SUMMARY  [Julia $(VERSION), Reactant $(pkgversion(Reactant)), backend=$BACKEND, CUDA loaded]")
println("="^78)
for (n, v, d) in results
    println(rpad(n, 46), rpad(v, 24), first(d, 60))
end
