# LSQR (Paige & Saunders 1982) with damping, written against three callbacks so
# the identical code runs serially and row-block distributed:
#   Ax!(u, x)  : u  = A*x       (u is the local row block; x replicated)
#   Atu!(v, u) : v  = A'*u      (global: local product + Allreduce)
#   unorm(u)   : ||u||_2 global (local sum of squares + Allreduce)
# Stopping: ||A'r||/(||A|| ||r||) <= atol (LSQR's own estimate) or maxiter.
using LinearAlgebra
function lsqr_cb(Ax!, Atu!, unorm, b::Vector{Float64}, n::Int;
                 damp = 0.0, atol = 1e-14, btol = 1e-14, maxiter = 1000)
    m = length(b)
    u = copy(b); v = zeros(n); w = zeros(n); x = zeros(n)
    β = unorm(u); u ./= β
    Atu!(v, u); α = norm(v); v ./= α
    w .= v
    φ̄ = β; ρ̄ = α; bnorm = β
    Anorm = 0.0; xnorm = 0.0; ddnorm = 0.0; res2 = 0.0; z = 0.0; cs2 = -1.0; sn2 = 0.0
    tmpu = similar(u); tmpv = similar(v)
    it = 0; test2 = Inf; test1 = Inf
    while it < maxiter
        it += 1
        Ax!(tmpu, v); @. u = tmpu - α * u; β = unorm(u); u ./= β
        Atu!(tmpv, u); @. v = tmpv - β * v; α = norm(v); v ./= α
        Anorm = sqrt(Anorm^2 + α^2 + β^2 + damp^2)
        # damping rotation
        ρ̄1 = sqrt(ρ̄^2 + damp^2); c1 = ρ̄ / ρ̄1; s1 = damp / ρ̄1
        ψ = s1 * φ̄; φ̄ = c1 * φ̄
        ρ = sqrt(ρ̄1^2 + β^2); c = ρ̄1 / ρ; s = β / ρ
        θ = s * α; ρ̄ = -c * α; φ = c * φ̄; φ̄ = s * φ̄
        @. x += (φ / ρ) * w
        @. w = v - (θ / ρ) * w
        # stopping estimates (Paige-Saunders)
        res1 = φ̄^2; res2 += ψ^2; rnorm = sqrt(res1 + res2)
        arnorm = α * abs(s * φ)   # ||A' r|| estimate for damp=0
        test1 = rnorm / bnorm
        test2 = arnorm / (Anorm * rnorm)
        xnorm = norm(x)
        (test2 <= atol || test1 <= btol + atol * Anorm * xnorm / bnorm) && break
    end
    return x, (iters = it, test1 = test1, test2 = test2)
end
