# Tensor-train coefficient format over the neighbour-species index, fitted by
# alternating linear least squares on the cached categorical design matrix.
#
# Per central species z0 and body order ν (separate trains per order):
#   c^{(z0)}[(β,η), ζ] = G_1[ζ_1] G_2[ζ_2] ⋯ G_ν[ζ_ν] v^{(z0)}_{β,η}
#   G_t[z] ∈ R^{r_{t-1} × r_t}, r_0 = 1 (left boundary absorbed into G_1),
#   v ∈ R^{r_ν} per coupling block (β, η) -- the "λ readout".
# Cores are shared across z0 (share_z0 = true; CP's E[z,k] is too) or per z0.
# CP-K (the frozen embedding) is the special case G_1[z] = E[z,:], G_t[z] = diag(E[z,:]),
# bond dims (1, K, ..., K).
#
# Categorical coefficient vector: x = Φ c  (Φ from ttmap.jl, block-diagonal over z0).
# Objective (the categorical one, pulled back):
#   J = ‖W(A_MB Φ c + A_pair p − y)‖² + λ² (‖P_MB Φ c‖² + ‖P_pair p‖²)
# With all but one core (or v) frozen the model is linear in that block; every
# ALS step is a generalised-Tikhonov least-squares solve of the augmented system.

using LinearAlgebra, SparseArrays, Random, Printf

struct TTSpec
   S::Int
   nz0::Int
   ranks::Vector{Vector{Int}}     # ranks[ν] = [1, r_1, ..., r_ν]
   share_z0::Bool
end
nz0c(sp::TTSpec) = sp.share_z0 ? 1 : sp.nz0

mutable struct TTModel
   sp::TTSpec
   tm::TTMap
   G::Vector{Vector{Array{Float64,4}}}     # G[ν][t] :: (r_{t-1}, r_t, S, nz0c)
   v::Vector{Vector{Matrix{Float64}}}      # v[z0][ib] :: (r_ν, nη)
   p::Vector{Float64}                      # pair coefficients
end

n_core_params(m::TTModel) = sum(length(G) for Gs in m.G for G in Gs)
n_v_params(m::TTModel) = sum(length(v) for vs in m.v for v in vs)
n_params(m::TTModel) = n_core_params(m) + n_v_params(m)

function TTModel(sp::TTSpec, tm::TTMap, npair; rng = MersenneTwister(0), init = :zeros)
   νmax = length(sp.ranks)
   G = [[init == :zeros ? zeros(sp.ranks[ν][t], sp.ranks[ν][t+1], sp.S, nz0c(sp)) :
         randn(rng, sp.ranks[ν][t], sp.ranks[ν][t+1], sp.S, nz0c(sp)) / sqrt(sp.ranks[ν][t] * sp.S)
         for t in 1:ν] for ν in 1:νmax]
   v = [[zeros(sp.ranks[order(tm, ib)][end], n_eta(tm, ib)) for ib in 1:length(tm.betas)]
        for _ in 1:sp.nz0]
   TTModel(sp, tm, G, v, zeros(npair))
end

Base.copy(m::TTModel) = TTModel(m.sp, m.tm, [[copy(G) for G in Gs] for Gs in m.G],
                                [[copy(v) for v in vs] for vs in m.v], copy(m.p))

# ---------------------------------------------------------------- coefficients
# c is z0-major: c[(z0-1)*N_c + j]
function coefficients(m::TTModel)
   tm = m.tm; sp = m.sp; Nc = length(tm.idx)
   c = zeros(sp.nz0 * Nc)
   for z0 in 1:sp.nz0
      iz = sp.share_z0 ? 1 : z0
      for (j, ix) in enumerate(tm.idx)
         ν = length(ix.ζ)
         row = ones(1, 1)
         for t in 1:ν
            row = row * @view(m.G[ν][t][:, :, ix.ζ[t], iz])
         end
         c[(z0-1)*Nc + j] = (row * @view(m.v[z0][ix.ibeta][:, ix.η]))[1]
      end
   end
   return c
end

# ---------------------------------------------------------------- data bundle
struct TTData
   AΦ::Matrix{Float64}      # m × nz0*N_c  (weighted, MB columns mapped)
   Apair::Matrix{Float64}   # m × npair    (weighted)
   y::Vector{Float64}       # weighted targets
   PΦ::SparseMatrixCSC{Float64,Int}   # n_MB × nz0*N_c   (prior-scaled map)
   Ppair::Vector{Float64}
end

function objective(m::TTModel, d::TTData, λ)
   c = coefficients(m)
   r = d.AΦ * c + d.Apair * m.p - d.y
   return sum(abs2, r) + λ^2 * (sum(abs2, d.PΦ * c) + sum(abs2, d.Ppair .* m.p))
end

# generalised-Tikhonov solve of  min ‖A x − y‖² + λ²‖R x‖².
# Exact least squares (thin QR, then SVD of the triangular factor with a relative
# cutoff), so that a step can never increase the objective: Julia's `\` uses a
# pivoted QR whose rank truncation is NOT a minimiser when the current iterate
# lives in ill-conditioned directions (it produced a 1e8 jump in the O1 check).
# Exactly-zero columns (dead bond directions) are dropped and get x = 0.
function tikh_solve(A, R, y, λ; cutoff = 1e-12, ry = zeros(size(R, 1)))
   B = vcat(A, λ .* R)
   yy = vcat(y, λ .* ry)
   nz = findall(j -> any(!=(0.0), @view B[:, j]), 1:size(B, 2))
   Bn = length(nz) == size(B, 2) ? B : B[:, nz]
   F = qr(Bn)
   k = min(size(Bn)...)
   c = (F.Q' * yy)[1:k]
   U, Σ, V = svd(Matrix(F.R)[1:k, :])
   keep = Σ .> cutoff * Σ[1]
   xn = V[:, keep] * ((U[:, keep]' * c) ./ Σ[keep])
   x = zeros(size(B, 2)); x[nz] .= xn
   return x
end

# ---------------------------------------------------------------- ALS steps
# Contraction map for core slot t: c = M_t g, g = vec of all G[ν][t] (ν ≥ t).
function core_map(m::TTModel, t)
   tm = m.tm; sp = m.sp; Nc = length(tm.idx); νmax = length(sp.ranks)
   # parameter offsets
   off = Dict{Int,Int}(); n_g = 0
   for ν in t:νmax
      off[ν] = n_g; n_g += length(m.G[ν][t])
   end
   I_ = Int[]; J_ = Int[]; V_ = Float64[]
   for z0 in 1:sp.nz0
      iz = sp.share_z0 ? 1 : z0
      for (j, ix) in enumerate(tm.idx)
         ν = length(ix.ζ); ν >= t || continue
         L = ones(1, 1)
         for s in 1:t-1; L = L * @view(m.G[ν][s][:, :, ix.ζ[s], iz]); end
         R = m.v[z0][ix.ibeta][:, ix.η]
         for s in ν:-1:t+1; R = @view(m.G[ν][s][:, :, ix.ζ[s], iz]) * R; end
         ra, rb = size(m.G[ν][t], 1), size(m.G[ν][t], 2)
         row = (z0-1)*Nc + j
         for b in 1:rb, a in 1:ra
            val = L[a] * R[b]
            abs(val) > 0 || continue
            # linear index of G[ν][t][a, b, ζ_t, iz]
            col = off[ν] + a + ra * ((b-1) + rb * ((ix.ζ[t]-1) + sp.S * (iz-1)))
            push!(I_, row); push!(J_, col); push!(V_, val)
         end
      end
   end
   return sparse(I_, J_, V_, sp.nz0 * Nc, n_g), off
end

function set_core!(m::TTModel, t, g, off)
   for ν in t:length(m.sp.ranks)
      n = length(m.G[ν][t])
      m.G[ν][t][:] .= g[off[ν]+1:off[ν]+n]
   end
end

# Contraction map for the readout: c = M_v w, w = all v[z0][ib][:, η]
function readout_map(m::TTModel)
   tm = m.tm; sp = m.sp; Nc = length(tm.idx)
   off = zeros(Int, sp.nz0, length(tm.betas)); n_w = 0
   for z0 in 1:sp.nz0, ib in 1:length(tm.betas)
      off[z0, ib] = n_w; n_w += length(m.v[z0][ib])
   end
   I_ = Int[]; J_ = Int[]; V_ = Float64[]
   for z0 in 1:sp.nz0
      iz = sp.share_z0 ? 1 : z0
      for (j, ix) in enumerate(tm.idx)
         ν = length(ix.ζ)
         T = ones(1, 1)
         for s in 1:ν; T = T * @view(m.G[ν][s][:, :, ix.ζ[s], iz]); end
         rν = length(T); row = (z0-1)*Nc + j
         for b in 1:rν
            abs(T[b]) > 0 || continue
            col = off[z0, ix.ibeta] + b + rν * (ix.η - 1)
            push!(I_, row); push!(J_, col); push!(V_, T[b])
         end
      end
   end
   return sparse(I_, J_, V_, sp.nz0 * Nc, n_w), off
end

function set_readout!(m::TTModel, w, off)
   for z0 in 1:m.sp.nz0, ib in 1:length(m.tm.betas)
      n = length(m.v[z0][ib])
      m.v[z0][ib][:] .= w[off[z0, ib]+1:off[z0, ib]+n]
   end
end

# one linear step on the block described by (Mmap); pair coefficients solved
# jointly.  Coefficients NOT touched by the block (orders ν < t for a core step)
# are constants: their contribution moves to the right-hand side of both the
# data rows and the regulariser rows.
function block_step!(m::TTModel, d::TTData, λ, Mmap, setter!)
   c = coefficients(m)
   touched = vec(any(!=(0.0), Mmap; dims = 2))
   cfix = copy(c); cfix[touched] .= 0
   A = hcat(d.AΦ * Matrix(Mmap), d.Apair)
   R = hcat(Matrix(d.PΦ * Mmap), zeros(size(d.PΦ, 1), size(d.Apair, 2)))
   R = vcat(R, hcat(zeros(length(d.Ppair), size(Mmap, 2)), Diagonal(d.Ppair)))
   ry = vcat(-(d.PΦ * cfix), zeros(length(d.Ppair)))
   x = tikh_solve(A, R, d.y - d.AΦ * cfix, λ; ry = ry)
   ng = size(Mmap, 2)
   setter!(x[1:ng]); m.p .= x[ng+1:end]
   return ng
end

function core_step!(m, d, λ, t)
   Mt, off = core_map(m, t)
   block_step!(m, d, λ, Mt, g -> set_core!(m, t, g, off))
end
function readout_step!(m, d, λ)
   Mv, off = readout_map(m)
   block_step!(m, d, λ, Mv, w -> set_readout!(m, w, off))
end

# gauge: left-orthogonalise core t (push R into core t+1, or into v if t = ν)
function left_orth!(m::TTModel, ν, t)
   G = m.G[ν][t]; ra, rb, S, nz = size(G)
   for iz in 1:nz
      Mt = reshape(permutedims(G[:, :, :, iz], (1, 3, 2)), ra * S, rb)   # (a,z) × b
      F = qr(Mt); Q = Matrix(F.Q)[:, 1:min(ra*S, rb)]; Rm = F.R[1:min(ra*S, rb), :]
      # if rb > ra*S the core is rank-deficient in b; keep shapes by padding Q, R
      if size(Q, 2) < rb
         Q = hcat(Q, zeros(ra*S, rb - size(Q, 2))); Rm = vcat(Rm, zeros(rb - size(Rm, 1), rb))
      end
      G[:, :, :, iz] .= permutedims(reshape(Q, ra, S, rb), (1, 3, 2))
      if t < ν
         for z in 1:S; m.G[ν][t+1][:, :, z, iz] .= Rm * m.G[ν][t+1][:, :, z, iz]; end
      else
         for z0 in 1:m.sp.nz0
            (m.sp.share_z0 || z0 == iz) || continue
            for ib in 1:length(m.tm.betas)
               order(m.tm, ib) == ν || continue
               m.v[z0][ib] .= Rm * m.v[z0][ib]
            end
         end
      end
   end
end

# ---------------------------------------------------------------- ALS driver
function als!(m::TTModel, d::TTData, λ; nsweeps = 10, rtol = 1e-4, log = stdout,
              orth = true, callback = nothing, tag = "")
   νmax = length(m.sp.ranks)
   trace = Float64[]
   J0 = objective(m, d, λ); push!(trace, J0)
   @printf(log, "%s  init  J = %.10e\n", tag, J0); flush(log)
   nonmono = 0
   function record(what, tstep)
      J = objective(m, d, λ)
      if J > trace[end] * (1 + 1e-10)
         nonmono += 1
         @printf(log, "%s  NON-MONOTONE step %s: J %.10e -> %.10e (rel +%.2e)\n", tag, what, trace[end], J, J / trace[end] - 1)
      end
      push!(trace, J)
      @printf(log, "%s  %-6s J = %.10e  (%.1f s)\n", tag, what, J, tstep); flush(log)
   end
   for sweep in 1:nsweeps
      Jstart = trace[end]
      t0 = time()
      for t in 1:νmax                      # right sweep
         ts = @elapsed core_step!(m, d, λ, t)
         orth && t < νmax && for ν in t:νmax; left_orth!(m, ν, t); end
         record("G$t", ts)
      end
      ts = @elapsed readout_step!(m, d, λ); record("v", ts)
      for t in νmax:-1:1                   # left sweep (no re-gauging; v absorbs)
         ts = @elapsed core_step!(m, d, λ, t); record("G$t", ts)
      end
      ts = @elapsed readout_step!(m, d, λ); record("v", ts)
      callback === nothing || callback(sweep, trace[end])
      @printf(log, "%s sweep %d: J = %.10e  rel change %.3e  wall %.1f s\n", tag, sweep,
              trace[end], 1 - trace[end] / Jstart, time() - t0); flush(log)
      (Jstart - trace[end]) / Jstart < rtol && break
   end
   return trace, nonmono
end

# ---------------------------------------------------------------- init by TT-SVD
# TT-SVD of a dense species tensor X[ζ, k] (S^ν × K, ζ_1 fastest) to the given
# ranks [1, r_1, ..., r_ν]; returns cores G_t (r_{t-1}, r_t, S) and the r_ν × K
# matrix R with  G_1[ζ_1]⋯G_ν[ζ_ν] R ≈ X[ζ, :].  Exact when the ranks are at least
# the unfolding ranks; otherwise the truncation error is reported.
function tt_svd(X::Matrix{Float64}, S::Int, ranks::Vector{Int})
   ν = length(ranks) - 1; K = size(X, 2)
   @assert size(X, 1) == S^ν
   cores = Array{Float64,3}[]
   Mcur = Matrix(X); rprev = 1
   for t in 1:ν
      Mt = reshape(Mcur, rprev * S, :)
      F = svd(Mt)
      r = min(ranks[t+1], count(>(1e-13 * F.S[1]), F.S))
      U = F.U[:, 1:r]; Σ = F.S[1:r]; Vt = F.Vt[1:r, :]
      trunc = r < length(F.S) ? sqrt(sum(abs2, F.S[r+1:end]) / sum(abs2, F.S)) : 0.0
      trunc > 1e-12 && @info @sprintf("  tt_svd: slot %d truncated rank %d -> %d, rel. err %.2e", t, length(F.S), r, trunc)
      Gt = zeros(rprev, ranks[t+1], S)
      Gt[:, 1:r, :] .= permutedims(reshape(U, rprev, S, r), (1, 3, 2))
      push!(cores, Gt)
      Mcur = zeros(ranks[t+1], size(Vt, 2)); Mcur[1:r, :] .= Diagonal(Σ) * Vt
      rprev = ranks[t+1]
   end
   return cores, reshape(Mcur, rprev, K)
end

# the CP-K species tensor Π_t E[ζ_t, k] as an S^ν × K matrix
function cp_tensor(E::Matrix{Float64}, ν::Int)
   S, K = size(E)
   X = zeros(S^ν, K)
   for (i, ζ) in enumerate(Iterators.product(ntuple(_ -> 1:S, ν)...))
      for k in 1:K; X[i, k] = prod(E[ζ[t], k] for t in 1:ν); end
   end
   return X
end
# the species tensor T(ζ) = G_1[ζ_1]⋯G_ν[ζ_ν] of an existing model (shared cores)
function tt_tensor(m::TTModel, ν::Int; iz = 1)
   S = m.sp.S; rν = m.sp.ranks[ν][end]
   X = zeros(S^ν, rν)
   for (i, ζ) in enumerate(Iterators.product(ntuple(_ -> 1:S, ν)...))
      T = ones(1, 1)
      for t in 1:ν; T = T * @view(m.G[ν][t][:, :, ζ[t], iz]); end
      X[i, :] .= vec(T)
   end
   return X
end

# initialise cores (shared across z0) from dense tensors Xs[ν] (S^ν × K_ν) and a
# readout vsrc[z0][ib] (K_ν × nη); v_TT = R * vsrc.  `noise` perturbs the cores
# (relative to their max entry) so that zero-padded bond directions are not dead.
function init_from_tensor!(m::TTModel, Xs::Vector{Matrix{Float64}}, vsrc; noise = 0.0, rng = MersenneTwister(1))
   sp = m.sp
   for ν in 1:length(sp.ranks)
      cores, R = tt_svd(Xs[ν], sp.S, sp.ranks[ν])
      for t in 1:ν, iz in 1:nz0c(sp)
         m.G[ν][t][:, :, :, iz] .= cores[t]
         noise > 0 && (m.G[ν][t][:, :, :, iz] .+= noise * maximum(abs, cores[t]) .* randn(rng, size(cores[t])...))
      end
      for z0 in 1:sp.nz0, ib in 1:length(m.tm.betas)
         order(m.tm, ib) == ν || continue
         m.v[z0][ib] .= R * vsrc[z0][ib]
      end
   end
   return m
end

# ---------------------------------------------------------------- flop count
# per-site cost model (multiply-adds), see FINDINGS: per 𝔸-DAG node at depth t the
# TT extends a prefix state by one slot: r_{t-1} r_t ; CP: K.  The embedding of
# the one-particle basis into the format costs S r_{t-1} r_t (TT, per slot) or
# S K (CP) per (n', l, m).  Readout r_ν (TT) / K (CP) per (β,η).
function flops_per_site(sp::TTSpec, n_nodes::Vector{Vector{Int}}, n_A1::Int, n_blocks::Vector{Int})
   νmax = length(sp.ranks)
   f = 0.0
   for ν in 1:νmax
      r = sp.ranks[ν]
      for t in 1:ν
         f += n_A1 * sp.S * r[t] * r[t+1]                 # embedding, slot t
      end
      f += sum(n_nodes[ν][t] * r[t] * r[t+1] for t in 1:ν) # DAG nodes at depth t of order-ν train
      f += n_blocks[ν] * r[ν+1]                            # readout
   end
   return f
end
