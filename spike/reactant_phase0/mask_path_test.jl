# Is the Reactant ACE export's "eager fine, compiled wrong" actually a
# COMPILATION problem -- or do the two paths run different code?
#
# ace_export.jl checks parity with  site_energies(..., em = nothing)
# but core_energy (the compiled entry point) calls site_energies(..., em = mask).
# The masking arithmetic is therefore never exercised by the parity assert.
# This runs BOTH paths eagerly, in plain Julia, with no Reactant involved.
#
# SPIKE CODE.

using ACEpotentials, LinearAlgebra, Random, StaticArrays, Printf
import Polynomials4ML as P4ML
M = ACEpotentials.Models

const NGRID = 32768
const MAX_ATOMS = 64          # small: segment_sum here is a dense one-hot matmul
const MAX_EDGES = 2048

segment_sum(rows, idx, n) = permutedims(idx .== permutedims(1:n)) * rows

monomial_col(u, e) = begin
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])

function hermite(r, grid0, h, n, values, derivs)
    x = clamp.((r .- grid0) ./ h, 0.0, n - 1.0 - 1e-9)
    i = Int32.(floor.(x)); t = x .- i
    y0, y1 = values[i .+ 1, :], values[i .+ 2, :]
    d0, d1 = derivs[i .+ 1, :] .* h, derivs[i .+ 2, :] .* h
    h00 = @. (1 + 2t) * (1 - t)^2; h10 = @. t * (1 - t)^2
    h01 = @. t^2 * (3 - 2t);       h11 = @. t^2 * (t - 1)
    @. h00 * y0 + h10 * d0 + h01 * y1 + h11 * d1
end

function site_energies(positions, centers, neighbors, n_atoms, tb, em)
    vectors = positions[:, neighbors] .- positions[:, centers]
    if em !== nothing
        em_r = permutedims(Float64.(em))
        vectors = vectors .* em_r .+ [1.0, 0.0, 0.0] .* (1.0 .- em_r)
    end
    lengths = sqrt.(sum(abs2, vectors; dims = 1))[1, :]
    u = permutedims(vectors ./ permutedims(lengths))
    rnl = hermite(lengths, HR[1], HR[2], HR[3], tb.rnl, tb.drnl)
    ylm = reduce(hcat, [monomials(u, EXPONENTS[l+1]) * tb.maps[l+1] for l in 0:LMAX])
    edge_a = rnl[:, A_NL] .* ylm[:, A_LM]
    pair_rows = hermite(lengths, HP[1], HP[2], HP[3], tb.rpair, tb.drpair)
    if em !== nothing
        edge_a = Float64.(em) .* edge_a
        pair_rows = Float64.(em) .* pair_rows
    end
    a = segment_sum(edge_a, centers, n_atoms)
    aa = reduce(hcat, [reduce(.*, [a[:, spec[:, k]] for k in 1:size(spec,2)]) for spec in AA_SPECS])
    return (aa * tb.a2b') * tb.wb .+ segment_sum(pair_rows, centers, n_atoms) * tb.wpair .+ E0
end

# ---------------------------------------------------------------- model
model = ace1_model(elements = [:Al], order = 3, totaldegree = 8)
m = model.model; ps, st = model.ps, model.st
rng = MersenneTwister(7)
ps.WB .= 0.02 .* randn(rng, size(ps.WB)); ps.Wpair .= 0.02 .* randn(rng, size(ps.Wpair))
z = m._i2z[1]; RCUT = m.rbasis.rin0cuts[1,1].rcut

function radial_table(basis, ps_b, st_b, rcut)
    grid = collect(range(0.1, rcut - 1e-8, length = NGRID))
    Rnl, dRnl = M.evaluate_ed_batched(basis, grid, z, fill(z, NGRID), ps_b, st_b)
    return grid, Matrix(Rnl), Matrix(dRnl)
end
R_GRID, RNL, DRNL = radial_table(m.rbasis, ps.rbasis, st.rbasis, RCUT)
P_GRID, RPAIR, DRPAIR = radial_table(m.pairbasis, ps.pairbasis, st.pairbasis,
                                     m.pairbasis.rin0cuts[1,1].rcut)
const HR = (R_GRID[1], R_GRID[2]-R_GRID[1], Float64(length(R_GRID)))
const HP = (P_GRID[1], P_GRID[2]-P_GRID[1], Float64(length(P_GRID)))
const A_NL = [t[1] for t in m.tensor.abasis.spec]
const A_LM = [t[2] for t in m.tensor.abasis.spec]
const AA_SPECS = [reduce(hcat, collect.(spec))' for spec in m.tensor.aabasis.specs]
const E0 = m.Vref.E0[z]
const LMAX = isqrt(maximum(A_LM) - 1)
const EXPONENTS = [[(a,b,c) for a in 0:l for b in 0:l for c in 0:l if a+b+c==l] for l in 0:LMAX]
probes = [normalize(SVector{3}(randn(rng,3))) for _ in 1:64]
probe_mat = permutedims(reduce(hcat, [collect(v) for v in probes]))
y_ref = Matrix(P4ML.evaluate(m.ybasis, probes))
YLM_MAPS = map(0:LMAX) do l
    monomials(probe_mat, EXPONENTS[l+1]) \ y_ref[:, l*l+1:(l+1)*(l+1)]
end
TABLES = (rnl=RNL, drnl=DRNL, rpair=RPAIR, drpair=DRPAIR, maps=Tuple(YLM_MAPS),
          a2b=Matrix(m.tensor.A2Bmaps[1]), wb=ps.WB[:,1], wpair=ps.Wpair[:,1])

# ---------------------------------------------------------------- cluster
a0 = 4.05
cellb = [SVector(0.,0.,0.), SVector(0.,.5,.5), SVector(.5,0.,.5), SVector(.5,.5,0.)]
pos = [a0*(SVector(i,j,k)+b) + 0.05*SVector{3}(randn(rng,3))
       for i in 0:1, j in 0:1, k in 0:1 for b in cellb]
n_check = length(pos)
positions = reduce(hcat, [collect(p) for p in pos])
edges = [(i,j) for i in 1:n_check for j in 1:n_check if i != j && norm(pos[i]-pos[j]) < RCUT]
println("n_atoms=$n_check n_edges=$(length(edges)) LMAX=$LMAX")

# reference from ACEpotentials itself
ref_E = 0.0
for i in 1:n_check
    js = [j for j in 1:n_check if j != i && norm(pos[j]-pos[i]) < RCUT]
    Rs = [pos[j]-pos[i] for j in js]
    global ref_E += M.evaluate(m, Rs, fill(z,length(Rs)), z, ps, st)
end

# PATH A: exactly what ace_export.jl's parity assert runs (em = nothing)
siteA = site_energies(positions, Int32.(first.(edges)), Int32.(last.(edges)),
                      n_check, TABLES, nothing)
EA = sum(siteA)

# PATH B: exactly what core_energy runs (padded + edge_mask), eagerly
pos_pad = [positions zeros(3, MAX_ATOMS - n_check)]
senders   = Int32.([first.(edges) .- 1; fill(MAX_ATOMS, MAX_EDGES - length(edges))])
receivers = Int32.([last.(edges)  .- 1; fill(MAX_ATOMS, MAX_EDGES - length(edges))])
edge_mask = [ones(Bool, length(edges)); zeros(Bool, MAX_EDGES - length(edges))]
em_i = Int32.(edge_mask)
centers   = em_i .* (senders   .+ Int32(1)) .+ (Int32(1) .- em_i)
neighbors = em_i .* (receivers .+ Int32(1)) .+ (Int32(1) .- em_i)
siteB = site_energies(pos_pad, centers, neighbors, MAX_ATOMS, TABLES, edge_mask)
EB = sum(siteB[1:n_check])

@printf("ACEpotentials reference E = %.12f eV\n", ref_E)
@printf("PATH A (em = nothing)   E = %.12f   delta = %.3e\n", EA, abs(EA-ref_E))
@printf("PATH B (em = mask)      E = %.12f   delta = %.3e\n", EB, abs(EB-ref_E))
@printf("A vs B                              delta = %.3e\n", abs(EA-EB))
@printf("per-site max |A - B|                      = %.3e\n",
        maximum(abs.(siteA .- siteB[1:n_check])))
@printf("padding sites B[%d:%d]: min=%.3e max=%.3e (E0=%.6f)\n",
        n_check+1, MAX_ATOMS, minimum(siteB[n_check+1:end]),
        maximum(siteB[n_check+1:end]), E0)
