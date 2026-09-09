# Shared setup: builds the ace1_model, the Hermite radial tables, the monomial
# -> Ylm maps and the padded test cluster, mirroring lammps-jax
# dev/julia_export examples/julia/ace_export.jl. SPIKE CODE.
# Expects monomials/hermite/segment_sum to be defined by the caller.

model = ace1_model(elements = [:Al], order = 3, totaldegree = 8)
m = model.model; ps, st = model.ps, model.st
rng = MersenneTwister(7)
ps.WB .= 0.02 .* randn(rng, size(ps.WB)); ps.Wpair .= 0.02 .* randn(rng, size(ps.Wpair))
z = m._i2z[1]; RCUT = m.rbasis.rin0cuts[1,1].rcut

function _radial_table(basis, ps_b, st_b, rcut)
    grid = collect(range(0.1, rcut - 1e-8, length = NGRID))
    Rnl, dRnl = M.evaluate_ed_batched(basis, grid, z, fill(z, NGRID), ps_b, st_b)
    return grid, Matrix(Rnl), Matrix(dRnl)
end
R_GRID, RNL, DRNL = _radial_table(m.rbasis, ps.rbasis, st.rbasis, RCUT)
P_GRID, RPAIR, DRPAIR = _radial_table(m.pairbasis, ps.pairbasis, st.pairbasis,
                                      m.pairbasis.rin0cuts[1,1].rcut)
const HR = (R_GRID[1], R_GRID[2]-R_GRID[1], Float64(length(R_GRID)))
const HP = (P_GRID[1], P_GRID[2]-P_GRID[1], Float64(length(P_GRID)))
const A_NL = [t[1] for t in m.tensor.abasis.spec]
const A_LM = [t[2] for t in m.tensor.abasis.spec]
const AA_SPECS = [reduce(hcat, collect.(spec))' for spec in m.tensor.aabasis.specs]
const E0 = m.Vref.E0[z]
const LMAX = isqrt(maximum(A_LM) - 1)
const EXPONENTS = [[(a,b,c) for a in 0:l for b in 0:l for c in 0:l if a+b+c==l] for l in 0:LMAX]

_probes = [normalize(SVector{3}(randn(rng,3))) for _ in 1:64]
_pm = permutedims(reduce(hcat, [collect(v) for v in _probes]))
_yref = Matrix(P4ML.evaluate(m.ybasis, _probes))
const YLM_MAPS = map(0:LMAX) do l
    monomials(_pm, EXPONENTS[l+1]) \ _yref[:, l*l+1:(l+1)*(l+1)]
end
const TABLES = (rnl=RNL, drnl=DRNL, rpair=RPAIR, drpair=DRPAIR,
                maps=Tuple(YLM_MAPS), a2b=Matrix(m.tensor.A2Bmaps[1]),
                wb=ps.WB[:,1], wpair=ps.Wpair[:,1])

_a0 = 4.05
_cb = [SVector(0.,0.,0.), SVector(0.,.5,.5), SVector(.5,0.,.5), SVector(.5,.5,0.)]
pos = [_a0*(SVector(i,j,k)+b) + 0.05*SVector{3}(randn(rng,3))
       for i in 0:1, j in 0:1, k in 0:1 for b in _cb]
n_check = length(pos)
positions = reduce(hcat, [collect(p) for p in pos])
edges = [(i,j) for i in 1:n_check for j in 1:n_check if i != j && norm(pos[i]-pos[j]) < RCUT]
pos_pad = [positions zeros(3, MAX_ATOMS - n_check)]
_senders   = Int32.([first.(edges) .- 1; fill(MAX_ATOMS, MAX_EDGES - length(edges))])
_receivers = Int32.([last.(edges)  .- 1; fill(MAX_ATOMS, MAX_EDGES - length(edges))])
edge_mask  = [ones(Bool, length(edges)); zeros(Bool, MAX_EDGES - length(edges))]
_emi = Int32.(edge_mask)
centers   = _emi .* (_senders   .+ Int32(1)) .+ (Int32(1) .- _emi)
neighbors = _emi .* (_receivers .+ Int32(1)) .+ (Int32(1) .- _emi)
println("setup: n_atoms=$n_check n_edges=$(length(edges)) LMAX=$LMAX rcut=$RCUT")
