# Measure the conditioning *structure* of a real ACE design matrix:
# raw A, A/P (what ACEfit.QR sees), and A/P after column equilibration
# (what LSQR with a diagonal right-preconditioner would see).
using ACEpotentials, ExtXYZ, LazyArtifacts, LinearAlgebra, Printf, Statistics
ACE1compat = ACEpotentials.ACE1compat
using Pkg.Artifacts
art = Pkg.Artifacts.ensure_artifact_installed("Si_tiny_dataset", joinpath(pkgdir(ACEpotentials), "Artifacts.toml"))
data = ExtXYZ.load(joinpath(art, "Si_tiny.xyz"))
keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
for totaldegree in (10, 14, 18)
    model = ACE1compat.ace1_model(; elements=[:Si], Eref=[:Si => -158.54496821], rcut=5.5, order=3, totaldegree)
    A, Y, W = ACEpotentials.assemble(data, model; weights, keys...)
    m, n = size(A)
    P = ACEpotentials.Models.algebraic_smoothness_prior(model.model; p = 4)
    Ap = Diagonal(W) * (A / P)
    cn = vec(sqrt.(sum(abs2, Ap; dims=1)))
    Ae = Ap ./ cn'
    s_raw = svdvals(A); s_p = svdvals(Ap); s_e = svdvals(Ae)
    @printf("\ndegree=%d  A is %d x %d (%.1f MB)\n", totaldegree, m, n, m*n*8/1e6)
    @printf("  cond(A raw)                          = %.3e\n", s_raw[1]/s_raw[end])
    @printf("  cond(W*(A/P))   [what ACEfit.QR sees] = %.3e\n", s_p[1]/s_p[end])
    @printf("  column-norm spread of W*(A/P): max/min = %.3e\n", maximum(cn)/minimum(cn))
    @printf("  cond(W*(A/P) column-equilibrated)    = %.3e   <- what LSQR+diag precond would see\n", s_e[1]/s_e[end])
    # how many singular values below 1e-8, 1e-12 relative?
    for t in (1e-8, 1e-12, 1e-14)
        @printf("  equilibrated: #sv < %.0e*max = %d of %d\n", t, count(s_e .< t*s_e[1]), n)
    end
    flush(stdout)
end
