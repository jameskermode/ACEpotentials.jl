# Fixed sqrt(rho) Finnis-Sinclair columns appended to a linear ACE fit.
#
#   julia -p 4 --project=acejax/julia acejax/spike_fs/run_spike.jl
#
# Env: DEGS="4,6" (degrees), NTRAIN=200, NTEST=100, DATASET=cantor|tial,
#      ASMCACHE=<dir> (assembly cache, default scratchpad/fs_spike_cache).
using Distributed
@everywhere using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization
include(joinpath(@__DIR__, "fs_columns.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)
M = ACEpotentials.Models

const DATASET = get(ENV, "DATASET", "cantor")
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = get(ENV, "ASMCACHE", joinpath(SCRATCH, "fs_spike_cache"))
mkpath(CACHEDIR)
const DEGS = parse.(Int, split(get(ENV, "DEGS", "4,6"), ","))
const NTRAIN = parse(Int, get(ENV, "NTRAIN", "200"))
const NTEST = parse(Int, get(ENV, "NTEST", "100"))
const LAMS = 10.0 .^ (0:-1:-8)
const SMOOTH = 4

if DATASET == "cantor"
   DATAFILE = joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz")
   data_all = ACEpotentials.ExtXYZ.load(DATAFILE)
   kw = (energy_key = "mace_energy", force_key = "mace_force", virial_key = "mace_virial")
   ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
   rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
   tr, te = data_all[p[1:NTRAIN]], data_all[p[NTRAIN+1:NTRAIN+NTEST]]
   mkmodel(D) = ace1_model(elements = ELS, order = 3, totaldegree = D)
elseif DATASET == "tial"
   # TiAl_tiny has only 33 structures -- too few to split.  TiAl_tutorial is the
   # same source (329 DFT structures, FLD_TiAl + TiAl_T5000), so use that with
   # the tutorial's hyperparameters (rcut = 5.5, one-body Eref).
   data_all, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
   kw = (energy_key = "energy", force_key = "force", virial_key = "virial")
   ELS = [:Ti, :Al]
   rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
   ntr = min(NTRAIN, round(Int, 0.7 * length(data_all)))
   tr, te = data_all[p[1:ntr]], data_all[p[ntr+1:end]]
   mkmodel(D) = ace1_model(elements = ELS, order = 3, totaldegree = D, rcut = 5.5,
                           Eref = [:Ti => -1586.0195, :Al => -105.5954])
else
   error("unknown DATASET")
end
@info "dataset $DATASET: train $(length(tr))  test $(length(te))"
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]

# ---------------------------------------------------------------- row layout
# per structure: [1 energy; 3*natoms forces; 6 virial], see src/atoms_data.jl
function row_layout(data)
   kind = Int[]; nat = Int[]      # kind: 1=E, 2=F, 3=V
   for sys in data
      n = length(sys)
      push!(kind, 1); push!(nat, n)
      append!(kind, fill(2, 3n)); append!(nat, fill(n, 3n))
      append!(kind, fill(3, 6)); append!(nat, fill(n, 6))
   end
   return kind, nat
end
function rmse_efv(resid, kind, nat)
   r = copy(resid)
   r[kind .!= 2] ./= nat[kind .!= 2]          # energy and virial per atom
   f(k) = sqrt(sum(abs2, r[kind .== k]) / count(==(k), kind))
   return (F = f(2), E = f(1), V = f(3))
end

# ---------------------------------------------------------------- assembly (cached)
function assemble_cached(tag, data, model)
   f = joinpath(CACHEDIR, "asm_$(DATASET)_$(tag).jls")
   if isfile(f)
      @info "  reusing $(basename(f))"; return deserialize(f)
   end
   t = @elapsed (A, Y, W) = ACEpotentials.assemble(data, model; kw...)
   @info @sprintf("  assembled %s %s in %.0f s", tag, size(A), t)
   A = Matrix(A)
   serialize(f, (A, Y, W)); return (A, Y, W)
end

asm = Dict{Int, Any}()
for D in DEGS
   m = mkmodel(D)
   rc = unique([x.rcut for x in m.model.rbasis.rin0cuts])
   @info "degree $D: length_basis=$(M.length_basis(m))  rcut=$rc"
   Atr, Ytr, Wtr = assemble_cached("D$(D)_train$(length(tr))", tr, m)
   Ate, Yte, Wte = assemble_cached("D$(D)_test$(length(te))", te, m)
   asm[D] = (Atr, Ytr, Wtr, Ate, Yte, Wte)
end
nworkers() > 1 && (rmprocs(workers()); GC.gc())

# ---------------------------------------------------------------- FS columns
# rcut/r0 of the FS densities follow the ACE model's own uniform cutoff.
m0 = mkmodel(DEGS[1])
RCUT = maximum(x.rcut for x in m0.model.rbasis.rin0cuts)
R0 = maximum(x.r0 for x in m0.model.rbasis.rin0cuts)
@info "FS densities: r0=$R0  rcut=$RCUT"
fsvariants = [
   ("+ rho, K=3 (control)",     FSSpec(ZS; phi = :linear, r0 = R0, rcut = RCUT)),
   ("+ sqrt(rho), K=3",         FSSpec(ZS; phi = :sqrt,   r0 = R0, rcut = RCUT)),
   ("+ sqrt(rho), K=1 (a=4)",   FSSpec(ZS; phi = :sqrt,   r0 = R0, rcut = RCUT, alphas = [4.0])),
   ("+ sqrt(rho), K=3 shared",  FSSpec(ZS; phi = :sqrt,   r0 = R0, rcut = RCUT, shared = true)),
   ("+ rho, K=3 shared (ctrl)", FSSpec(ZS; phi = :linear, r0 = R0, rcut = RCUT, shared = true)),
]
Xcache = Dict{String, Any}()
for (name, fs) in fsvariants
   t = @elapsed Xcache[name] = (fs_feature_matrix(fs, tr), fs_feature_matrix(fs, te))
   @info @sprintf("  FS columns %-28s ncol=%3d  built in %.1f s", name, ncolumns(fs), t)
end

kind_tr, nat_tr = row_layout(tr)
kind_te, nat_te = row_layout(te)

# ---------------------------------------------------------------- fits
results = []
function fit_and_report(label, D, Aw, Yw, Wtr, Atr_raw, Ate_raw, Ytr, Yte, Pdiag, X;
                        xscale = 1.0)
   # Aw: weighted, prior-scaled ACE columns (train). X = (Xtr, Xte) or nothing.
   nace = size(Aw, 2)
   if X === nothing
      Bw = Aw; ncol = 0
   else
      Xtr, Xte = X
      ncol = size(Xtr, 2)
      Bw = hcat(Aw, (Wtr .* Xtr) .* reshape(xscale .* ones(ncol), 1, :))
   end
   T = TikhonovFactor(Bw, Yw)
   best = nothing
   for lam in LAMS
      z = tikhonov_solve(T, lam)
      c_ace = z[1:nace] ./ Pdiag                     # x = P \ z
      pred_te = Ate_raw * c_ace
      pred_tr = Atr_raw * c_ace
      if ncol > 0
         c_x = z[nace+1:end] .* xscale
         pred_te .+= Xte * c_x
         pred_tr .+= Xtr * c_x
      end
      ete = rmse_efv(pred_te .- Yte, kind_te, nat_te)
      etr = rmse_efv(pred_tr .- Ytr, kind_tr, nat_tr)
      @info @sprintf("    %-28s lam=%-6g test F=%.4f E=%.5f V=%.4f | train F=%.4f",
                     label, lam, ete.F, ete.E, ete.V, etr.F)
      if best === nothing || ete.F < best.te.F
         best = (te = ete, tr = etr, lam = lam, c_ace = c_ace)
      end
   end
   push!(results, (label = label, D = D, n = nace + ncol, te = best.te, tr = best.tr, lam = best.lam))
   @printf("RES %-32s D=%d ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lambda=%g\n",
           label, D, nace + ncol, best.te.F, best.te.E, best.te.V, best.tr.F, best.lam)
   flush(stdout)
   return best
end

for D in DEGS
   Atr, Ytr, Wtr, Ate, Yte, Wte = asm[D]
   m = mkmodel(D)
   @assert size(Atr, 1) == length(kind_tr) "train row count mismatch"
   @assert size(Ate, 1) == length(kind_te) "test row count mismatch"
   P = M.algebraic_smoothness_prior(m.model; p = SMOOTH)
   Pdiag = P.diag
   Aw = (Atr ./ reshape(Pdiag, 1, :)) .* Wtr
   Yw = Wtr .* Ytr
   # --- baseline, and the bookkeeping check against compute_errors
   best = fit_and_report("linear ACE", D, Aw, Yw, Wtr, Atr, Ate, Ytr, Yte, Pdiag, nothing)
   M.set_linear_parameters!(m, best.c_ace)
   err = ACEpotentials.compute_errors(te, m; kw..., verbose = false)["rmse"]["set"]
   @printf("CHECK D=%d compute_errors: F=%.6f E=%.6f V=%.6f | mine: F=%.6f E=%.6f V=%.6f | max diff %.2e\n",
           D, err["F"], err["E"], err["V"], best.te.F, best.te.E, best.te.V,
           maximum(abs, [err["F"] - best.te.F, err["E"] - best.te.E, err["V"] - best.te.V]))
   flush(stdout)
   # --- augmented fits.  xscale: FS columns are rescaled so that each weighted
   # column has the median 2-norm of the weighted, prior-scaled ACE columns;
   # this puts them on the same footing under the single Tikhonov lambda.
   # (A raw, unscaled run is included for the main variant to show whether it
   # matters.)
   colnorms = [sqrt(sum(abs2, c)) for c in eachcol(Aw)]
   med = sort(colnorms)[div(end, 2)]
   @info @sprintf("  ACE column norms (weighted, prior-scaled): median %.3g, range %.3g..%.3g",
                  med, minimum(colnorms), maximum(colnorms))
   for (name, fs) in fsvariants
      Xtr, Xte = Xcache[name]
      xn = [sqrt(sum(abs2, Wtr .* c)) for c in eachcol(Xtr)]
      xscale = med ./ max.(xn, 1e-12)
      fit_and_report(name, D, Aw, Yw, Wtr, Atr, Ate, Ytr, Yte, Pdiag, (Xtr, Xte); xscale)
      if name == "+ sqrt(rho), K=3"
         fit_and_report(name * " [unscaled]", D, Aw, Yw, Wtr, Atr, Ate, Ytr, Yte, Pdiag, (Xtr, Xte))
      end
   end
end

println("\n==== SUMMARY ($DATASET, train $(length(tr)), test $(length(te))) ====")
@printf("%-40s %5s %8s %9s %8s %8s %8s\n", "model", "ncol", "test F", "test E", "test V", "train F", "lambda")
for r in results
   @printf("%-40s %5d %8.4f %9.5f %8.4f %8.4f %8g\n", "D=$(r.D) " * r.label, r.n,
           r.te.F, r.te.E, r.te.V, r.tr.F, r.lam)
end
