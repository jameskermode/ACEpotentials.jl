# Distil MACE-MP-0-small into ACE models of the Cantor alloy, three ways:
#   1. categorical ace1_model            -- the reference
#   2. embedded, LOSSLESS widths         -- an exact reparameterisation: it must
#                                           match, or the fault is conditioning
#   3. embedded, capped d_max            -- what truncation actually costs
#
# lambda is swept PER MODEL and the best held-out error reported.  Comparing
# regularised fits at one shared lambda is tuning one side, and doing that once
# made a 16-20% gap look like 2x.
# Run with `julia -p N`: ACEfit parallelises assembly over PROCESSES, and a
# single process leaves this CPU-bound for no reason.
using Distributed
@everywhere using ACEpotentials, ACEfit

# FASTBASIS=1: replace ACEfit.feature_matrix with the type-stable pushforward
# path from the assembly profile (docs/findings/FINDINGS_assembly_profile.md):
# 65-150x per structure, exact to 1e-12 on the design matrix. The original
# `evaluate_basis_ed` is type-unstable (ForwardDiff Jacobian + collect chain
# inferred as Any) and its consumer loop boxes an SVector per element -- 1e8
# allocations per structure. Self-checked below on the first two structures.
const FASTBASIS = get(ENV, "FASTBASIS", "0") == "1"
if FASTBASIS
   @everywhere begin
      include(joinpath(@__DIR__, "..", "profile_assembly", "basis_opt.jl"))
      using ACEpotentials: AtomsData
      using ACEpotentials.Models: ACEPotential, ACEModel, length_basis
      const _PFWS = Dict{UInt, Any}()
      const _FEATURE_ORIG = Ref(false)      # set true to route to the original
      function ACEfit.feature_matrix(d::AtomsData, calc::ACEPotential{<: ACEModel}; kwargs...)
         _FEATURE_ORIG[] &&
            return invoke(ACEfit.feature_matrix, Tuple{AtomsData, Any}, d, calc; kwargs...)
         ws = get!(_PFWS, objectid(calc)) do; PFWorkspace(calc.model, 100); end
         efv = efv_basis_v4(d.system, calc; ws = ws)
         return feature_matrix_from_efv(efv, length(d.system), length_basis(calc);
                                        has_E = !isnothing(d.energy_key),
                                        has_F = !isnothing(d.force_key),
                                        has_V = !isnothing(d.virial_key))
      end
   end
end
using Random, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "tikhonov.jl"))

# The assembly is parallel over WORKERS; the solve is a single QR on the main
# process using BLAS threads.  These are different phases, so BOTH are wanted:
# `julia -p N` for assembly and BLASTHREADS for the solve.  Running with no
# workers to "give BLAS the cores" left a 101600 x 5990 assembly running
# serially for over an hour -- the workers idle during the solve, but they
# cost only ~1.5 GB each, which on a 256 GB node is nothing.
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", string(Sys.CPU_THREADS))))
@info "BLAS threads: $(BLAS.get_num_threads())"
M = ACEpotentials.Models

const DATAFILE = ENV["DATA"]
const CACHEDIR = get(ENV, "ASMCACHE", "asm_cache")
data_all = ACEpotentials.ExtXYZ.load(DATAFILE)
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
ntr = round(Int, 0.8 * length(data_all))
tr, te = data_all[p[1:ntr]], data_all[p[ntr+1:end]]
@info "train $(length(tr))  test $(length(te))"
# Energies, forces AND virials.  The virial convention was verified rather than
# assumed: MACE's stress is ASE's, sigma = (1/V) dU/deps (finite-difference
# check: FD dE/deps = 43.936 eV against V*tr(sigma) = 44.166, 0.5% apart at
# h=0.004), and ACEpotentials' virial is -sum dV_i (x) R_i = -dU/deps
# (calculators.jl:105).  So virial = -V * sigma.
#
# Virials are only ~6% of the observations but carry most of the information
# about the strain response, and this structure set is deliberately strained --
# without them the only volume information is one energy per structure.
kw = (energy_key = "mace_energy", force_key = "mace_force",
      virial_key = "mace_virial")
emb = M.read_mace_embedding(ENV["EMB"])
# cache-key tag for the embedding; "" keeps the pre-existing (MP-0-small) names
const EMBTAG = get(ENV, "EMBTAG", "")
ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
ORD, DEG = 3, parse(Int, get(ENV, "DEG", "6"))
# Assembly is shared across lambda, so a denser and WIDER sweep is nearly free.
# The previous sweep topped out at 1e-2 and three of four models chose it, i.e.
# the optimum was at or beyond the boundary -- those numbers were lower bounds.
LAMS = 10.0 .^ (0:-0.5:-9)          # 19 points; each is O(n^2) now
SMOOTH = parse(Float64, get(ENV, "SMOOTH", "4"))   # acefit!'s default

function sweep(name, mk)
   @info "fitting $name"; flush(stderr)
   m = mk(); nB = size(m.ps.WB, 1)
   # ASSEMBLE ONCE.  `acefit!` assembles inside the call, so sweeping lambda
   # through it rebuilds the same design matrix once per lambda -- four times
   # here, and assembly is the bulk of the wall time.  `ACEpotentials.assemble`
   # is exported precisely so the system can be built once and re-solved.
   # CACHE the assembly.  (A, Y, W) depend on the data and the basis only --
   # not on lambda, and not on the prior, which is applied afterwards as A / P.
   # So every subsequent sweep (wider lambda, different smoothness p, a
   # different prior entirely) is pure linear algebra on this file rather than a
   # ~10-minute reassembly.  Keyed on the model name and the data file.
   tag = replace(name, r"[^A-Za-z0-9]" => "_")
   # ORDER and DEGREE must be in the key: they change the basis, hence A.  Without
   # them a degree-8 run would silently reuse the degree-6 matrices -- a cache
   # that returns the wrong answer rather than a slow one.
   # the embedded design matrices depend on the embedding table; the
   # categorical one does not
   embtag = (startswith(tag, "embedded") && !isempty(EMBTAG)) ? "_" * EMBTAG : ""
   cache = joinpath(CACHEDIR,
                    "asm_$(tag)_o$(ORD)_d$(DEG)$(embtag)_$(basename(DATAFILE)).jls")
   if FASTBASIS && !isfile(cache)
      # self-check: fast path vs original on two training structures
      dd = ACEpotentials.make_atoms_data(tr[1:2], m; kw..., weights = ACEpotentials.default_weights())
      for d in dd
         Xf = ACEfit.feature_matrix(d, m)
         _FEATURE_ORIG[] = true
         Xo = ACEfit.feature_matrix(d, m)
         _FEATURE_ORIG[] = false
         err = maximum(abs, Xf - Xo) / max(maximum(abs, Xo), 1e-300)
         err < 1e-10 || error("FASTBASIS self-check failed: rel err $err")
      end
      @info "  FASTBASIS self-check passed on 2 structures"; flush(stderr)
   end
   if isfile(cache)
      @info "  reusing assembly from $(basename(cache))"; flush(stderr)
      A, Y, W = deserialize(cache)
   else
      A, Y, W = ACEpotentials.assemble(tr, m; kw...)
      # NOCACHE=1: with FASTBASIS the assembly is minutes, and a 4k-structure
      # degree-10 matrix is 150 GB on disk -- it exhausted the storage quota.
      get(ENV, "NOCACHE", "0") == "1" || (mkpath(CACHEDIR); serialize(cache, (A, Y, W)))
      @info "  cached assembly to $(basename(cache))"; flush(stderr)
   end
   # The REAL smoothness prior, as `acefit!` builds it -- not the identity.
   # `algebraic_smoothness_prior(model; p = smoothness)` penalises high-(n,l)
   # basis functions, and with P = I the fits were regularised only by the QR
   # lambda, which is not what a production fit does and disadvantages the
   # larger bases in particular.
   P = ACEpotentials.Models.algebraic_smoothness_prior(m.model; p = SMOOTH)
   # IN PLACE.  `Diagonal(W) * (A / P)` allocates TWO extra copies of A and keeps
   # the original alive -- on the degree-8 categorical arm that is 3 x 14.6 GB,
   # which drove a 62 GB node to 60 GB used and 16 GB of swap.  P is Diagonal and
   # W is a vector, so both transforms are just scalings and can be done in situ.
   A ./= reshape(P.diag, 1, :)      # column scaling: A / P
   A .*= W                          # row scaling:    Diagonal(W) * A
   Yw = W .* Y
   @info "  transformed in place, A is $(round(sizeof(A)/2^30, digits=1)) GiB"
   flush(stderr)
   @info "  assembled $(size(A))"; flush(stderr)
   # FACTOR ONCE.  lambda only touches the augmented rows, so qr(A) + svd(R) is
   # done once and every lambda after that is O(n^2) -- verified to match
   # ACEfit.QR to 1e-13..1e-15 (tikhonov.jl).  An 8-point sweep that was eight
   # O(m n^2) factorisations is now one.
   # BIGMEM=1: release the assembly workers (each holds a model plus compiled
   # code, ~7-10 GB) and factorise A in place, so the peak is A + R rather than
   # workers + 2A.  Only sensible for a single-model run (MODELS=...), since
   # the next model would have no workers to assemble with.
   if get(ENV, "BIGMEM", "0") == "1"
      rmprocs(workers()); GC.gc()
      @info "  BIGMEM: workers released, factorising in place"; flush(stderr)
      T = TikhonovFactor(A, Yw; inplace = true)
   else
      T = TikhonovFactor(A, Yw)
   end
   @info "  factorised; lambda sweep is now free"; flush(stderr)
   best = (Inf, Inf, 0.0)
   for lam in LAMS
      local r
      try
         ACEpotentials.Models.set_linear_parameters!(m, P \ tikhonov_solve(T, lam))
         r = ACEpotentials.compute_errors(te, m; kw..., verbose = false)["rmse"]["set"]
      catch err
         @warn "lambda=$lam failed: $(first(sprint(showerror, err), 80))"
         continue
      end
      r["F"] < best[1] && (best = (r["F"], r["E"], lam))
      @info @sprintf("    lambda=%-8g  F=%.4f  E=%.4f  V=%.4f", lam, r["F"], r["E"], r["V"])
      flush(stderr)
   end
   @printf("RES %-30s n_B=%5d  params=%6d  test F=%8.4f eV/A  test E=%8.4f  lambda=%g\n",
           name, nB, nB * length(ELS), best[1], best[2], best[3])
   flush(stdout)
end

# MODELS selects a subset, e.g. MODELS=categorical,d16 -- the categorical
# model at degree 10 on 4k structures is a 160 GB design matrix and does not
# fit any node we have, so the degree-10 sweep is embedded-only.
const MODELS = split(get(ENV, "MODELS", "categorical,lossless,d16,d8"), ",")
"categorical" in MODELS &&
   sweep("categorical ace1_model", () -> ace1_model(elements = collect(ELS),
             order = ORD, totaldegree = DEG))
"lossless" in MODELS &&
   sweep("embedded lossless", () -> M.ace_embedding_model(elements = ELS,
             order = ORD, totaldegree = DEG, embedding = emb))
for dm in (16, 8)
   "d$dm" in MODELS || continue
   sweep("embedded d_max=$dm", () -> M.ace_embedding_model(elements = ELS,
             order = ORD, totaldegree = DEG, embedding = emb, d_max = dm))
end
