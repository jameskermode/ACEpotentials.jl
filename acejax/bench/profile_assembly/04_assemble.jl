# (d) ACEpotentials.assemble on 32 structures, serial and with workers, with the
# original feature_matrix and with the pushforward path (scratch override of
# ACEfit.feature_matrix on every process).  Run as
#    julia -p 0|4 --project=acejax/julia 04_assemble.jl
# with env MODELS=cat_D6,emb16_D8  ORIG=1  OPT=1  NSTRUCT=32
using Distributed
@everywhere begin
   include(joinpath(@__DIR__, "common.jl"))
   include(joinpath(@__DIR__, "basis_opt.jl"))
   using Unitful: ustrip
   # switchable override: FEATURE_MODE = :orig | :pushfwd | :pushfwd_threads
   const FEATURE_MODE = Ref(:orig)
   const WS = Dict{UInt, Any}()
   const _ORIG_FEATURE_MATRIX = which(ACEfit.feature_matrix, (AtomsData, Any))
   function ACEfit.feature_matrix(d::AtomsData, calc::ACEPotential{<: ACEModel}; kwargs...)
      if FEATURE_MODE[] == :orig
         return invoke(ACEfit.feature_matrix, Tuple{AtomsData, Any}, d, calc; kwargs...)
      end
      if FEATURE_MODE[] == :pushfwd
         efv = efv_basis_v3(d.system, calc)
      elseif FEATURE_MODE[] == :pushfwd_ws          # v4: workspace reuse, no per-site allocation
         ws = get!(WS, objectid(calc)) do; PFWorkspace(calc.model, 100); end
         efv = efv_basis_v4(d.system, calc; ws = ws)
      else
         efv = efv_basis_v3t(d.system, calc)
      end
      return feature_matrix_from_efv(efv, length(d.system), length_basis(calc);
                                     has_E = !isnothing(d.energy_key),
                                     has_F = !isnothing(d.force_key),
                                     has_V = !isnothing(d.virial_key))
   end
end
using Serialization

nstruct = parse(Int, get(ENV, "NSTRUCT", "32"))
structs = load_structs(nstruct)
which_models = get(ENV, "MODELS", "cat_D6,emb16_D8")
models = [p for p in make_models() if p.first in split(which_models, ",")]
nrows = sum(1 + 3length(s) + 6 for s in structs)
println("procs: $(nprocs())  workers: $(nworkers())  threads/proc: $(Threads.nthreads())  structures: $nstruct  rows: $nrows")

function set_mode!(mode)
   for p in procs()
      remotecall_fetch(() -> (FEATURE_MODE[] = mode; nothing), p)
   end
end

for (name, m) in models
   model_info(name, m)
   # how big is the model when serialised (this is what pmap ships per task if
   # the closure captures it)
   io = IOBuffer(); serialize(io, m); nbytes = io.size
   tser = @elapsed (io2 = IOBuffer(); serialize(io2, m))
   @printf("   serialised model: %.1f MB, serialize %.3f s\n", nbytes/2^20, tser)
   if nprocs() > 1
      w = first(workers())
      # closure capturing the model, sent per call
      remotecall_fetch(() -> length_basis(m), w)
      t1 = @elapsed for _ in 1:5; remotecall_fetch(() -> length_basis(m), w); end
      # global on the worker, no capture
      ACEfit.sendto(workers(), _basis_global = m)
      remotecall_fetch(() -> length_basis(Base.invokelatest(getglobal, Main, :_basis_global)), w)
      t2 = @elapsed for _ in 1:5; remotecall_fetch(() -> length_basis(Base.invokelatest(getglobal, Main, :_basis_global)), w); end
      @printf("   remotecall with captured model: %.3f s/call; with worker-global model: %.4f s/call\n", t1/5, t2/5)
   end
   # cost of the GC.gc() ACEfit does after every structure
   tgc = @elapsed for _ in 1:5; GC.gc(); end
   @printf("   GC.gc() (full) on master: %.3f s per call\n", tgc/5)

   modes = Symbol[]
   get(ENV, "ORIG", "1") == "1" && push!(modes, :orig)
   get(ENV, "OPT", "1") == "1" && push!(modes, :pushfwd)
   get(ENV, "OPT", "1") == "1" && push!(modes, :pushfwd_ws)
   get(ENV, "OPT", "1") == "1" && Threads.nthreads() > 1 && push!(modes, :pushfwd_threads)
   results = Dict{Symbol, Any}()
   for mode in modes
      set_mode!(mode)
      # warm-up on 1 structure (compilation on all procs)
      ACEpotentials.assemble(structs[1:min(2, end)], m; KW...)
      GC.gc()
      t = @elapsed A, Y, W = ACEpotentials.assemble(structs, m; KW...)
      results[mode] = (A, Y, W)
      @printf("   assemble[%-16s] %d structs, %d procs: %8.2f s wall = %6.3f s/structure (x nworkers = %6.2f worker-s/structure)  size %s\n",
              mode, nstruct, nprocs(), t, t/nstruct, t*max(1, nworkers())/nstruct, size(A))
      flush(stdout)
   end
   if haskey(results, :orig) && haskey(results, :pushfwd)
      A0, Y0, W0 = results[:orig]; A1, Y1, W1 = results[:pushfwd]
      @printf("   max |A_orig - A_pushfwd| = %.2e  (max|A| = %.2e);  Y equal: %s;  W equal: %s\n",
              maximum(abs.(A0 .- A1)), maximum(abs.(A0)), Y0 == Y1, W0 == W1)
   end
   if haskey(results, :pushfwd) && haskey(results, :pushfwd_ws)
      A1 = results[:pushfwd][1]; A2 = results[:pushfwd_ws][1]
      @printf("   max |A_pushfwd - A_pushfwd_ws| = %.2e\n", maximum(abs.(A1 .- A2)))
   end
   if haskey(results, :pushfwd) && haskey(results, :pushfwd_threads)
      A1 = results[:pushfwd][1]; A2 = results[:pushfwd_threads][1]
      @printf("   max |A_pushfwd - A_pushfwd_threads| = %.2e\n", maximum(abs.(A1 .- A2)))
   end
end
