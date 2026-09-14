# Where does the per-structure time go in ACEfit.assemble once feature_matrix
# is fast?  Instrument the pieces on every process.  Run serially and with -p 4:
#    julia [-p 4] --project=acejax/julia 04b_overhead.jl
using Distributed
@everywhere begin
   include(joinpath(@__DIR__, "common.jl"))
   include(joinpath(@__DIR__, "basis_opt.jl"))
   using Unitful: ustrip
   const T_FM = Float64[]      # feature_matrix wall time per call, on this process
   function ACEfit.feature_matrix(d::AtomsData, calc::ACEPotential{<: ACEModel}; kwargs...)
      t0 = time()
      efv = efv_basis_v3(d.system, calc)
      dm = feature_matrix_from_efv(efv, length(d.system), length_basis(calc);
                                   has_E = !isnothing(d.energy_key),
                                   has_F = !isnothing(d.force_key),
                                   has_V = !isnothing(d.virial_key))
      push!(T_FM, time() - t0)
      return dm
   end
   # time GC.gc() calls on this process
   const T_GC = Float64[]
   const _gc_orig = GC.gc
   function timed_gc()
      t0 = time(); Base.GC.gc(); push!(T_GC, time() - t0)
   end
end
using ACEfit.SharedArrays, ACEfit.ParallelDataTransfer

structs = load_structs(32)
m = make_models((6,))[1].second       # cat_D6
model_info("cat_D6", m)
data = ACEpotentials.make_atoms_data(structs, m; KW..., weights = ACEpotentials.default_weights())
println("procs: $(nprocs())")

# ---- reference: ACEfit.assemble as is
ACEpotentials.assemble(structs[1:2], m; KW...)         # compile everywhere
for p in procs(); remotecall_fetch(() -> (empty!(T_FM); nothing), p); end
t_asm = @elapsed ACEfit.assemble(data, m)
tfm = vcat([remotecall_fetch(() -> copy(T_FM), p) for p in procs()]...)
@printf("ACEfit.assemble: %.2f s wall for 32; feature_matrix calls: n=%d, sum=%.2f s, mean=%.3f s, max=%.3f s\n",
        t_asm, length(tfm), sum(tfm), sum(tfm)/length(tfm), maximum(tfm))

# ---- a re-implementation of ACEfit.assemble with the pieces instrumented
function assemble_instrumented(data, basis; do_gc = true, capture = true)
   rows = Array{UnitRange}(undef, length(data))
   rows[1] = 1:ACEfit.count_observations(data[1])
   for i in 2:length(data)
      rows[i] = rows[i - 1][end] .+ (1:ACEfit.count_observations(data[i]))
   end
   packets = ACEfit.DataPacket.(rows, data)
   sort!(packets, by = length, rev = true)
   (nprocs() > 1) && sendto(workers(), basis = basis)
   A = SharedArray(zeros(rows[end][end], ACEfit.basis_size(basis)))
   Y = SharedArray(zeros(size(A, 1)))
   for p in procs(); remotecall_fetch(() -> (empty!(T_FM); empty!(T_GC); nothing), p); end
   if capture
      t = @elapsed pmap(packets) do p
         A[p.rows, :] .= ACEfit.feature_matrix(p.data, basis)
         Y[p.rows] .= ACEfit.target_vector(p.data)
         do_gc && timed_gc()
      end
   else
      # the model is looked up as a worker global, not captured by the closure
      t = @elapsed pmap(packets) do p
         b = Base.invokelatest(getglobal, Main, :basis)
         A[p.rows, :] .= ACEfit.feature_matrix(p.data, b)
         Y[p.rows] .= ACEfit.target_vector(p.data)
         do_gc && timed_gc()
      end
   end
   tfm = vcat([remotecall_fetch(() -> copy(T_FM), p) for p in procs()]...)
   tgc = vcat([remotecall_fetch(() -> copy(T_GC), p) for p in procs()]...)
   @printf("assemble_instrumented(gc=%s, capture=%s): pmap %.2f s wall; feature_matrix sum %.2f s (mean %.3f); GC.gc sum %.2f s (mean %.3f, n=%d)\n",
           do_gc, capture, t, sum(tfm), sum(tfm)/length(tfm),
           isempty(tgc) ? 0.0 : sum(tgc), isempty(tgc) ? 0.0 : sum(tgc)/length(tgc), length(tgc))
   return Array(A), Array(Y)
end

nprocs() == 1 && (global basis = m)
A0, _ = assemble_instrumented(data, m; do_gc = true, capture = true)
A0, _ = assemble_instrumented(data, m; do_gc = true, capture = true)
A1, _ = assemble_instrumented(data, m; do_gc = false, capture = true)
A2, _ = assemble_instrumented(data, m; do_gc = false, capture = false)
A3, _ = assemble_instrumented(data, m; do_gc = true, capture = false)
println("A equal across variants: ", A0 == A1 == A2 == A3)

# per-task pmap overhead with a trivial body, capturing the model or not
if nprocs() > 1
   t = @elapsed pmap(p -> length_basis(m), 1:32)
   @printf("pmap of 32 trivial tasks capturing the model: %.3f s (%.4f s/task)\n", t, t/32)
   t = @elapsed pmap(p -> p, 1:32)
   @printf("pmap of 32 trivial tasks, no capture:         %.3f s (%.4f s/task)\n", t, t/32)
end
# the AtomsData wrapper: constructing + target/weight vectors for 32 structures
t = @elapsed ACEpotentials.make_atoms_data(structs, m; KW..., weights = ACEpotentials.default_weights())
@printf("make_atoms_data (32): %.3f s\n", t)
t = @elapsed ACEfit.assemble_weights(data)
@printf("assemble_weights (32): %.3f s\n", t)
