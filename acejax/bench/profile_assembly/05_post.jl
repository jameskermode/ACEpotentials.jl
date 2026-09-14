# (e) post-assembly stages of fit_distilled.jl, timed:
#   (i)   serialize (A, Y, W) to disk        [synthetic 20k x 4k; the Mac was
#         swapping with anything larger while another job ran]
#   (ii)  in-place prior / weight scaling      [synthetic]
#   (iii) TikhonovFactor (qr + svd)            [synthetic 20k x 4k and 10k x 4k]
#   (iv)  compute_errors on 100 structures per model vs residual-from-design-matrix
#         + the 1e-10 agreement check on 32 structures.
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "basis_opt.jl"))
include(joinpath(@__DIR__, "residuals.jl"))
include(joinpath(@__DIR__, "..", "distil", "tikhonov.jl"))
using Serialization, Random
using Unitful: ustrip
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "8")))
println("BLAS threads: $(BLAS.get_num_threads())")

# use the fast basis path for assembling here (identical to 1e-13, see 04_assemble)
const WS = Dict{UInt, Any}()      # one workspace per model object
function ACEfit.feature_matrix(d::AtomsData, calc::ACEPotential{<: ACEModel}; kwargs...)
   ws = get!(WS, objectid(calc)) do; PFWorkspace(calc.model, 100); end
   efv = efv_basis_v4(d.system, calc; ws = ws)
   return feature_matrix_from_efv(efv, length(d.system), length_basis(calc);
                                  has_E = !isnothing(d.energy_key),
                                  has_F = !isnothing(d.force_key),
                                  has_V = !isnothing(d.virial_key))
end

# --------------------------------------------------------------------------
# synthetic stages
m_syn = parse(Int, get(ENV, "MSYN", "20000")); n_syn = parse(Int, get(ENV, "NSYN", "4000"))
println("\n=== synthetic (A, Y, W) of size $m_syn x $n_syn = $(round(8m_syn*n_syn/2^30, digits=2)) GiB ===")
A = randn(m_syn, n_syn); Y = randn(m_syn); W = rand(m_syn) .+ 0.5
cache = joinpath(SP, "asm_synthetic.jls")
t = @elapsed serialize(cache, (A, Y, W))
@printf("(i)   serialize to disk:            %7.2f s   (%.2f GB/s)\n", t, 8m_syn*n_syn/1e9/t)
t = @elapsed (A2, Y2, W2) = deserialize(cache)
@printf("      deserialize from disk:        %7.2f s\n", t); A2 = nothing; GC.gc()
rm(cache)
Pdiag = rand(n_syn) .+ 0.5
t = @elapsed begin
   A ./= reshape(Pdiag, 1, :)
   A .*= W
end
@printf("(ii)  in-place scaling A/P, W*A:    %7.2f s\n", t)
Yw = W .* Y
for msub in (m_syn ÷ 2, m_syn)
   local As = view(A, 1:msub, :); local Ys = Yw[1:msub]
   GC.gc()
   local t = @elapsed T = TikhonovFactor(Matrix(As), Ys)
   local tq = @elapsed qr(Matrix(As))
   @printf("(iii) TikhonovFactor(%6d x %d):  %7.2f s  (qr alone %.2f s; %.1f GFLOP/s on qr)\n",
           msub, n_syn, t, tq, 2 * msub * n_syn^2 / 1e9 / tq)
   local ts = @elapsed tikhonov_solve(T, 1e-3)
   @printf("      one tikhonov_solve:           %7.4f s\n", ts)
end
A = nothing; GC.gc()

# --------------------------------------------------------------------------
# (iv) compute_errors vs residual-from-design, per model
structs = load_structs(100)
models = make_models()
for (name, m) in models
   randomise_linear!(m)
   model_info(name, m)
   # 32-structure agreement check
   s32 = structs[1:32]
   A32, Y32, W32 = ACEpotentials.assemble(s32, m; KW...)
   L32 = RowLayout(s32; KW...)
   c = M.get_basis_params(m.model, m.ps)
   r_ce = ACEpotentials.compute_errors(s32, m; KW..., verbose = false)["rmse"]["set"]
   r_dm = rmse_from_design(A32, Y32, c, L32)
   for k in ("E", "F", "V")
      @printf("   32 structs  %s: compute_errors %.12e  design %.12e  |diff| %.2e  rel %.2e\n",
              k, r_ce[k], r_dm[k], abs(r_ce[k] - r_dm[k]), abs(r_ce[k] - r_dm[k]) / r_ce[k])
   end
   # timing on 100 structures
   ACEpotentials.compute_errors(structs[1:2], m; KW..., verbose = false)
   t_ce = @elapsed ACEpotentials.compute_errors(structs, m; KW..., verbose = false)
   t_asm = @elapsed (A100, Y100, W100) = ACEpotentials.assemble(structs, m; KW...)
   L100 = RowLayout(structs; KW...)
   rmse_from_design(A100, Y100, c, L100)
   t_res = @elapsed rmse_from_design(A100, Y100, c, L100)
   @printf("   100 structs: compute_errors %.2f s (%d threads) | assemble-once (pushfwd) %.2f s | residual per lambda %.4f s\n",
           t_ce, Threads.nthreads(), t_asm, t_res)
   @printf("   extrapolated 19 lambdas x 200 test structs: compute_errors %.0f s  vs  assemble 200 once %.0f s + 19 residuals %.1f s\n",
           19 * 2 * t_ce, 2 * t_asm, 19 * 2 * t_res)
   @printf("   extrapolated 19 lambdas x 1000 structs:     compute_errors %.0f s  vs  assemble 1000 once %.0f s + 19 residuals %.1f s\n",
           19 * 10 * t_ce, 10 * t_asm, 19 * 10 * t_res)
   flush(stdout)
end
