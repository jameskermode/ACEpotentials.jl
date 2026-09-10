# NVE MD with ACEpotentials driven by Molly.jl, matched to
# acejax/spike_distmd/ace_md.py: same fitted model, same initial state
# (positions AND velocities, written by dump_initial_state.py), same dt, same
# neighbour-list rebuild cadence, no CM-motion removal (ace_md.py does not do it
# either), and the SAME WINDOW OF STEPS.
#
#   julia -t N --project=. md_molly.jl --model m.json --state s.npz \
#         --mode cached|naive|bare --inner 10 --skin 1.0 \
#         [--verify] [--steps 60 --reps 3]
#
# The step window matters: `Si_tiny` has no repulsive core, so even at
# dt = 0.25 fs the configuration runs downhill and heats (T goes 276 K -> 422 K
# over the first 60 steps), which makes later steps more expensive than earlier
# ones.  ace_md.py --outer 7 --inner 10 discards leg 0 and times legs 1-6, i.e.
# steps 10-70.  Every repeat here therefore RESETS to the initial state, runs
# `--inner` untimed steps, and times the next `--steps`.  Repeats measure the
# same 60 steps of the same trajectory, so min-over-repeats is meaningful.
#
# Three modes:
#
#   naive  -- the ACEPotential handed to Molly as `general_inters` directly.
#             AtomsCalculatorsUtilities' SitePotential assembly defaults
#             `nlist = PairList(sys, cutoff)`, so the neighbour list is REBUILT
#             ON EVERY FORCE CALL.  This is what "just plug it in" gives you and
#             the stack offers no supported way to avoid it.
#   cached -- the wrapper below: keeps a PairList across steps and rebuilds it
#             every `--inner` steps with `--skin` added to the cutoff, which is
#             the cadence ace_md.py runs.  This is the matched comparison.
#   bare   -- the identical velocity-Verlet loop written out by hand against the
#             same System and the same cached wrapper, with Molly's `simulate!`
#             removed.  cached - bare is Molly's own per-step overhead, measured
#             by difference over whole runs rather than by timing a stage.
#
# Why the wrapper is not two lines: NeighbourLists.PairList is NOT a Verlet
# list.  `get_neighbours` reads geometry out of the list itself
# (`_getR(nlist,idx) = X[j]-X[i] + C'S`), so a PairList reused verbatim freezes
# the configuration -- energies stop changing while the integrator keeps moving
# atoms (observed: PE constant to 6 d.p. across three legs).  The list can be
# reused only if `nlist.X` is refreshed every step, and it must be refreshed
# with positions CONTINUOUS with the ones the shifts `S` were built from,
# whereas Molly re-wraps `sys.coords` into the box every step.  Hence the
# minimum-image displacement against a stored reference below.

using ACEpotentials, Molly, AtomsBase, AtomsCalculators, Unitful, StaticArrays,
      NPZ, LinearAlgebra, Printf, Statistics
using AtomsCalculatorsUtilities.SitePotentials: PairList, cutoff_radius

# ------------------------------------------------------------------ args
getarg(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])
hasflag(f) = f in ARGS

const MODEL  = getarg("--model", "si_model.json")
const STATE  = getarg("--state", "state.npz")
const INNER  = parse(Int, getarg("--inner", "10"))
const DT_FS  = parse(Float64, getarg("--dt", "0.25"))
const MODE   = getarg("--mode", "cached")
const SKIN   = parse(Float64, getarg("--skin", "1.0"))
const STEPS  = parse(Int, getarg("--steps", "60"))
const REPS   = parse(Int, getarg("--reps", "3"))
const VERIFY = hasflag("--verify")
const VLEGS  = parse(Int, getarg("--verify-legs", "6"))
const TAG    = getarg("--tag", "-")
const RMCM   = parse(Int, getarg("--remove-cm", "0"))   # Molly default is 1
@assert MODE in ("cached", "naive", "bare")

# ------------------------------------------- neighbour-list caching wrapper
mutable struct CachedNLCalculator{C, T}
    calc::C
    every::Int
    skin::Float64
    cell::SMatrix{3, 3, T, 9}      # rows are lattice vectors
    nlist::Any                     # concrete PairList type unknown until built
    xref::Vector{SVector{3, T}}    # positions the shifts S were built from
    step::Int                      # our own counter: Molly's step_n restarts
    n_builds::Int
    n_safety::Int
    max_disp::Float64
end
CachedNLCalculator(calc, every, skin, cell::SMatrix{3,3,T,9}) where {T} =
    CachedNLCalculator{typeof(calc), T}(calc, every, skin, cell, nothing,
                                        SVector{3, T}[], 0, 0, 0, 0.0)

@inline _min_image(d::SVector{3,T}, L::SVector{3,T}) where {T} =
    SVector{3,T}(d[1] - L[1]*round(d[1]/L[1]),
                 d[2] - L[2]*round(d[2]/L[2]),
                 d[3] - L[3]*round(d[3]/L[3]))

# function barriers: `w.nlist` is ::Any, so every use goes through a typed call
function _refresh!(nl::PairList, xref::Vector{SVector{3,T}}, coords,
                   L::SVector{3,T}) where {T}
    dmax = zero(T); X = nl.X
    @inbounds for i in eachindex(coords)
        ci = coords[i]
        xi = SVector{3,T}(ustrip(u"Å", ci[1]), ustrip(u"Å", ci[2]), ustrip(u"Å", ci[3]))
        d = _min_image(xi - xref[i], L)
        X[i] = xref[i] + d
        dd = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
        dd > dmax && (dmax = dd)
    end
    return dmax
end
_ef(sys, calc, nl::PairList) = AtomsCalculators.energy_forces(sys, calc; nlist = nl)
_pe(sys, calc, nl::PairList) = AtomsCalculators.potential_energy(sys, calc; nlist = nl)

function _nlist!(w::CachedNLCalculator, sys)
    L = SVector(w.cell[1,1], w.cell[2,2], w.cell[3,3])
    rebuild = (w.nlist === nothing) || (w.step % w.every == 0)
    if !rebuild
        dmax = _refresh!(w.nlist, w.xref, sys.coords, L)
        w.max_disp = max(w.max_disp, dmax)
        if dmax > 0.5 * w.skin          # Verlet criterion violated
            rebuild = true; w.n_safety += 1
        end
    end
    if rebuild
        w.nlist = PairList(sys, cutoff_radius(w.calc) + w.skin * u"Å")
        w.xref = copy(w.nlist.X)
        w.n_builds += 1
    end
    w.step += 1
    return w.nlist
end

AtomsCalculators.energy_unit(w::CachedNLCalculator) = AtomsCalculators.energy_unit(w.calc)
AtomsCalculators.length_unit(w::CachedNLCalculator) = AtomsCalculators.length_unit(w.calc)

AtomsCalculators.forces(sys, w::CachedNLCalculator; kwargs...) =
    _ef(sys, w.calc, _nlist!(w, sys))[:forces]

# potential_energy must NOT advance the rebuild counter (reporting only)
function AtomsCalculators.potential_energy(sys, w::CachedNLCalculator; kwargs...)
    if w.nlist === nothing
        w.nlist = PairList(sys, cutoff_radius(w.calc) + w.skin * u"Å")
        w.xref = copy(w.nlist.X); w.n_builds += 1
    end
    return _pe(sys, w.calc, w.nlist)
end

AtomsCalculators.virial(sys, w::CachedNLCalculator; kwargs...) =
    AtomsCalculators.virial(sys, w.calc; kwargs...)
Molly.virial(w::CachedNLCalculator, sys, neighbors, step_n; kwargs...) =
    AtomsCalculators.virial(sys, w; kwargs...)

# ------------------------------------------------------------------ setup
model, _ = ACEpotentials.load_model(MODEL)
S     = npzread(STATE)
pos   = S["positions"]; cellm = S["cell"]
vel   = S["velocities_A_per_fs"]; massv = S["mass_amu"]
n_atoms = size(pos, 1)
@assert isapprox(cellm, Diagonal(diag(cellm)); atol = 1e-10) "non-cubic cell"
L = cellm[1,1]
@assert isapprox(cellm[2,2], L; atol=1e-9) && isapprox(cellm[3,3], L; atol=1e-9)

const CELL   = SMatrix{3,3,Float64}(cellm')
const MASS   = massv * u"u"
const COORDS = [SVector{3,Float64}(pos[i,1], pos[i,2], pos[i,3]) * u"Å" for i in 1:n_atoms]
const VELS   = [SVector{3,Float64}(vel[i,1], vel[i,2], vel[i,3]) * u"Å/fs" for i in 1:n_atoms]
const MATOMS = [Molly.Atom(index=i, mass=MASS, charge=0.0, σ=0.0u"Å", ϵ=0.0u"eV") for i in 1:n_atoms]
const ADATA  = [Molly.AtomData(element="Si") for i in 1:n_atoms]

function fresh()
    calc = MODE == "naive" ? model : CachedNLCalculator(model, INNER, SKIN, CELL)
    sys = Molly.System(atoms = MATOMS, atoms_data = ADATA,
                       coords = copy(COORDS), velocities = copy(VELS),
                       boundary = Molly.CubicBoundary(L * u"Å"),
                       general_inters = (calc,),
                       neighbor_finder = Molly.NoNeighborFinder(),
                       energy_units = u"eV", force_units = u"eV/Å")
    return sys, calc
end

# remove_CM_motion=0: ace_md.py zeroes the CM velocity once at setup and never
# again, so removing it every step would be a different integrator.
const SIM  = Molly.VelocityVerlet(dt = DT_FS * u"fs", remove_CM_motion = RMCM)
const NTHR = Threads.nthreads()
# The site loop is already threaded (`Folds` + `ThreadedEx`), so a threaded BLAS
# underneath it only oversubscribes.  Pinned to 1 here so that "1 thread" means
# one thread; on the Linux host it was already 1 via MKL_NUM_THREADS.
LinearAlgebra.BLAS.set_num_threads(1)
const DT   = DT_FS * u"fs"

# hand-rolled velocity Verlet, algebraically identical to Molly's, used to
# measure Molly's own overhead by difference
function bare_run!(sys, calc, nsteps)
    a = AtomsCalculators.forces(sys, calc) ./ MASS
    for _ in 1:nsteps
        sys.coords .+= sys.velocities .* DT .+ a .* (DT^2 / 2)
        sys.coords .= Molly.wrap_coords.(sys.coords, (sys.boundary,))
        anew = AtomsCalculators.forces(sys, calc) ./ MASS
        sys.velocities .+= (a .+ anew) .* (DT / 2)
        a = anew
    end
    return sys
end

run!(sys, calc, n) = MODE == "bare" ? bare_run!(sys, calc, n) :
    Molly.simulate!(sys, SIM, n; n_threads = NTHR, run_loggers = false)

pe(sys) = ustrip(u"eV", Molly.potential_energy(sys, nothing, 0; n_threads = NTHR))
ke(sys) = ustrip(u"eV", Molly.kinetic_energy(sys))

println("# molly-ace  mode=$MODE  atoms=$n_atoms  julia_threads=$NTHR  ",
        "blas_threads=$(LinearAlgebra.BLAS.get_num_threads())  dt=$(DT_FS)fs  ",
        "inner=$INNER  skin=$SKIN  remove_CM=$RMCM  L=$(round(L,digits=4))  cutoff=$(cutoff_radius(model))  tag=$TAG")

if VERIFY
    sys, calc = fresh()
    e0 = Ref(NaN)
    println(@sprintf("  initial PE=%.6f  KE=%.4f", pe(sys), ke(sys)))
    for leg in 1:VLEGS
        run!(sys, calc, INNER)
        local E, K
        E, K = pe(sys), ke(sys); tot = E + K
        isnan(e0[]) && (e0[] = tot)
        println(@sprintf("  leg %2d  PE=%.6f  KE=%.4f  T=%6.1f K  Etot=%.6f  drift=%+.3e eV",
                leg, E, K, 2K/(3*n_atoms*8.617333262e-5), tot, tot - e0[]))
    end
    ps = VLEGS * INNER * DT_FS / 1000
    E, K = pe(sys), ke(sys)
    println(@sprintf("VERIFY %s atoms=%d final_PE=%.9f final_KE=%.9f drift_meV_atom_ps=%+.4f",
            TAG, n_atoms, E, K, (E + K - e0[])/n_atoms/ps*1e3))
    calc isa CachedNLCalculator && println("  nlist builds: $(calc.n_builds) ($(calc.n_safety) forced by skin), max disp $(round(calc.max_disp,digits=4)) Å")
else
    # Warm-up: one FULL repeat on a throwaway system, discarded.  It has to be
    # full-length, not a token dozen steps: with -t 16 the first repeat is still
    # 20% slower than the third (thread pool + JIT transients), so a short
    # warm-up leaves the transient inside the measurement.
    tw = time_ns(); let (s, c) = fresh(); run!(s, c, INNER); run!(s, c, STEPS); end
    println(@sprintf("# warmup (a full %d+%d-step repeat on a throwaway system, discarded): %.3f s",
            INNER, STEPS, (time_ns() - tw)/1e9))

    times = Float64[]
    finalE = Ref(0.0); finalK = Ref(0.0); nb = Ref(0); ns = Ref(0)
    for r in 1:REPS
        sys, calc = fresh()
        run!(sys, calc, INNER)                    # untimed: matches ace_md leg 0
        t = time_ns()
        run!(sys, calc, STEPS)                    # timed: steps INNER..INNER+STEPS
        dtr = (time_ns() - t)/1e9
        push!(times, dtr)
        finalE[], finalK[] = pe(sys), ke(sys)
        if calc isa CachedNLCalculator; nb[] = calc.n_builds; ns[] = calc.n_safety; end
        println(@sprintf("  repeat %d: %.4f s  (%.4f ms/step)  final PE=%.9f  final KE=%.9f",
                r, dtr, dtr/STEPS*1e3, finalE[], finalK[]))
    end
    tmin, tmax = minimum(times), maximum(times)
    ps = tmin/STEPS
    println(@sprintf("  step  min %.4f ms   spread (max-min)/min = %.2f%%", ps*1e3,
            100*(tmax-tmin)/tmin))
    nb[] > 0 && println("  nlist builds/repeat: $(nb[]) ($(ns[]) forced by skin)")
    println("RESULT tag=", TAG, " mode=", MODE, " atoms=", n_atoms,
            " julia_threads=", NTHR, " steps=", STEPS, " reps=", REPS,
            " ms_per_step_min=", ps*1e3, " atom_steps_per_s=", n_atoms/ps,
            " spread_pct=", 100*(tmax-tmin)/tmin,
            " final_PE=", finalE[], " final_KE=", finalK[])
end
