# Minimal reproducer: Reactant 0.2.222 (CPU) deduplicates two distinct
# elementwise products into one.
#
#   julia --project=. reactant_bug_repro.jl
#
# Expected: column 1 = z.*z, column 2 = y.*z
# Actual (compiled): both columns = z.*z
#
# Eager Julia is correct; only the compiled program is wrong, and it fails
# silently. Found via ACEpotentials -> lammps-jax dev/julia_export ACE export,
# where it corrupts the l=2 spherical harmonics.

using Reactant
Reactant.set_default_backend("cpu")

f(u) = hcat(u[:, 3] .* u[:, 3], u[:, 2] .* u[:, 3])

u  = [1.0 2.0 3.0; 4.0 5.0 6.0]
ru = Reactant.to_rarray(u)

eager    = f(u)
compiled = Array((@compile f(ru))(ru))

println("eager:    ", eager)      # [9.0 6.0; 36.0 30.0]
println("compiled: ", compiled)   # [9.0 9.0; 36.0 36.0]   <-- wrong
println(maximum(abs.(compiled .- eager)) < 1e-12 ? "OK" : "MISMATCH")
