# Many-element basis scaling: categorical species encoding vs a frozen element
# embedding at capped channel width.  NO FITTING -- throughput and basis size
# depend on the basis, not on the weights, and a fit at these element counts
# would need data that does not exist here.
#
#   EMB=<embedding.json> julia --project=acejax/julia acejax/bench/manyelem/basis_scaling.jl
#
# The element pool is taken from the tabulated bond lengths rather than hardcoded:
# ACEpotentials can only build a model for elements in
# data/length_scales_VASP_auto_length_scales.yaml (76 of them), and picking any
# other gives `UndefVarError: rnn` -- which looks like a scaling failure and is
# not one.
using ACEpotentials, Printf, AtomsBase
M = ACEpotentials.Models

e = M.read_mace_embedding(ENV["EMB"])
ORD  = parse(Int, get(ENV, "ORD", "3"))
DEG  = parse(Int, get(ENV, "DEG", "6"))
DMAX = parse(Int, get(ENV, "DMAX", "16"))

zs = sort(collect(keys(ACEpotentials.DefaultHypers._lengthscales)))
zs = [z for z in zs if z in e.Z]                    # and present in the table
POOL = [Symbol(ChemicalSpecies(z)) for z in zs]
@printf("pool: %d elements with both a bond length and an embedding row\n", length(POOL))
@printf("order=%d degree=%d d_max=%d\n\n", ORD, DEG, DMAX)

@printf("%-5s %-14s %10s %9s %s\n", "S", "kind", "n_B", "build_s", "status")
# The categorical basis grows as S^order, so beyond ~20 elements it stops being
# a benchmark and becomes an OOM: n_B is already 70_755 at S=20 and the build
# takes 16 s.  SMAX_CAT bounds it so a sweep cannot take the machine down -- the
# point is the growth rate, which is established well before the wall.
SMAX_CAT = parse(Int, get(ENV, "SMAX_CAT", "20"))
for S in (2, 5, 10, 20, 30, 40, 60, length(POOL))
   S <= length(POOL) || continue
   els = POOL[1:S]
   if S > SMAX_CAT
      @printf("%-5d %-14s %10s %9s %s\n", S, "categorical", "-", "-",
              "skipped: n_B ~ S^$ORD, see SMAX_CAT")
   else
   t = @elapsed begin
      ok = true
      local mc
      try
         mc = ace1_model(elements = els, order = ORD, totaldegree = DEG)
      catch err
         ok = false
         global lasterr = first(sprint(showerror, err), 120)
      end
   end
   @printf("%-5d %-14s %10s %9.1f %s\n", S, "categorical",
           ok ? string(size(mc.ps.WB, 1)) : "-", t, ok ? "ok" : "FAILED: $lasterr")
   end
   flush(stdout)

   t2 = @elapsed begin
      ok2 = true
      local me
      try
         me = M.ace_embedding_model(elements = tuple(els...), order = ORD,
                  totaldegree = DEG, embedding = e, d_max = DMAX)
      catch err
         ok2 = false
         global lasterr2 = first(sprint(showerror, err), 120)
      end
   end
   @printf("%-5d %-14s %10s %9.1f %s\n", S, "embed d<=$DMAX",
           ok2 ? string(size(me.ps.WB, 1)) : "-", t2,
           ok2 ? "widths=$(M.embedding_widths(S, ORD; d_max = DMAX))" : "FAILED: $lasterr2")
   flush(stdout)
end
