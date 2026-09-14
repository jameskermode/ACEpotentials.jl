# Tests 1 and 2: per-order channel widths, and the lossless boundary.
#
#   d_nu = min(d_max, C(S+nu-1, nu))
#
# Each correlation order saturates at its own species-tensor dimension, so a
# single width either starves the high orders or pads the low ones.  Channel k of
# radial n maps to n' = (n-1)*d_max + k throughout, so the radial index space is
# shared and only the per-order channel RANGE differs.
using ACEpotentials, Printf
const M = ACEpotentials.Models
# ET is an indirect dep of the acejax/julia manifest; load it without adding it
# to that project, whose Manifest is committed and drives the CI fixtures.
const ET = Base.require(Base.PkgId(
      Base.UUID("5e107534-7145-4f8f-b06f-47a52840c895"), "EquivariantTensors"))
const WL = 1.5
ace1_level(w) = M.TotalDegree(1.0*w, 1/WL)
dim_sym(S, nu) = binomial(S + nu - 1, nu)

function build_from_mb(mb, rsp)
   maxl = maximum(maximum(b.l for b in bb) for bb in mb)
   tensor = ET.sparse_equivariant_tensor(L=0, mb_spec=mb, Rnl_spec=rsp,
                             Ylm_spec=M._make_Y_spec(maxl), basis=real)
   nB = size(tensor.A2Bmaps[1], 1)
   # per-order counts of the mb_spec that generated them
   byord = Dict{Int,Int}()
   for bb in mb; byord[length(bb)] = get(byord, length(bb), 0) + 1; end
   nB, byord
end

function categorical(S, order, deg)
   lvl = ace1_level(S); r_spec = M.oneparticle_spec(lvl, deg)
   AA = M.sparse_AA_spec(; order=order, r_spec=r_spec, level=lvl, max_level=deg)
   mb = unique([[(n=b.n,l=b.l) for b in bb] for bb in AA])
   build_from_mb(mb, r_spec)[1]
end

"""widths(nu) gives the channel count for correlation order nu."""
function diagonal(widths, order, deg)
   lvl1 = ace1_level(1); r1 = M.oneparticle_spec(lvl1, deg)
   dmax = maximum(widths(nu) for nu = 1:order)
   AA1 = M.sparse_AA_spec(; order=order, r_spec=r1, level=lvl1, max_level=deg)
   mb1 = unique([[(n=b.n,l=b.l) for b in bb] for bb in AA1])
   mb = [ [(n=(b.n-1)*dmax+k, l=b.l) for b in bb]
          for bb in mb1 for k = 1:widths(length(bb)) ]
   mb = sort(mb, by=length)
   rsp = sort([(n=(b.n-1)*dmax+k, l=b.l) for b in r1 for k=1:dmax], by=x->(x.l,x.n))
   build_from_mb(mb, rsp)
end

const DEG = 8
categorical(2,3,DEG); diagonal(nu->2, 3, DEG)          # warm up

@printf("%-3s %-3s %10s %10s %10s %9s %9s %9s\n",
        "nu","S","n_B cat","uniform d","per-order","d_uni","saving","vs uniform")
for nu in (2,3,4), S in (3,5,10)
   (nu == 4 && S == 10) && continue                     # build cost
   nBcat = categorical(S, nu, DEG)
   dmax  = dim_sym(S, nu)                               # lossless width for the TOP order
   nBu, _   = diagonal(_->dmax, nu, DEG)                # same width everywhere
   nBp, byo = diagonal(v->min(dmax, dim_sym(S, v)), nu, DEG)
   @printf("%-3d %-3d %10d %10d %10d %9d %8.2fx %9.2fx\n",
           nu, S, nBcat, nBu, nBp, dmax, nBcat/nBp, nBu/nBp)
end
