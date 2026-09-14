# where does the scratch evaluator still allocate?
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fast_efv.jl"))
using Profile
frames = load_frames(1); sys = frames[1]
models = make_models((6,); si = false)
m = models[1].second; fe = FastEFV(m)
println("cat_D6: LEN=$(fe.rad.LEN) NU=$(fe.rad.NU)  pair LEN=$(fe.pair.LEN) NU=$(fe.pair.NU)")
m2 = models[2].second; fe2 = FastEFV(m2)
println("emb16_D6: LEN=$(fe2.rad.LEN) NU=$(fe2.rad.NU)  pair LEN=$(fe2.pair.LEN) NU=$(fe2.pair.NU)")
res, wss = fast_efv(sys, fe; ntasks = 1)
nlist = PairList(sys, fe.rcut * u"Å")
izs = [ findfirst(==(atomic_number(sys, i)), fe.i2z)::Int for i = 1:length(sys) ]
ws = wss[1]
nX = _gather!(ws, fe, nlist, izs, 1)
site_ed!(fe, ws, nX, izs[1])
st = @timed site_ed!(fe, ws, nX, izs[1])
println("site_ed!: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs, nX = $nX")
rs = view(ws.rs, 1:nX); jzs = view(ws.jzs, 1:nX)
macro timed2(ex)   # run once to compile, then time
   quote
      $(esc(ex)); @timed $(esc(ex))
   end
end
st = @timed2 radial_ed!(view(ws.Rnl,1:nX,:), view(ws.dRnl,1:nX,:), fe.rad, rs, 1, jzs, ws.es, ws.Pv, ws.Pg)
println("radial_ed!: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 _edge_spl(fe.rad, rs[1], 1, jzs[1])
println("_edge_spl: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 P4ML.evaluate_ed!(view(ws.Ylm,1:nX,:), view(ws.dYlm,1:nX,:), fe.model.ybasis, view(ws.Rs,1:nX))
println("Ylm: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 ET.evaluate!(ws.A, fe.model.tensor.abasis, (view(ws.Rnl,1:nX,:), view(ws.Ylm,1:nX,:)))
println("A: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 ET.evaluate!(ws.AA, fe.model.tensor.aabasis, ws.A)
println("AA: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 ET.pullback!(ws.∂A, fe.wAA[1], fe.model.tensor.aabasis, ws.A)
println("∂A: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 ET.pullback!((view(ws.∂Rnl,1:nX,:), view(ws.∂Ylm,1:nX,:)), ws.∂A, fe.model.tensor.abasis, (view(ws.Rnl,1:nX,:), view(ws.Ylm,1:nX,:)))
println("∂Rnl,∂Ylm: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed2 _gather!(ws, fe, nlist, izs, 1)
println("_gather!: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
F = zeros(SVector{3,Float64}, length(sys))
_chunk!(fe, ws, nlist, izs, F, 1:length(sys))
st = @timed _chunk!(fe, ws, nlist, izs, F, 1:length(sys))
println("_chunk! (32 sites): $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")
st = @timed fast_efv(sys, fe; ntasks = 1, wss = wss, nlist = nlist)
println("fast_efv: $(st.bytes) bytes, $(Base.gc_alloc_count(st.gcstats)) allocs")

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1.0 _chunk!(fe, ws, nlist, izs, F, 1:length(sys))
res = Profile.Allocs.fetch()
agg = Dict{String, Tuple{Int, Int}}()
for a in res.allocs
   fr = nothing
   for f in a.stacktrace
      s = string(f.file)
      if !occursin("julia/base", s) && !occursin("@Base", s) && !startswith(s, "./")
         fr = "$(basename(s)):$(f.line) $(f.func)"; break
      end
   end
   fr === nothing && (fr = "<other>")
   key = "$(first(string(a.type), 50)) @ $fr"
   c, b = get(agg, key, (0, 0))
   agg[key] = (c + 1, b + a.size)
end
for (k, (c, b)) in sort(collect(agg), by = x -> -x[2][1])[1:min(end,15)]
   @printf("  %6d allocs %9.3f MB  %s\n", c, b / 1e6, first(k, 140))
end
