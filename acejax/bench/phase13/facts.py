import sys, json, functools, torch
torch.load = functools.partial(torch.load, weights_only=False)
from e3nn import o3
for path in sys.argv[1:]:
    m = torch.load(path, map_location="cpu", weights_only=False)
    irr = o3.Irreps(str(m.products[0].linear.irreps_out))
    width = next((mul for mul, ir in irr if ir.l == 0), None)
    print(json.dumps({
      "model": path.split("/")[-1],
      "parameters": sum(p.numel() for p in m.parameters()),
      "r_max": float(m.r_max), "layers": len(m.interactions),
      "hidden_irreps": str(irr), "channel_width": width,
      "max_ell": int(m.spherical_harmonics._lmax),
      "species": len(m.atomic_numbers),
      "first_block": type(m.interactions[0]).__name__,
      "second_block": type(m.interactions[1]).__name__}))
