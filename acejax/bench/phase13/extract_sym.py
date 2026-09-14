import sys, json, pathlib, importlib.util
import matplotlib; matplotlib.use("Agg")
P = "/storage/eng/essswb/phase13/symmetrix/symmetrix/source/symmetrix/extract_mace_data.py"
spec = importlib.util.spec_from_file_location("emd", P)
emd = importlib.util.module_from_spec(spec); spec.loader.exec_module(emd)
ckpt, out = sys.argv[1], sys.argv[2]
try:
    data = emd.extract_mace_data(ckpt, [14])   # Si only
except Exception as e:
    print("REJECTED", type(e).__name__, e); raise SystemExit(3)
pathlib.Path(out).write_text(json.dumps(data, indent=1))
print("WROTE", out, pathlib.Path(out).stat().st_size, "bytes")
