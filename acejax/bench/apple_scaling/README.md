# apple_scaling — why the JAX MD path degraded with system size on the M3 Pro

Findings: [`docs/findings/FINDINGS_apple_scaling.md`](../../../docs/findings/FINDINGS_apple_scaling.md).

Short version: not padding (the padding fraction is a constant 0.491 at every
size), but the reverse-mode adjoint of `Rnl[:, aspec_r] * Ylm[:, aspec_y]` in
`acejax/model.py:156`, which lowers to an inner-axis scatter whose per-row cost
grows 4.6x with array length on this host. Fixing it flattens the scaling; a
tighter edge capacity on top gives 5.1x end-to-end at 1728 atoms.

| file | what it is |
|---|---|
| `ace_md_cap.py` | `spike_distmd/ace_md.py`'s `run()` with the two hardcoded capacities lifted to `--cap-owned` / `--cap-sub`, per-shape overrides (`--max-edges`, `--max-nb`, `--cell-cap`, `--max-owned`), a `--gather-fix` monkeypatch, and a machine-readable `RESULT` line. **Defaults reproduce the spike exactly**; `spike_distmd/` is untouched. |
| `sweep.py` | runs `ace_md_cap.py` once per point, fresh process each time, appending `RESULT` json |
| `analyse.py` | tables from the jsonl |
| `ace_stage_scan.py` | nested prefixes of `site_energies`, to find which stage scales badly |
| `gather_probe.py` | the axis-1 gather-product against an algebraically identical one-hot matmul |
| `gather_mechanism.py` | forward vs adjoint, duplicate vs unique indices, vs matmul |
| `micro_kernel.py` | a generic kernel of the same shape — the control that stays flat |
| `points/`, `logs/` | the point files as run, and the raw results behind every table |

`logs/A_BROKEN_override_noop.jsonl` is a retracted sweep whose capacity override
silently did nothing; it backs no table. See §8 of the findings.
