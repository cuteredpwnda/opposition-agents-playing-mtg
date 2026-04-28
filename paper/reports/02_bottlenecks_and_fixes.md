# Report 02 — Bottlenecks Found and Fixed

**Date:** 2026-04-28
**Author:** training-pipeline triage session

The first end-to-end attempt of the JEPA pipeline appeared to hang for many
minutes despite a working CUDA install. Root-causing it surfaced three
independent bugs, all now fixed.

## Bug 1 — Trajectory load was O(N × K)

**File:** [src/world_model/trajectory.py](../../src/world_model/trajectory.py)

The original `TrajectoryStore.save()` wrote a separate `npz` key per
`(transition_index, feature_name)` pair:

```text
state_0_player_features, state_0_hand_cards, state_0_hand_mask, …
state_1_player_features, …
…
state_3885_player_features, …
```

A single 4 000-transition trajectory ended up with **70 000+ npz keys**.
`load()` then did

```python
while f"action_{i}" in data:
    for key in data:                  # ← O(K) scan
        if key.startswith(f"state_{i}_"): …
    i += 1
```

making the loader **O(N × K)** ≈ 280 M dict lookups for a single game.
Measured: **234 s to load one trajectory**.

### Fix

New dense format (`_format = "dense_v1"`): per-feature stacked arrays
(~14 keys total per file) plus stacked `actions`, `rewards`, `dones`:

```python
arrays[f"state__{key}"]   = np.stack([t.state_features[key] for t in traj])
arrays["actions"]         = np.stack([t.action_encoding for t in traj])
arrays["rewards"]         = np.array([t.reward for t in traj], dtype=np.float32)
arrays["dones"]           = np.array([t.done   for t in traj], dtype=bool)
```

`load()` detects `_format` and returns in **~0.05 s/game** (≈4 600× speed-up).
The legacy reader is retained behind the `_format` check so old files still
work.

## Bug 2 — Encoder shape mismatch

**File:** [src/world_model/game_tokenizer.py](../../src/world_model/game_tokenizer.py)

`StateEncoder.player_encoder` is `nn.Linear(22, …)` — built on the
assumption that `_encode_player()` returns 11 features per player (player +
opponent → 22). But the early-return path used `np.zeros(12)`:

```python
if player is None:
    return np.zeros(12, dtype=np.float32)   # ← off by one
```

leading to:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x23 and 22x128)
```

Fixed to `np.zeros(11, dtype=np.float32)`.

## Bug 3 — PowerShell pipeline buffering hides progress

**File:** [scripts/train_pipeline.py](../../scripts/train_pipeline.py)

PowerShell's `Tee-Object`, `Out-String`, and `Start-Process
-RedirectStandardOutput` all buffer Python child stdout for many tens of
seconds, so a healthy training run looks frozen.

### Fix

Three layers of defence at the top of `train_pipeline.py`:

1. `sys.stdout.reconfigure(line_buffering=True)` (Python 3.7+).
2. `logging.basicConfig(force=True, handlers=[StreamHandler, FileHandler])`.
3. A dedicated `FileHandler` writing to `runs/train_pipeline.log` directly,
   so progress is always visible regardless of how the parent shell
   buffers stdout.

We also added a one-line GPU diagnostic at stage 5 entry:

```
GPU check: torch=2.5.1+cu121 cuda_available=True device_count=1 cuda_version=12.1
GPU 0: NVIDIA GeForce RTX 3060
JEPA training: device=cuda epochs=2 batch=64
```

so the operator can confirm CUDA is actually engaged in one glance.

## Lessons captured

- Long-running Python processes on Windows must be invoked **without**
  PowerShell pipes (or with explicit file handlers) to avoid silent stdout
  buffering.
- For npz dumps of long sequences, **always** stack per-feature, never
  per-step — np.savez metadata overhead scales linearly with the key
  count.
- Add a startup diagnostic that prints `device`, `dtype`, and parameter
  count before any heavy work begins; it is the cheapest way to catch
  CPU-fallback regressions.
