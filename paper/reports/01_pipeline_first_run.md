# Report 01 — First End-to-End Pipeline Run

**Date:** 2026-04-28
**Hardware:** NVIDIA GeForce RTX 3060 (12 GB), driver 581.57, CUDA 12.1
**Stack:** Python 3.10, PyTorch 2.5.1+cu121, torch-geometric 2.7.0, Neo4j 5 (docker `mtg-neo4j`)

## Goal

Run the full V + M + C + JEPA + KG training pipeline against the live Neo4j
graph and the local GPU; confirm the world model trains end-to-end and that
GPU acceleration is actually engaged.

## Result (TL;DR)

| Stage | Description | Wall time | Output |
|------|-------------|-----------|--------|
| 4    | Self-play trajectory collection (`Heuristic` vs `Random`, 5 games) | ~12 s | 5 npz + `metadata.json` |
| 4.5  | KG enrichment (synergies, combos, win-rate stats) | < 1 s | proposals + Neo4j writes |
| 5    | JEPA encoder + predictor training (2 epochs, batch 64, 13 709 transition pairs) | ~6 s | `checkpoints/jepa/jepa_final.pt` |
| **Total** | stages 4 → 5 | **27.7 s** | — |

Final epoch metrics:

```
Epoch 2/2 done in 6.2s: avg_total=0.9125 avg_pred=0.9109 avg_kl=0.0082
```

GPU confirmation log line emitted at stage 5 entry:

```
GPU check: torch=2.5.1+cu121 cuda_available=True device_count=1 cuda_version=12.1
GPU 0: NVIDIA GeForce RTX 3060
JEPA training: device=cuda epochs=2 batch=64
```

## Pipeline data flow

```
Scryfall JSON ──► Stage 2 ──► Neo4j (37 384 :Card nodes, legalities)
                                │
                                ▼
                            Stage 3 ──► GraphSAGE 128-d card embeddings
                                            (36 909 cards, written back to :Card.embedding)
                                            cached at data/card_graph_embeddings.pt
                                            │
HeuristicAgent vs RandomAgent ──► Stage 4 ──► TrajectoryStore (5 games, 13 709 transitions)
                                            │
                                            ├─► Stage 4.5 KGEnrichment ──► Neo4j
                                            │       (SYNERGIZES_WITH edges, win-rate stats)
                                            │
                                            ▼
                            Stage 5 train_jepa(device='cuda')
                                            │
                                            ▼
                            checkpoints/jepa/jepa_final.pt
```

## Run command

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'
.\.venv\Scripts\python.exe -u scripts\train_pipeline.py `
    --stage 4 --end-stage 5 --num-games 5 --max-turns 20 `
    --jepa-epochs 2 --jepa-batch-size 32 --no-kg --skip-eval --skip-dream
```

(`--no-kg` disables the optional KG-context fusion in the JEPA encoder; KG
*enrichment* in stage 4.5 still runs.)

## Notes

- World model has **11 751 311** parameters.
- JEPA β warm-up linear from 0 → 1.0 over the first 10 epochs (we ran 2,
  so β stayed at 0.1).
- Action vector dim = 136 (8 action types one-hot + 128-d card embedding).
- Latent dim = 256 (default).
- Per-step throughput: ~36 batches / second at batch 64 on the RTX 3060.

## Next runs

- Re-run with `--num-games 200 --jepa-epochs 80` for the proper paper figures.
- Toggle `--no-kg` off to ablate KG-context fusion (Section 5 ablation).
- Run stage 6 (dream training) and stage 7 (eval game).
