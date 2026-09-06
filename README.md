# MediX-R1 training

This directory contains the SFT configuration and the multimodal RL training
code used to reproduce a 2B medical VLM training pipeline. The current goal is
an end-to-end RL smoke test: generation, reward computation, log-probability
calculation, and one full-parameter Adam update.

## GPU requirement

The supplied smoke test targets either:

- 2 GPUs with 24 GB VRAM each (tested configuration: RTX 3090), or
- 1 GPU with 48/80 GB VRAM after adjusting `trainer.n_gpus_per_node` to `1`.

A single 24 GB GPU reaches the optimizer update but runs out of memory during
the full-parameter Adam step.

## Quick start for a remote GPU

See [GPU_RUN.md](GPU_RUN.md) for the exact environment, data preparation, and
launch commands, the required validation target, and the results to return.
The smallest verification run is:

```bash
python scripts/create_smoke_dataset.py
bash examples/medix-r1_2b_3090_smoke.sh 2>&1 | tee medix_smoke.log
```

Model checkpoints, datasets, API keys, and logs are intentionally excluded
from this repository.

## Training variants

- `examples/medix-r1_2b_3090_smoke.sh`: one-step end-to-end RL verification.
- `examples/medix-r1_2b_3090_rl_small.sh`: short configurable RL run.
- `examples/medix-r1_2b_3090_gspo_600.sh`: 600-step GSPO experiment.
- `TWO_STAGE_TRAINING.md`: SFT-to-RL model lineage and 8B example.

This implementation is based on the veRL-style training code included under
`verl/`.
