# Two-stage medical VLM training

The standard experiment uses one consistent model lineage:

```text
Qwen3-VL-8B-Instruct
  -> LLaMA-Factory LoRA SFT
  -> merged SFT checkpoint
  -> GSPO reinforcement learning
```

## 1. Prepare the SFT subset

Run this command from the project root. The preparation script samples from the
RL training data and excludes VQA-RAD records to avoid benchmark leakage.

```bash
python scripts/prepare_llamafactory_sft.py \
  --input data/medix-rl-data/data/train-00000-of-00049.parquet \
  --output-dir sft_data \
  --train-size 224 \
  --val-size 32
```

## 2. Run SFT and merge the LoRA adapter

Run these commands from `training/`:

```bash
llamafactory-cli train examples/qwen3vl_8b_lora_sft.yaml
llamafactory-cli export examples/qwen3vl_8b_lora_merge.yaml
```

The merged model is written to:

```text
training/checkpoints/medix-sft-8b-merged
```

## 3. Start GSPO from the SFT model

```bash
bash examples/medix-r1_8b_gspo.sh
```

The GSPO script defaults to the merged SFT checkpoint instead of the base
model. `MODEL_PATH` and `N_GPUS` remain configurable:

```bash
MODEL_PATH=/path/to/another/sft-model N_GPUS=2 \
  bash examples/medix-r1_8b_gspo.sh
```
