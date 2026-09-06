# Remote GPU run instructions

These instructions are for a Linux machine with two NVIDIA GPUs (24 GB each),
CUDA drivers, Git, and Conda installed. Run all commands from the repository
root unless noted otherwise.

## What I need you to do

Please run the one-step end-to-end RL smoke test on either two 24 GB GPUs or one
48/80 GB GPU. The purpose is to verify the complete pipeline:

```text
response generation -> reward computation -> log-probability computation
-> backward pass -> full-parameter Adam optimizer update
```

Generation, reward computation, and log-probability computation have already
worked on a single RTX 3090. The remaining issue is that a single 24 GB GPU
runs out of memory at the final full-parameter Adam update.

The merged SFT checkpoint path and the reward-judge API settings will be shared
privately. They are intentionally not stored in this public repository.

## 1. Clone and create the environment

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
conda create -n medix-rl python=3.12 -y
conda activate medix-rl
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`flash-attn` may need to be installed after PyTorch with:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## 2. Download the dataset

The preparation script expects the Hugging Face dataset at
`../data/medix-rl-data` relative to this repository. One reproducible option is:

```bash
mkdir -p ../data
huggingface-cli download MBZUAI/medix-rl-data \
  --repo-type dataset \
  --local-dir ../data/medix-rl-data
python scripts/create_smoke_dataset.py
```

The default smoke test downloads `Qwen/Qwen3-VL-2B-Instruct` automatically.
If the merged SFT checkpoint is available on the server, point `MODEL_PATH` to
it instead.

## 3. Configure the reward judge

The medical reward uses an OpenAI-compatible chat endpoint. Export these values
in the shell; do not commit them:

```bash
export LLM_BASE_URL="https://<host>/v1"
export LLM_API_KEY="<api-key>"
export LLM_MODEL_NAME="<judge-model-name>"
```

## 4. Run the one-step end-to-end test

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
bash examples/medix-r1_2b_3090_smoke.sh 2>&1 | tee medix_smoke.log
```

Success means the run completes one optimizer update without CUDA OOM. Please
return the following after the run:

- whether one optimizer update completed successfully;
- `medix_smoke.log`;
- the output of `nvidia-smi` showing the GPU model and memory;
- if it fails, the full traceback and the stage at which it failed.

## Optional short run

After the smoke test succeeds, run up to eight steps with the merged SFT model:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/absolute/path/to/medix-sft-2b-merged \
TRAIN_DATA=../data/medix-rl-data-smoke/train.parquet \
VAL_DATA=../data/medix-rl-data-smoke/test.parquet \
MAX_STEPS=8 \
bash examples/medix-r1_2b_3090_rl_small.sh 2>&1 | tee medix_rl_small.log
```

## Notes

- Recommended system RAM: at least 64 GB because parameter and optimizer
  offloading are enabled.
- First launch downloads the base model and embedding model, so Hugging Face
  access and sufficient disk space are required.
- The scripts default to two GPUs. A 48/80 GB single-GPU machine requires
  changing `trainer.n_gpus_per_node=2` to `1` in the selected launch script and
  setting `CUDA_VISIBLE_DEVICES=0`.
