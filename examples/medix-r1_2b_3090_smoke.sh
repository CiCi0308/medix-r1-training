#!/usr/bin/env bash

set -euo pipefail
set -x

# Minimal end-to-end RL smoke test for two 24 GB RTX 3090 GPUs.
# This is intended to validate the data -> rollout -> reward -> update path,
# not to produce a useful medical checkpoint.
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-../data/medix-rl-data-smoke/train.parquet}"
VAL_DATA="${VAL_DATA:-../data/medix-rl-data-smoke/test.parquet}"

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
    echo "Smoke-test data not found."
    echo "Create it first with: python scripts/create_smoke_dataset.py"
    exit 1
fi

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=problem \
    data.answer_key=solution \
    data.image_key=image \
    data.max_prompt_length=1024 \
    data.max_response_length=384 \
    data.min_pixels=50176 \
    data.max_pixels=262144 \
    data.filter_overlong_prompts_workers=1 \
    data.format_prompt=./examples/format_prompt/medical_format.jinja \
    data.rollout_batch_size=2 \
    data.mini_rollout_batch_size=2 \
    data.val_batch_size=2 \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.padding_free=false \
    worker.actor.dynamic_batching=false \
    worker.actor.use_torch_compile=false \
    worker.actor.fsdp.enable_full_shard=true \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.reward.reward_function=./examples/reward_function/medical.py:compute_score \
    worker.rollout.n=2 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.max_num_batched_tokens=1536 \
    worker.rollout.enforce_eager=true \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.tensor_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=false \
    trainer.project_name=medix_smoke \
    trainer.experiment_name=medix-r1_2b_3090_smoke \
    trainer.logger='[console]' \
    trainer.max_steps=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.val_generations_to_log=1 \
    trainer.max_try_make_batch=2 \
    trainer.save_freq=-1 \
    trainer.save_checkpoint_path=./checkpoints/medix-r1_2b_3090_smoke \
    trainer.find_last_checkpoint=false \
    trainer.n_gpus_per_node=2
