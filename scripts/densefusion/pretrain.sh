#!/bin/bash
set -x
## multinode
WORLD_SIZE=$1
RANK=$2
MASTER_PORT=$3
MASTER_ADDR=$4

echo $WORLD_SIZE
echo $RANK
echo $MASTER_PORT
echo $MASTER_ADDR

DATA_PATH="/your_data_path/DenseFusion-1M.json"
IMAGE_FOLDER="your_data_path/"

MODEL="densefusion-7b-pretrain-llava-v1.5"

export CKPT_PATH="./checkpoints/densefusion-7b-prealign-llava-v1.5/mm_projector.bin"
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=12
export BASE_LR=2e-5
export GRADIENT_ACCU_STEPS=2


torchrun --nproc_per_node=8 --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
densefusion/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${CKPT_PATH} \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --tune_vit_from_layer ${TUNE_VIT_FROM} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${MODEL} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2100  \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb tensorboard \
