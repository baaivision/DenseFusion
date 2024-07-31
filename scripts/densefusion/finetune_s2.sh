#!/bin/bash
set -x

WORLD_SIZE=$1
RANK=$2
MASTER_PORT=$3
MASTER_ADDR=$4

echo $WORLD_SIZE
echo $RANK
echo $MASTER_PORT
echo $MASTER_ADDR

MODEL="densefusion-7b-finetune-llava-s2"

DATA_PATH="your_data_path/llava_v1_5_mix665k.json"
IMAGE_FOLDER="your_data_path/"

export CKPT_PATH=checkpoints/densefusion-7b-pretrain-llava-s2
export VIT_PATH=checkpoints/densefusion-7b-pretrain-llava-s2/vision_tower
export LEARNIG_RATE=2e-5


torchrun --nproc_per_node=8 --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
densefusion/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --data_path $DATA_PATH \
    --s2 True \
    --s2_scales "336,672,1008" \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VIT_PATH} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${MODEL} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb tensorboard