WORLD_SIZE=$1
RANK=$2
MASTER_PORT=$3
MASTER_ADDR=$4
bash scripts/densefusion/prealign_s2.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
bash scripts/densefusion/pretrain_s2.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
bash scripts/densefusion/finetune_s2.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
