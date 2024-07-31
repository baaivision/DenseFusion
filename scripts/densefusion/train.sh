WORLD_SIZE=$1
RANK=$2
MASTER_PORT=$3
MASTER_ADDR=$4
bash scripts/densefusion/prealign.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
bash scripts/densefusion/pretrain.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
bash scripts/densefusion/finetune.sh $WORLD_SIZE $RANK $MASTER_PORT $MASTER_ADDR
