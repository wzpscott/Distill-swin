CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
deepspeed $(dirname "$0")/train.py --deepspeed $CONFIG --gpus 2 --deepspeed_config './mmcv_custom/ds_config.json'