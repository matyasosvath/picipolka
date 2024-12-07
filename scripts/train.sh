DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
DIR="${DIR%/*}/training"

ARGS="--project-name thesis \
    --model-name vit \
    --dataset-name mnist
    --optimizer adam \
    --batch-size 128 \
    --num-epochs 10 \
    --lr 0.01 \
    --gamma 0.7 \
    --use-cuda true \
    --cuda-id 0 \
    --seed 42"

# ensure all processes are killed if the user interrupts the script (with Ctrl+C).
(trap 'kill 0' SIGINT; \
python ${DIR}/cli.py $(echo ${ARGS}) \
    & \
wait)
