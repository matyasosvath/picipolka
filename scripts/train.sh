DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
DIR="${DIR%/*}"
# DIR="${DIR%/*}/training"

export PYTHONPATH="${DIR}:${PYTHONPATH}"

ARGS="--project-name thesis \
    --model-name cnn \
    --dataset-name mnist
    --optimizer adam \
    --batch-size 128 \
    --n-epochs 10 \
    --lr 0.01 \
    --gamma 0.7 \
    --use-cuda true \
    --cuda-id 0 \
    --seed 42"

# ensure all processes are killed if the user interrupts the script (with Ctrl+C).
(trap 'kill 0' SIGINT; python ${DIR}/training/cli.py $(echo ${ARGS}) & wait)
