DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

export MODEL_NAME=puli2-gpt
export SHOW_DATA=1


TOTAL_STEPS=1000
CHECKPOINT_STEPS=1000
CHECKPOINT_PATH="${DIR}/../model_checkpoitns/${MODEL_NAME}"}

DATASETS="common_crawl"

ARGS="--model-name ${BASE_MODEL} \
    --tokenizer-name ${BASE_MODEL} \
    --project-name puli \
    --optimizer adam \
    --num-epochs 10 \
    --seed 42"

# ensure all processes are killed if the user interrupts the script (with Ctrl+C).
(trap 'kill 0' SIGINT; \
python ${DIR}/train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
wait)
