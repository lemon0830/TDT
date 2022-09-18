#!/usr/bin/env bash
set -e

TPATH=your_root_path
CODE_PATH=${TPATH}/you_code_path

export PYTHONPATH="$PYTHONPATH:${CODE_PATH}/"

sep="===================================="

pip install sklearn
pip install tensorboardX


export DATA_DIR=${TPATH}/glue/MNLI
export TASK_NAME=mnli
export MODEL_PATH=${TPATH}/models/bert-large-uncased
export OUTPUT_PATH=${CODE_PATH}/output
export OMP_NUM_THREADS=1

maxstep=10000
warmup=1000
lr=2e-5
batch=64
seed=42

# 0. 0.5 1

for mrate in 1
do
# 0.5 1 2 4
for krate in 2
do
# 0 2
for maxg in 0
do

filename=bert_large_glue_mnli_maxg${maxg}_krate${krate}_mrate${mrate}_seed${seed}_maxstep${maxstep}_warmup${warmup}_lr${lr}_batch${batch}

# -m torch.distributed.launch --nproc_per_node 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup \
python ${CODE_PATH}/examples/run_glue.py \
            --model_type bert \
            --kl_rate ${krate} \
            --m_rate ${mrate} \
            --maxg ${maxg} \
            --model_name_or_path $MODEL_PATH \
            --task_name $TASK_NAME \
            --do_train \
            --do_lower_case \
            --evaluate_during_training \
            --data_dir $DATA_DIR \
            --max_seq_length 256 \
            --seed ${seed} \
            --per_gpu_train_batch_size=8 \
            --per_gpu_eval_batch_size=32 \
            --learning_rate ${lr} \
            --max_steps ${maxstep} \
            --warmup_steps ${warmup} \
            --gradient_accumulation_steps 1 \
            --save_steps 250 \
            --logging_steps 250 \
            --output_dir $OUTPUT_PATH/$TASK_NAME/${filename} \
> ${CODE_PATH}/log.${filename} 2>&1


wait $!
done

wait $!
done

wait $!
done


