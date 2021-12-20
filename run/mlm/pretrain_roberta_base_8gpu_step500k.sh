#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export EXP_NAME=roberta_base.step500k.lr6e4.bs256.8gpu
export PROJECT_DIR=/export/share/ruimeng/exp/pretrain/$EXP_NAME
export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
mkdir -p $PROJECT_DIR
echo $PROJECT_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=4
export MAX_STEPS=500000

cd /export/share/ruimeng/project/te/fairseq-kpg
nohup /export/share/ruimeng/env/anaconda/envs/kp/bin/python train.py /export/share/ruimeng/data/wiki/paragraph/train/:/export/share/ruimeng/data/books1/train/ --valid-data /export/share/ruimeng/data/wiki/paragraph/valid/:/export/share/ruimeng/data/books1/valid/ --save-dir $PROJECT_DIR/ckpts --task mlm_otf --arch roberta_base --bpe hf_pretrained_bpe --bpe-vocab /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/merges.txt --dict-path /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --lr 6e-4 --lr-scheduler polynomial_decay --label-smoothing 0.1 --report-accuracy --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --no-bos-eos --min-tokens-per-sample 16 --tokens-per-sample 510 --max-positions 512 --batch-size 16 --update-freq 2 --batch-size-valid 16 --validate-interval-updates 5000 --text-field text --clip-norm 0.0 --save-interval-updates 10000 --warmup-updates 24000 --total-num-update $MAX_STEPS --max-update $MAX_STEPS --num-workers $NUM_WORKER --find-unused-parameters --memory-efficient-fp16 --ddp-backend=no_c10d --wandb-project bottleneck_bert > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid

