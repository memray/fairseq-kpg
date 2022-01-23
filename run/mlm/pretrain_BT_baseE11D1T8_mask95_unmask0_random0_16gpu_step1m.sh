#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export EXP_NAME=BT.baseE11D1T8.mask95.unmask0.random0.step1m.lr1e4.bs256
export PROJECT_DIR=/export/share/ruimeng/exp/pretrain/$EXP_NAME
export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
mkdir -p $PROJECT_DIR
echo $PROJECT_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=4
export MAX_STEPS=1000000

cd /export/share/ruimeng/project/te/fairseq-kpg
nohup /export/share/ruimeng/env/anaconda/envs/kp/bin/python train.py /export/share/ruimeng/data/wiki/paragraph/train/:/export/share/ruimeng/data/books1/train/ --valid-data /export/share/ruimeng/data/wiki/paragraph/valid/:/export/share/ruimeng/data/books1/valid/ --validate-interval-updates 5000 --save-dir $PROJECT_DIR/ckpts --task mlm_otf --criterion masked_lm --arch bottleneck_bert_base_E11D1 --fuse-bottleneck token --bottleneck-tokens 8 --mask-prob 0.95 --valid-mask-prob 0.15 --leave-unmasked-prob 0.0 --random-token-prob 0.0 --bpe hf_pretrained_bpe --bpe-vocab /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/merges.txt --dict-path /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --lr 1e-4 --lr-scheduler polynomial_decay --report-accuracy --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --no-bos-eos --min-tokens-per-sample 16 --tokens-per-sample 512 --max-positions 512 --batch-size 16 --update-freq 1 --batch-size-valid 16 --text-field text --clip-norm 0.0 --save-interval-updates 10000 --warmup-updates 10000 --total-num-update $MAX_STEPS --max-update $MAX_STEPS --num-workers $NUM_WORKER --find-unused-parameters --memory-efficient-fp16 --ddp-backend=no_c10d --wandb-project bottleneck_bert > $PROJECT_DIR/nohup.log 2>&1 &

