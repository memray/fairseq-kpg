#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export TOKENIZERS_PARALLELISM=false
export WANDB_NAME=bart_kpgen_pretrain
export WANDB_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export CUDA_VISIBLE_DEVICES=0

cd /home/ubuntu/efs/rum20/fairseq-kpg/fairseq_cli

UPDATE_FREQ=16

/home/ubuntu/efs/.conda/kp/bin/python3.7 train.py /home/ubuntu/efs/rum20/data/wiki/processed/json_phrase/train/ --valid-data /home/ubuntu/efs/rum20/data/wiki/processed/json_phrase/valid/ --validate-interval 2500 --save-dir /home/ubuntu/efs/rum20/exps/kppretrain_bart_wiki/ckpts --task keyphrasification_pretrain --max-source-length 512 --max-target-length 256 --max-phrase-len 6 --max-target-phrases 16 --phrase-corr-rate 0.1 --random-span-rate 0.05 --arch bart_large --restore-file /home/ubuntu/efs/rum20/data/kp/cache/bart.large/model.pt --bpe hf_pretrained_bpe --bpe-vocab /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 --lr 5e-4 --update-freq $UPDATE_FREQ --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --max-tokens 512 --save-interval-updates 2500 --warmup-updates 5000 --total-num-update 50000 --num-workers 8 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_wiki
