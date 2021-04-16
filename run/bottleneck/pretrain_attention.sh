#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export TOKENIZERS_PARALLELISM=false
export WANDB_NAME=bart_kpgen_pretrain.lr=5e-5
export WANDB_API_KEY=xxxxx
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/ubuntu/efs/rum20/fairseq-kpg/fairseq_cli

NUM_BOTTLENECK_TOKEN=4
MAX_SEQ_LEN=512
UPDATE_FREQ=16
SEED=7

# w/o loading pretrained checkpoint
/home/ubuntu/efs/.conda/kp/bin/python3.7 train.py /home/ubuntu/efs/rum20/data/roberta/data/wiki/train/:/home/ubuntu/efs/rum20/data/roberta/data/book/train --valid-data /home/ubuntu/efs/rum20/data/roberta/data/wiki/valid/:/home/ubuntu/efs/rum20/data/roberta/data/book/valid --validate-interval 10 --save-dir /home/ubuntu/efs/rum20/data/roberta/ckpt/ --task mlm_otf --arch bottleneck_bert_base --bpe hf_pretrained_bpe --bpe-vocab /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/merges.txt --dict-path /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --lr 5e-6 --update-freq $UPDATE_FREQ --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed $SEED --fixed-validation-seed $SEED --bottleneck-tokens $NUM_BOTTLENECK_TOKEN --max-tokens $MAX_SEQ_LEN --no-bos-eos --text-field text --clip-norm 0.1 --save-interval-updates 5000 --warmup-updates 5000 --total-num-update 100000 --num-workers 12 --find-unused-parameters --fp16 --ddp-backend=no_c10d --fuse-bottleneck attention --wandb-project bottleneck_bert

# w/ loading pretrained checkpoint (not working yet)
#/home/ubuntu/efs/.conda/kp/bin/python3.7 train.py /home/ubuntu/efs/rum20/data/roberta/data/wiki/train/:/home/ubuntu/efs/rum20/data/roberta/data/book/train --valid-data /home/ubuntu/efs/rum20/data/roberta/data/wiki/valid/:/home/ubuntu/efs/rum20/data/roberta/data/book/valid --validate-interval 10 --save-dir /home/ubuntu/efs/rum20/data/roberta/ckpt/ --task mlm_otf --arch bottleneck_bert_base --restore-file /home/ubuntu/efs/rum20/data/roberta/cache/roberta.base/model.pt --bpe hf_pretrained_bpe --bpe-vocab /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/merges.txt --dict-path /home/ubuntu/efs/rum20/data/roberta/cache/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --lr 5e-6 --update-freq $UPDATE_FREQ --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed $SEED --fixed-validation-seed $SEED --bottleneck-tokens $NUM_BOTTLENECK_TOKEN --max-tokens $MAX_SEQ_LEN --no-bos-eos --text-field text --clip-norm 0.1 --save-interval-updates 5000 --warmup-updates 5000 --total-num-update 100000 --num-workers 12 --find-unused-parameters --fp16 --ddp-backend=no_c10d --fuse-bottleneck attention --wandb-project bottleneck_bert
