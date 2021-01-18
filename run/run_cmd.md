## RoBERTa
https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md
8x32GB V100 GPUs. Each GPU uses a batch size of 16 sequences ($MAX_SENTENCES) and accumulates gradients to further increase the batch size by 16x ($UPDATE_FREQ), for a total batch size of 2048 sequences.

## Ours
let's do 256 equivalent batch size

On a single 16gb V100, max-tokens=512
  - update-freq=4, the dynamic batch size is 7~8, ~1.5min per 100step, occasionally OOM
  - update-freq=8, the dynamic batch size is 15.x, 2min per 100step, occasionally OOM, 3k step per hr, 72k step per day
  - update-freq=16, the dynamic batch size is 30.x, 4min per 100step, occasionally OOM, 1.5k step per hr, 36k step per day

With A100*8, expect max-tokens=512*4=2048, update-freq=4, the dynamic batch size is base*num_gpu*max_token_ratio=8*8*(2048/512)=256


## Commands
cd ~/efs/rum20/exps/
chmod +x ../fairseq-kpg/run/pretrain_1gpu.sh
nohup ../fairseq-kpg/run/pretrain_1gpu.sh > pretrain_1gpu.log &
2727

## A100
export TOKENIZERS_PARALLELISM=false
export WANDB_NAME=bart_kpgen_pretrain
export WANDB_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
SAVE_INTERVAL=5000      # how often to save checkpoints
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
MAX_TOKEN=2048
UPDATE_FREQ=4
NUM_WORKER=96
DROPOUT=0.1
DATA_DIR=/home/ubuntu/efs/rum20/data/wiki/processed/json_phrase/train/
VALID_DATA_DIR=/home/ubuntu/efs/rum20/data/wiki/processed/json_phrase/valid/
SAVE_DIR=/home/ubuntu/efs/rum20/exps/kppretrain_bart_wiki/ckpts

$DATA_DIR --valid-data $VALID_DATA_DIR --validate-interval 5000 --save-dir $SAVE_DIR --task keyphrasification_pretrain --max-source-length 512 --max-target-length 256 --max-phrase-len 6 --max-target-phrases 16 --phrase-corr-rate 0.1 --random-span-rate 0.05 --arch bart_large --restore-file /home/ubuntu/efs/rum20/data/kp/cache/bart.large/model.pt
--bpe hf_pretrained_bpe --bpe-vocab /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout $DROPOUT
--ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed
--reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 --lr $PEAK_LR
--update-freq $UPDATE_FREQ --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout $DROPOUT --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --max-tokens $MAX_TOKEN --save-interval-updates $SAVE_INTERVAL --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES --num-workers $NUM_WORKER --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_wiki
