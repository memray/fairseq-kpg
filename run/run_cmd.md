# pretraining
cd /export/share/ruimeng/project/te/fairseq-kpg
source run/mlm/pretrain_BT_baseE11D1T8_mask95_unmask0_random0_16gpu_step1m_resume.sh
source run/mlm/pretrain_BT_baseE11D1T8_mask95_unmask0_random0_16gpu_step1m.sh
source run/mlm/pretrain_BT_baseE11D1T8_mask50_unmask0_random0_16gpu_step1m.sh
source run/mlm/pretrain_BT_baseE11D1T4_mask50_unmask0_random0_16gpu_step1m.sh
source run/mlm/pretrain_BT_baseE11D1T1_mask50_unmask0_random0_16gpu_step1m.sh
source run/mlm/pretrain_BT_baseE11D1T1_maskprob05_16gpu_step1m.sh

source run/mlm/pretrain_bottleneck_token_baseE12D3_maskprob015_16gpu_step1m_resume.sh
source run/mlm/pretrain_bottleneck_token_baseE12D3_maskprob015_16gpu_step1m.sh
source run/mlm/pretrain_bottleneck_token_baseE12D3_maskprob03_16gpu_step1m_resume.sh
source run/mlm/pretrain_bottleneck_token_baseE12D3_maskprob03_16gpu_step1m.sh
source run/mlm/pretrain_roberta_base_16gpu_step1m_resume.sh
source run/mlm/pretrain_roberta_base_16gpu_step1m.sh
source run/mlm/pretrain_roberta_base_8gpu.step100k.sh
source run/mlm/pretrain_roberta_base_8gpu_step500k_resume.sh
source run/mlm/pretrain_roberta_base_8gpu_step500k.sh
source run/mlm/pretrain_bottleneck_attention_base_8gpu.sh
source run/mlm/pretrain_bottleneck_token_base_8gpu.sh
source run/mlm/pretrain_roberta_base_8gpu.sh

# fairseq GLUE
cd /export/share/ruimeng/project/te/fairseq-kpg/run/glue/
mkdir /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32/
# done: RTE MRPC STSB 1268541 1268542 1271645
# todo: CoLA  SST-2 1275955 1275956 QQP QNLI 1336448 1336449
source run_baseline_glue_g8.sh QQP /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.bs32/lr_0.00001/MNLI/seed_100/checkpoint_best.pt /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32 > /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32/QQP.nohup.out 2>&1 &
source run_baseline_glue_g8.sh QNLI /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.bs32/lr_0.00001/MNLI/seed_100/checkpoint_best.pt /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32 > /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32/QNLI.nohup.out 2>&1 &
source run_baseline_glue_g8.sh STS-B /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.bs32/lr_0.00001/MNLI/seed_100/checkpoint_best.pt /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32 > /export/home/exp/fairseq_glue/roberta_base.step1m.lr1e4.bs256.16gpu.step1m.fromMNLI.bs32/STS-B.nohup.out 2>&1 &


## Bart-MAG
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg/
source run/kp_DA/bart-mag_4gpu.sh


## Freezable
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg/
source run/bottleneck/pretrain_token.sh
source run/bottleneck/pretrain_attention.sh

## RoBERTa
https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md
8x32GB V100 GPUs. Each GPU uses a batch size of 16 sequences ($MAX_SENTENCES) and accumulates gradients to further increase the batch size by 16x ($UPDATE_FREQ), for a total batch size of 2048 sequences.

batch size | peak learning rate
---|---
256 | 0.0001=1e-4
2048 | 0.0005=5e-4
8192 | 0.0007=7e-4
ours 64 | 1e-5
LUKE 2048 | 1e-5


## Ours
let's do 256 equivalent batch size

On a single 16gb V100, max-tokens=512
  - update-freq=4, the dynamic batch size is 7~8, ~1.5min per 100step, occasionally OOM
  - update-freq=8, the dynamic batch size is 15.x, 2min per 100step, occasionally OOM, 3k step per hr, 72k step per day
  - update-freq=16, the dynamic batch size is 30.x, 4min per 100step, occasionally OOM, 1.5k step per hr, 36k step per day

With A100*8, expect max-tokens=512*4=2048, update-freq=4, the dynamic batch size is base*num_gpu*max_token_ratio=8*8*(2048/512)=256


## Reproduce RoBERTa
export/share/ruimeng/data/wiki/paragraph/train/;/export/share/ruimeng/data/books1/train/ --valid-data /export/share/ruimeng/data/wiki/paragraph/valid/;/export/share/ruimeng/data/books1/valid/ --validate-interval 1000 --save-dir /export/share/ruimeng/exp/pretrain/roberta/ckpts --task mlm_otf --arch roberta_base --bpe hf_pretrained_bpe --bpe-vocab /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/merges.txt --dict-path /export/share/ruimeng/data/misc/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --lr 5e-4 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --no-bos-eos --min-tokens-per-sample 32 --tokens-per-sample 500 --max-positions 512 --max-tokens 1024 --update-freq 8 --text-field text --clip-norm 0.0 --save-interval-updates 5000 --warmup-updates 5000 --total-num-update 100000 --num-workers 32 --find-unused-parameters --memory-efficient-fp16 --ddp-backend=no_c10d --wandb-project bottleneck_bert



## Commands
cd ~/efs/rum20/exps/
nohup ~/efs/rum20/fairseq-kpg/run/pretrain_1gpu_gcp.sh > pretrain_1gpu_5e6_200k.log &
chmod +x ../fairseq-kpg/run/pretrain_1gpu_gcp.sh
vim ../fairseq-kpg/run/pretrain_1gpu_gcp.sh

### on single A100, usage=40gb, avg_batchsize=60
https://wandb.ai/memray/transfer_kp_wiki?workspace=user-memray
#### diverged after an "overflow detected", try --clip-norm 0.1
#### diverged (see gnorm and valid of https://wandb.ai/memray/transfer_kp_wiki/runs/d14hf3ma?workspace=user-memray) after around 5k steps, may because of too large learning rate. Set peak_lr=1e-5, valid_interval=2500, warm_up=5000  
#### 1e5 looks like overfitting. Set peak_lr=1e-6, valid_interval=2000 
#### 1e6, step=100k, eventually doesn't reach better loss
#### try 5e6, step=200k, freq=24, warmup=10k, pid=11444
train.py ~/efs/rum20/data/wiki/processed/json_phrase/train/ --valid-data ~/efs/rum20/data/wiki/processed/json_phrase/valid/ --validate-interval 2500 --save-dir ~/efs/rum20/exps/kppretrain_bart_wiki_lr1e5/ckpts --task keyphrasification_pretrain --max-source-length 512 --max-target-length 256 --max-phrase-len 6 --max-target-phrases 16 --phrase-corr-rate 0.1 --random-span-rate 0.05 --arch bart_large --restore-file ~/efs/rum20/data/kp/cache/bart.large/model.pt --bpe hf_pretrained_bpe --bpe-vocab ~/efs/rum20/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges ~/efs/rum20/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path ~/efs/rum20/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-08 --lr 5e-6 --update-freq 10 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --max-tokens 512 --clip-norm 0.1 --save-interval-updates 5000 --warmup-updates 5000 --total-num-update 200000 --num-workers 12 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_wiki



## GCP A100
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

## AWS V100*4
mkdir ~/efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-68c8399d.efs.us-east-1.amazonaws.com:/ efs

	conda config --set env_prompt '({name})'
	conda activate ~/efs/.conda/kp


cd /home/ubuntu/efs/rum20/fairseq-kpg
nohup bash run/kp/wikida-kptimes_4gpu.sh > /home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled-DA_kptimes-NP_TL/train.log 2>&1 &

nohup bash run/kp/wikida-kp20k_4gpu.sh > /home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled-DA_kp20k-NP_TL/train.log 2>&1 &
(OOM) nohup bash run/kp/wikida-openkp_4gpu.sh > /home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled-DA_openkp-NP_TL/train.log 2>&1 &
