#!/usr/bin/env bash


# args: checkpoint_folder step

# PREFIX = /mnt/vdb/
# CKP = electnf-bs5-ws64-20200831/checkpoint_32_1000000.pt

TASK=$1
PREFIX=$2
DIR=$3
CKP=$4
LR=$5
SEED=$6

REL_POS=""

if [[ $DIR  == *"rel-pos"* ]]; then
  echo "use --rel-pos"
  REL_POS=--rel-pos
fi


ARCH=binb_base
N_EPOCH=5
WEIGHT_DECAY=0.1
SENT_PER_GPU=32
WARMUP_RATIO=0.06
# valid 2 times per epoch
VALID_FREQ=2

BERT_MODEL_PATH=$PREFIX/binb-model/$CKP # done.

if [ ! -e $BERT_MODEL_PATH ]; then
    echo "Checkpoint $BERT_MODEL_PATH doesn't exist"
    exit 0
fi

ROOT=binb-glue-results
GLUE_DIR=glue-data
BINB_DATA_NAME=data-binb-0.01-0.03-20200527
DATA_DIR=$PREFIX/$GLUE_DIR/$BINB_DATA_NAME/$TASK/data-bin
BINB_DATA_DIR=$PREFIX/$GLUE_DIR/$BINB_DATA_NAME/$TASK/binb-data-bin # done. good.

OPTION=""
METRIC=accuracy
N_CLASSES=2

if [ "$TASK" = "MNLI" ]
then
N_EPOCH=10
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
fi

if [ "$TASK" = "QQP" ]
then
N_EPOCH=10
fi

if [ "$TASK" = "CoLA" ]
then
METRIC=mcc
fi

if [ "$TASK" = "STS-B" ]
then
METRIC=pearson_spearman
N_CLASSES=1
OPTION="--regression-target"
fi

echo $DATA_DIR

OUTPUT_PATH=$PREFIX/$ROOT/${CKP}/${TASK}/$LR-$SEED
echo $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi
cmd="python train.py $DATA_DIR --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--binb-data $BINB_DATA_DIR \
--binb-running-lambda 0.1 --update-binb-scale 0 \
--binb-emb-zero-init 1 --fix-dict-shift True \
--restore-file $BERT_MODEL_PATH \
--max-positions 512 \
--max-sentences $SENT_PER_GPU \
--max-tokens 4400 \
--task sentence_prediction \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch $ARCH \
--criterion sentence_prediction $OPTION \
--num-classes $N_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 --validate-interval-updates $VALID_FREQ \
--lr-scheduler polynomial_decay --lr $LR --warmup-ratio $WARMUP_RATIO \
--max-epoch $N_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-best-checkpoints \
--find-unused-parameters --skip-invalid-size-inputs-valid-test --truncate-sequence --embedding-normalize \
--tensorboard-logdir . \
--best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric $REL_POS | tee $OUTPUT_PATH/train_log.txt"

echo $cmd
