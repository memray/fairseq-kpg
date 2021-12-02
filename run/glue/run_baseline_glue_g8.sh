#!/usr/bin/env bash


# args: checkpoint_folder step

# PREFIX = /mnt/vdb/
# CKP = /models/electra_1/checkpoint_22_727000.pt
# TASK = ALL
# bash run_new_model_glue_g8.sh ALL /mnt/vdb/ hehe /models/electra_1/checkpoint_22_727000.pt

TASK=$1
PREFIX=$2
DIR=$3
CKP=$4
# needs to think about 5,6,7,8.
i=0

if [ "$TASK" = "ALL" ]
then
    for STASK in RTE CoLA MRPC STS-B SST-2 QNLI QQP MNLI
    do
        for SEED in 100 200 300 400 500
        do
            for LR in 0.00001 0.00002 0.00003 0.00004
            do
                CUDA_VISIBLE_DEVICES=$i bash run_bert_glue_g1.sh $STASK $PREFIX $DIR $CKP $LR $SEED  &
                i=$((i+1))
                if [ $i -eq 8 ]; then
                    wait
                    i=0
                fi
            done
        done
    done
else
    for SEED in 100 200 300 400 500
    do
        for LR in 0.00001 0.00002 0.00003 0.00004
        do
            CUDA_VISIBLE_DEVICES=$i bash run_bert_glue_g1.sh $TASK $PREFIX $DIR $CKP $LR $SEED &
            i=$((i+1))
            if [ $i -eq 8 ]; then
                wait
                i=0
            fi
        done
    done
fi
wait