#!/usr/bin/env bash
set -e

DATASET=dataset
MODEL=resnet18
LR=3e-3
IMSIZE=224
AUG=medium
N_EPOCHS=30
BS=256
N_WORKERS=10
FAST=False
DEBUG=False


# bash argparse
while (( "$#" )); do
  case "$1" in
    --dataset)
      DATASET=$2
      shift 2
      ;;
    --model)
      MODEL=$2
      shift 2
      ;;
    --n-epochs)
      N_EPOCHS=$2
      shift 2
      ;;
    --image-size)
      IMSIZE=$2
      shift 2
      ;;
    --aug)
      AUG=$2
      shift 2
      ;;
    --lr)
      LR=$2
      shift 2
      ;;
    --batch-size)
      BS=$2
      shift 2
      ;;
    --n-workers)
      N_WORKERS=$2
      shift 2
      ;;
    --fast)
      FAST=True
      shift 1
      ;;
    --debug)
      DEBUG=True
      shift 1
      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done

for fold in 0 1 2 3 4; do
    notes="${MODEL}.multistep.lr_${LR}.n_epochs_${N_EPOCHS}.aug_${AUG}"
    if [ "${FAST}" == "True" ]; then
        notes="${notes}.fast"
    fi

    python run_nn.py train-fold \
        --in-csv=./data/${DATASET}/dataset.csv \
        --in-dir=./data/${DATASET} \
        --model=${MODEL} \
        --fold=${fold} \
        --n_epochs=${N_EPOCHS} \
        --image-size=${IMSIZE} \
        --augmentation=${AUG} \
        --learning-rate=${LR} \
        --batch-size=${BS} \
        --n-workers=${N_WORKERS} \
        --fast=${FAST} \
        --logdir=./logs/${DATASET}/${notes}/fold_${fold} \
        --verbose=${DEBUG}
done
