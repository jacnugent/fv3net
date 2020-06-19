#!/bin/bash

set -x
set -e

TRAINING_DATA=gs://vcm-ml-data/testing-noah/7563d34ad7dd6bcc716202a0f0c8123653f50ca4/training_data/
OUTPUT=gs://vcm-ml-scratch/annak/2020-06-09/sklearn_train/

#mkdir -p data
#gsutil -m rsync -d -r "$TRAINING_DATA" data

python -m fv3net.regression.sklearn \
    data \
    train_sklearn_model_onestep_source.yml  \
    $OUTPUT \
    --timesteps-file times.json