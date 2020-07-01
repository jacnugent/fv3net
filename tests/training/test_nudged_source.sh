#!/bin/bash

TRAINING_DATA=gs://vcm-ml-code-testing-data/andrep/test-nudging-workflow/nudging
OUTPUT=gs://vcm-ml-scratch/annak/test-nudging-workflow/test-training/

gsutil -m rm -r $OUTPUT

python -m fv3net.regression.sklearn \
    $TRAINING_DATA \
    tests/sklearn_regression/train_sklearn_model_nudged_source.yaml  \
    $OUTPUT \
    --no-train-subdir-append