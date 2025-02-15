#!/bin/sh

GCS_SRC="gs://vcm-ml-data/2019-12-02-40-day-X-SHiELD-simulation-C384-restart-files"
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy"
SRC_RESOLUTION=384
TARGET_RESOLUTION=48
GCS_DST="gs://vcm-ml-intermediate/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts"

user=$(whoami)
user=${user,,}

python -m fv3net.pipelines.coarsen_restarts\
    $GCS_SRC \
    $GCS_GRIDSPEC \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    $GCS_DST \
    --runner DataflowRunner \
    --job_name coarsen-restarts-$user \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 3 \
    --max_num_workers 50 \
    --disk_size_gb 50 \
    --worker_machine_type n1-highmem-4 \
    --setup_file ./setup.py \
    --extra_package external/vcm/dist/vcm-0.1.1.tar.gz \
    --extra_package external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz