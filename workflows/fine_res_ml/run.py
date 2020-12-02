import os
import sys

import download_data
import tempfile
import joblib
import subprocess

cache = joblib.Memory(location="/home/noahb/joblib_cache/fine-res-ml", verbose=10)


@cache.cache
def train(path: str, config: str, output: str):
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(config)
        f.flush()
        subprocess.check_call(
            [sys.executable, "-m", "fv3fit.sklearn", path, f.name, output]
        )


fine_res = "gs://vcm-ml-experiments/2020-05-27-40day-fine-res-coarsening/"
local_fine_res = "/home/noahb/data/dev/2020-11-25-fine-res.zarr"
local_ml_final = "/home/noahb/data/dev/2020-11-25-fine-res/ml.zarr"
trained = "gs://vcm-ml-scratch/noah/trained-ml"
config = "workflows/fine_res_ml/training.yaml"

cache.cache(download_data.fine_res_to_zarr)(fine_res, local_fine_res)
cache.cache(download_data.save_fine_res_to_zarr)(local_fine_res, local_ml_final)

with open(config) as f:
    train(local_ml_final, f.read(), trained)