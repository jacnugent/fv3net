import loaders
import fv3fit

ml_model = "gs://vcm-ml-experiments/2021-02-15-hybrid-full-physics/hybrid-full-physics/trained_model"
path = "2021-02-15-hybrid-full-physics-training-data"


config = fv3fit.load_training_config(ml_model)
mapping_kwargs = config.batch_kwargs["mapping_kwargs"]
mapper = loaders.mappers.open_fine_resolution_nudging_hybrid(
    data_paths=None, **mapping_kwargs
)
loaders.mappers.mapper_to_local(mapper, path)