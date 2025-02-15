from typing import Iterable, Any, Tuple
import numpy as np
import fv3fit
import fv3fit.keras._models.recurrent
import xarray as xr
import tensorflow as tf
import argparse
import random
import vcm
import loaders
import yaml
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Trains model using backpropagation through time."
    )
    parser.add_argument(
        "arrays_dir",
        type=str,
        help=(
            "directory containing TrainingArrays data, "
            "last file is used for validation"
        ),
    )
    parser.add_argument(
        "config", type=str, help="yaml file with training configuration"
    )
    parser.add_argument(
        "model_output_dir", type=str, help="directory to write the trained model"
    )
    return parser


def load_dataset(path):
    ds = xr.open_dataset(path)
    ds.load()
    return ds


def shuffled(values: Iterable[Any]) -> Tuple[Any, ...]:
    values = list(values)
    random.shuffle(values)
    return tuple(values)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    fs = vcm.get_fs(args.arrays_dir)
    filenames = sorted(fs.listdir(args.arrays_dir, detail=False))
    first_filename = filenames[0]

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)
    np.random.seed(config.get("random_seed", 0))
    random.seed(config.get("random_seed", 0) + 1)
    tf.random.set_seed(config.get("random_seed", 0) + 2)

    regularizer = getattr(tf.keras.regularizers, config["regularizer"]["name"])(
        **config["regularizer"]["kwargs"]
    )
    optimizer_class = getattr(tf.keras.optimizers, config["optimizer"]["name"])
    optimizer_kwargs = config["optimizer"]["kwargs"]
    optimizer = optimizer_class(**optimizer_kwargs)
    input_names = config["input_variables"]

    with open(first_filename, "rb") as f:
        ds = xr.open_dataset(f)
        sample_dim_name = ds["air_temperature"].dims[0]
        model = fv3fit.keras._models.recurrent._BPTTTrainer(
            sample_dim_name,
            input_names,
            kernel_regularizer=regularizer,
            optimizer=optimizer,
            **config["hyperparameters"],
        )
        model.fit_statistics(ds)

    train_filenames = filenames[:-1]
    validation = xr.open_dataset(filenames[-1])

    base_epoch = 0
    for i_epoch in range(config["total_epochs"]):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        for i, ds in enumerate(
            loaders.OneAheadIterator(shuffled(train_filenames), function=load_dataset)
        ):
            model.fit(ds, epochs=1)
        val_loss = model.loss(validation)
        print(f"val_loss: {val_loss}")
        dirname = os.path.join(
            args.model_output_dir, f"model-epoch_{epoch:03d}-loss_{val_loss:.04f}"
        )
        os.makedirs(dirname, exist_ok=True)
        # dump doesn't need an estimator's fit methods, it only needs .dump to exist
        # which predictor_model has defined
        fv3fit.dump(model.predictor_model, dirname)  # type: ignore
        if i_epoch == config["decrease_learning_rate_epoch"] - 1:
            optimizer_kwargs["lr"] = config["decreased_learning_rate"]
            # train_keras_model will not be None because we call fit above
            # (assuming there is any training data)
            model.train_keras_model.compile(  # type: ignore
                optimizer=optimizer_class(**optimizer_kwargs), loss=model.losses
            )
