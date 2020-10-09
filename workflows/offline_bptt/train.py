from typing import Iterable, Sequence
import os
import fv3fit
import fv3fit.keras._models.loss
import fv3fit.keras._models.normalizer
import loaders
import numpy as np
import tensorflow as tf
import random
import vcm
import fv3gfs.util
import argparse
import concurrent.futures

COARSE_DATA_DIR = (
    "gs://vcm-ml-experiments/2020-06-17-triad-round-1/nudging/nudging/outdir-3h"
)
REF_DATA_DIR = "gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts"
SAMPLE_DIM_NAME = "sample"
TIME_NAME = "time"


def stack(ds):
    stack_dims = [dim for dim in ds.dims if dim not in fv3gfs.util.Z_DIMS]
    if len(set(ds.dims).intersection(fv3gfs.util.Z_DIMS)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = vcm.safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=list(fv3gfs.util.Z_DIMS) + [TIME_NAME],
    )
    return ds_stacked.transpose()


def load_blocks(
    nudged_mapper,
    ref_mapper,
    X_packer,
    y_packer,
    block_dir,
    n_timesteps=None,
    n_timestep_start=None,
    n_timesteps_per_block=1,
    n_timestep_frequency=None,
    label="default",
    cache_dir=None,
) -> Iterable[str]:
    use_cache = cache_dir is not None
    shared_keys = sorted(list(set(nudged_mapper.keys()).union(ref_mapper.keys())))
    if n_timestep_frequency is not None:
        shared_keys = shared_keys[::n_timestep_frequency]
    if n_timestep_start is not None:
        shared_keys = shared_keys[n_timestep_start:]
    if n_timesteps is not None:
        shared_keys = shared_keys[:n_timesteps]
    if n_timestep_start is None:
        n_timestep_start = 0

    def pack(packer, ds):
        drop_names = set(ds.data_vars).union(ds.coords).difference(packer.pack_names)
        ds = ds.drop_vars(drop_names)
        return_value = packer.to_array(stack(ds))[:, None, :].astype(np.float32)
        return return_value

    if use_cache:
        data_cache_dir = os.path.join(cache_dir, f"cache-load-{label}")
        if not os.path.exists(data_cache_dir):
            os.mkdir(data_cache_dir)
    block_filename_list = []
    X_coarse_list = []
    y_coarse_list = []
    y_ref_list = []
    for i_add, timestamp in enumerate(shared_keys):
        i = n_timestep_start + i_add
        print(f"loading timestep {i + 1} of {n_timestep_start + len(shared_keys)}")
        block_index = (int(i / n_timesteps_per_block) + 1) * n_timesteps_per_block - 1
        block_filename = os.path.join(block_dir, f"block-{block_index:05d}.npz")
        if os.path.exists(block_filename):
            if len(block_filename_list) > 0 and (
                block_filename_list[-1] != block_filename
            ):
                block_filename_list.append(block_filename)
        else:
            filename = os.path.join(data_cache_dir, f"timestep_{timestamp}.npz")
            if use_cache and os.path.isfile(filename):
                with open(filename, "rb") as f:
                    data = np.load(f)
                    X_coarse, y_coarse, y_ref = (data[k] for k in data)
            else:
                ref_ds = ref_mapper[timestamp]
                y_ref = pack(y_packer, ref_ds)
                nudged_ds = nudged_mapper[timestamp]
                X_coarse = pack(X_packer, nudged_ds)
                y_coarse = pack(y_packer, nudged_ds)
                if use_cache:
                    with open(filename, "wb") as f:
                        np.savez(f, X_coarse, y_coarse, y_ref)
            X_coarse_list.append(X_coarse)
            y_coarse_list.append(y_coarse)
            y_ref_list.append(y_ref)
            if len(y_coarse_list) == n_timesteps_per_block:
                X_coarse = np.concatenate(X_coarse_list, axis=1)
                X_coarse_list = []
                y_coarse = np.concatenate(y_coarse_list, axis=1)
                y_coarse_list = []
                y_ref = np.concatenate(y_ref_list, axis=1)
                y_ref_list = []
                block_filename = os.path.join(block_dir, f"block-{i:05d}.npz")
                with open(block_filename, "wb") as f:
                    data = {
                        "X_coarse": X_coarse,
                        "y_coarse": y_coarse,
                        "y_ref": y_ref,
                    }
                    np.savez(f, **data)
                block_filename_list.append(block_filename)
    return block_filename_list


class WindowedDataIterator:
    def __init__(
        self,
        block_filenames: Sequence[str],
        n_blocks_window: int,
        n_blocks_between_window: int,
        batch_size: int,
    ):
        """
        Args:
            block_filenames: npz files containing blocks of data
            n_blocks_window: how many blocks in a window
            n_blocks_between_window: how many blocks between window starts
            batch_size: how many samples (randomly selected) to include in a window
        """
        self.block_filenames = block_filenames
        self.n_blocks_window = n_blocks_window
        self.n_blocks_between_window = n_blocks_between_window
        self.batch_size = batch_size
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        with open(block_filenames[0], "rb") as f:
            data = np.load(f)
            X_coarse = data["X_coarse"]
            if X_coarse.shape[0] < batch_size:
                raise ValueError(
                    "cannot have more samples in batch than in a block, "
                    f"got shape {X_coarse.shape} and batch_size {batch_size}"
                )
            self._data_batch_size = X_coarse.shape[0]
        window_indices = []
        for i_window in range(self._n_windows):
            window_indices.append(i_window)
        random.shuffle(window_indices)
        self._window_indices = window_indices
        self._batch_indices = None
        self._i_window = 0
        self._load_thread = None
        self._start_load()

    def _get_random_indices(self, n: int, i_max: int):
        batch_indices = np.arange(i_max)
        np.random.shuffle(batch_indices)
        return batch_indices[:n]

    @property
    def _n_windows(self):
        return (
            int(
                (len(self.block_filenames) - self.n_blocks_window)
                / self.n_blocks_between_window
            )
            + 1
        )

    def _start_load(self):
        if self._i_window < self._n_windows:
            batch_idx = self._get_random_indices(self.batch_size, self._data_batch_size)
            window_start = self._window_indices[self._i_window]
            window_filenames = self.block_filenames[
                window_start : window_start + self.n_blocks_window
            ]
            self._load_thread = self._executor.submit(
                load_window, batch_idx, window_filenames,
            )

    def __next__(self):
        if self._i_window >= self._n_windows:
            raise StopIteration()
        else:
            X_coarse, y_coarse, y_ref = self._load_thread.result()
            self._load_thread = None
            self._i_window += 1
            if self._i_window < self._n_windows:
                self._start_load()
            return X_coarse, y_coarse, y_ref

    def __iter__(self):
        self._i_window = 0
        if self._load_thread is None:
            self._start_load()
        return self

    def __len__(self):
        return self._n_windows


def load_window(batch_idx, window_filenames):
    X_coarse_list = []
    y_coarse_list = []
    y_ref_list = []
    for filename in window_filenames:
        with open(filename, "rb") as f:
            data = np.load(f)
            X_coarse_list.append(data["X_coarse"][batch_idx, :])
            y_coarse_list.append(data["y_coarse"][batch_idx, :])
            y_ref_list.append(data["y_ref"][batch_idx, :])
    X_coarse = np.concatenate(X_coarse_list, axis=1)
    y_coarse = np.concatenate(y_coarse_list, axis=1)
    y_ref = np.concatenate(y_ref_list, axis=1)
    return X_coarse, y_coarse, y_ref


def build_model(
    n_batch,
    n_input,
    n_state,
    n_window,
    units,
    n_hidden_layers,
    X_scaler,
    y_scaler,
    tendency_ratio,
):
    forcing_input = tf.keras.layers.Input(batch_shape=[n_batch, n_window, n_input])
    state_input = tf.keras.layers.Input(batch_shape=[n_batch, n_window, n_state])
    x_forcing = X_scaler.normalize_layer(forcing_input)
    x_state = y_scaler.normalize_layer(state_input)
    rnn = tf.keras.layers.RNN(
        fv3fit.keras.GCMCell(
            units=units,
            n_input=n_input,
            n_state=n_state,
            n_hidden_layers=n_hidden_layers,
            tendency_ratio=tendency_ratio,
        ),
        name="rnn",
        return_sequences=True,
    )
    x = rnn(inputs=(x_forcing, x_state))
    outputs = y_scaler.denormalize_layer(x)
    model = tf.keras.Model(inputs=[forcing_input, state_input], outputs=outputs)
    return model


def penalize_negative_water(loss, negative_water_weight):
    def custom_loss(y_true, y_pred):
        # we can assume temperature will never be even close to zero
        # TODO this assumes temperature + humidity are the prognostic outputs,
        # better would be to get the specific indices corresponding to humidity
        negative_water = tf.math.multiply(
            tf.constant(negative_water_weight, dtype=tf.float32),
            tf.math.reduce_mean(
                tf.where(
                    y_pred < tf.constant(0.0, dtype=tf.float32),
                    tf.math.multiply(tf.constant(-1.0, dtype=tf.float32), y_pred),
                    tf.constant(0.0, dtype=tf.float32),
                )
            ),
        )
        base_loss = tf.cast(loss(y_true, y_pred), tf.float32)
        return tf.math.add(negative_water, base_loss)

    return custom_loss


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("label", action="store", type=str)
    parser.add_argument("--cache-dir", action="store", type=str, default=".")
    args = parser.parse_args()
    cache_filename = os.path.join(args.cache_dir, f"cache-{args.label}.npz")
    cache_filename_X_packer = os.path.join(args.cache_dir, f"X_packer-{args.label}.yml")
    cache_filename_y_packer = os.path.join(args.cache_dir, f"y_packer-{args.label}.yml")
    n_timestep_frequency = 4
    n_block = 12  # number of times in a npz file
    data_fraction = 0.125  # fraction of data to use from a window
    blocks_per_day = 24 // n_block
    n_blocks_train = 30 * blocks_per_day
    n_blocks_val = 5 * blocks_per_day
    n_blocks_window = 5 * blocks_per_day
    n_blocks_between_window = int(1.5 * blocks_per_day)
    n_timesteps = n_blocks_train * n_block
    n_timesteps_val = n_blocks_val * n_block
    n_window = n_blocks_window * n_block
    units = 128
    n_hidden_layers = 3
    tendency_ratio = 1e-3
    batch_size = 48
    input_variables = [
        "surface_geopotential",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "land_sea_mask",
        "total_precipitation",
        "latent_heat_flux",
        "sensible_heat_flux",
    ]
    prognostic_variables = [
        "air_temperature",
        "specific_humidity",
    ]
    if os.path.isfile(cache_filename_X_packer):
        with open(cache_filename_X_packer, "r") as f:
            X_packer = fv3fit.ArrayPacker.load(f)
        with open(cache_filename_y_packer, "r") as f:
            y_packer = fv3fit.ArrayPacker.load(f)
    else:
        X_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, input_variables)
        y_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, prognostic_variables)

    block_dir = f"block-{args.label}"
    if not os.path.exists(block_dir):
        os.mkdir(block_dir)
    elif len(os.listdir(block_dir)) >= (n_blocks_train + n_blocks_val):
        all_filenames = [
            os.path.join(block_dir, filename) for filename in os.listdir(block_dir)
        ]
        train_block_filenames = all_filenames[:n_blocks_train]
        val_block_filenames = all_filenames[
            n_blocks_train : n_blocks_train + n_blocks_val
        ]
    else:
        nudged_mapper = loaders.mappers.open_merged_nudged(
            COARSE_DATA_DIR,
            merge_files=("before_dynamics.zarr", "nudging_tendencies.zarr"),
        )
        ref_mapper = loaders.mappers.CoarsenedDataMapper(REF_DATA_DIR)
        train_block_filenames = load_blocks(
            nudged_mapper,
            ref_mapper,
            X_packer,
            y_packer,
            block_dir,
            n_timesteps=n_timesteps,
            n_timestep_frequency=n_timestep_frequency,
            n_timesteps_per_block=n_block,
            label=args.label,
            cache_dir=args.cache_dir,
        )
        val_block_filenames = load_blocks(
            nudged_mapper,
            ref_mapper,
            X_packer,
            y_packer,
            block_dir,
            n_timesteps=n_timesteps_val,
            n_timestep_start=n_timesteps,
            n_timestep_frequency=n_timestep_frequency,
            n_timesteps_per_block=n_block,
            label=args.label,
            cache_dir=args.cache_dir,
        )
    if len(X_packer.feature_counts) == 0:
        raise RuntimeError(
            "need to process at least one file to store feature counts, "
            "please delete a file from the cache"
        )
    if not os.path.isfile(cache_filename_X_packer):
        with open(cache_filename_X_packer, "w") as f:
            X_packer.dump(f)
        with open(cache_filename_y_packer, "w") as f:
            y_packer.dump(f)
    with open(train_block_filenames[0], "rb") as f:
        data = np.load(f)
        X_coarse = data["X_coarse"]
        y_ref = data["y_ref"]
    X_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler()
    X_scaler.fit(X_coarse)
    y_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler()
    y_scaler.fit(y_ref)
    X_train = WindowedDataIterator(
        train_block_filenames,
        n_blocks_window=n_blocks_window,
        n_blocks_between_window=n_blocks_between_window,
        batch_size=int(48 * 48 * 6 * data_fraction),
    )
    # X_val = WindowedDataIterator(
    #     val_block_filenames,
    #     n_blocks_window=n_blocks_window,
    #     n_blocks_between_window=n_blocks_between_window,
    #     batch_size=int(48 * 48 * 6 * data_fraction),
    # )
    loss = fv3fit.keras._models.loss.get_weighted_mse(y_packer, y_scaler.std)
    loss = penalize_negative_water(loss, 1.0)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model_dir = f"models-{args.label}"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_filenames = os.listdir(model_dir)
    base_epoch = 0
    if len(model_filenames) > 0:
        last_model_filename = os.path.join(model_dir, sorted(model_filenames)[-1])
        model = tf.keras.models.load_model(
            last_model_filename,
            custom_objects={
                "custom_loss": loss,
                "GCMCell": fv3fit.keras._models.gcm_cell.GCMCell,
            },
        )
        base_epoch = int(last_model_filename[-6:-3]) + 1
        print(f"loaded model, resuming at epoch {base_epoch}")
    else:
        model = build_model(
            batch_size,
            X_coarse.shape[-1],
            y_ref.shape[-1],
            n_window=n_window,
            units=units,
            n_hidden_layers=n_hidden_layers,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
            tendency_ratio=tendency_ratio,
        )
    model.compile(
        optimizer=optimizer, loss=loss,
    )

    batch_idx = np.random.choice(
        np.arange(X_coarse.shape[0]), size=int(0.125 * X_coarse.shape[0]), replace=False
    )
    X_coarse_val, y_coarse_val, y_ref_val = load_window(batch_idx, val_block_filenames)
    coarse_loss_val = loss(y_ref_val, y_coarse_val)

    for i_epoch in range(50):
        print(f"starting epoch {i_epoch}")
        for X_coarse, y_coarse, y_ref in X_train:
            model.fit(
                x=[X_coarse, y_coarse],
                y=y_ref,
                batch_size=batch_size,
                epochs=1,
                shuffle=True,
            )
            del X_coarse
            del y_coarse
            del y_ref
        # coarse_losses = []
        # val_losses = []
        # for X_coarse, y_coarse, y_ref in X_val:
        #     print("doin a block")
        #     coarse_losses.append(loss(y_ref, y_coarse))
        #     y_pred = model.predict([X_coarse, y_coarse])
        #     val_losses.append(loss(y_ref, y_pred))

        print(f"coarse loss: {coarse_loss_val}")
        val_loss = loss(y_ref_val, model.predict([X_coarse_val, y_coarse_val]))
        print(f"val loss: {val_loss}")

        epoch = base_epoch + i_epoch
        print(f"saving model for epoch {epoch}")
        model.save(
            os.path.join(
                model_dir, f"model-{args.label}-{1e5*val_loss:.2f}-{epoch:03d}.tf"
            )
        )

    export_model = fv3fit.keras.RecurrentModel(
        SAMPLE_DIM_NAME,
        input_variables,
        prognostic_variables,
        model=model,
        X_packer=X_packer,
        y_packer=y_packer,
    )
    export_model.dump(f"fv3fit-model-{args.label}")
