import matplotlib.pyplot as plt
import argparse
import numpy as np
import fv3fit
import vcm
from preprocessing import TrainingArrays
from train import prepare_keras_arrays
import xarray as xr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "model_dir", type=str, help="directory containing trained model"
    )
    return parser


def get_vmin_vmax(*arrays):
    vmin = min(np.min(a) for a in arrays)
    vmax = max(np.max(a) for a in arrays)
    return vmin, vmax


SAMPLE_DIM_NAME = "column"

def arrays_to_dataset(arrays):
    inputs = arrays.inputs_baseline
    prognostic = arrays.prognostic_baseline
    nudging = arrays.nudging_tendency


    data_vars = {}
    for i, name in enumerate(arrays.input_names):
        data_vars[name] = xr.DataArray(inputs[:, :, i], dims=[SAMPLE_DIM_NAME, "time"])
    data_vars["air_temperature"] = xr.DataArray(prognostic[:, :, :79], dims=[SAMPLE_DIM_NAME, "time", "z"])
    data_vars["specific_humidity"] = xr.DataArray(prognostic[:, :, 79:], dims=[SAMPLE_DIM_NAME, "time", "z"])
    data_vars["pQ1"] = xr.DataArray(arrays.given_tendency[:, :, :79], dims=[SAMPLE_DIM_NAME, "time", "z"])
    data_vars["pQ2"] = xr.DataArray(arrays.given_tendency[:, :, 79:], dims=[SAMPLE_DIM_NAME, "time", "z"])
    data_vars["nQ1"] = xr.DataArray(nudging[:, :, :79], dims=[SAMPLE_DIM_NAME, "time", "z"])
    data_vars["nQ2"] = xr.DataArray(nudging[:, :, 79:], dims=[SAMPLE_DIM_NAME, "time", "z"])
    return xr.Dataset(data_vars)


if __name__ == "__main__":
    timestep_seconds = 3 * 60 * 60
    parser = get_parser()
    args = parser.parse_args()

    model = fv3fit.load(args.model_dir)

    fs = vcm.get_fs(args.arrays_dir)
    last_filename = sorted(fs.listdir(args.arrays_dir, detail=False))[-1]
    with open(last_filename, "rb") as f:
        arrays = TrainingArrays.load(f)

    ds = arrays_to_dataset(arrays)

    state_ds = ds.isel(time=1)
    state_out_list = []
    n_timesteps = len(ds["time"])
    for i in range(n_timesteps):
        print(f"Step {i+1} of {n_timesteps}")
        forcing_ds = ds.isel(time=i)
        state_ds = state_ds.assign({
            name: forcing_ds[name] for name in arrays.input_names
        })
        tendency_ds = model.predict_columnwise(state_ds, sample_dims=(SAMPLE_DIM_NAME,))
        assert not np.any(np.isnan(state_ds["air_temperature"].values))
        state_ds["air_temperature"] += (forcing_ds["pQ1"] + tendency_ds["dQ1"]) * timestep_seconds
        state_ds["specific_humidity"] += (forcing_ds["pQ2"] + tendency_ds["dQ2"]) * timestep_seconds
        state_ds = state_ds.assign({
            "nQ1": forcing_ds["nQ1"],
            "nQ2": forcing_ds["nQ1"],
            "air_temperature_reference": forcing_ds["air_temperature"],
            "specific_humidity_reference": forcing_ds["specific_humidity"],
            **tendency_ds.data_vars
        })
        state_out_list.append(state_ds)
    state_out = xr.concat(state_out_list, dim="time")
    print(state_out)

    def plot_single(predicted, reference, label, ax):
        vmin, vmax = get_vmin_vmax(predicted, reference)
        im = ax[0].pcolormesh(predicted.T, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(f"predicted {label}")
        im = ax[1].pcolormesh(reference.T, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"reference {label}")
        plt.colorbar(im, ax=ax[1])

    i = 0
    seconds_in_day = 60 * 60 * 24
    fig, ax = plt.subplots(4, 1, figsize=(12, 8))

    plot_single(
        state_out["dQ1"][:, i, :].values * seconds_in_day,
        state_out["nQ1"][:, i, :].values * seconds_in_day,
        "air_temperature (K/day)",
        ax[:2],
    )
    plot_single(
        state_out["air_temperature"][:, i, :].values,
        state_out["air_temperature_reference"][:, i, :].values,
        "air_temperature (K)",
        ax[2:],
    )
    lat = arrays.latitude[i] * 180.0 / np.pi
    lon = arrays.longitude[i] * 180.0 / np.pi
    plt.suptitle(f"lat: {lat}, lon: {lon}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plot_single(
    #     state_out["dQ2"][:, i, :].values * seconds_in_day,
    #     state_out["nQ2"][:, i, :].values * seconds_in_day,
    #     "specific_humidity (kg/kg/day)",
    #     ax[:2],
    # )
    # plot_single(
    #     state_out["specific_humidity"][:, i, :].values,
    #     state_out["specific_humidity_reference"][:, i, :].values,
    #     "specific_humidity (kg/kg)",
    #     ax[2:],
    # )
    lat = arrays.latitude[i] * 180.0 / np.pi
    lon = arrays.longitude[i] * 180.0 / np.pi
    plt.suptitle(f"lat: {lat}, lon: {lon}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    assert False, state_out

    val_inputs, val_given_tendency, val_target_out = prepare_keras_arrays(
        arrays,
        model.input_scaler,
        model.tendency_scaler,
        model.prognostic_scaler,
        timestep_seconds,
    )
    val_base_out = timestep_seconds * np.cumsum(val_given_tendency, axis=1)

    val_out = np.zeros_like(val_base_out)
    val_out[:, 0, :] = val_base_out[:, 0, :]
    profile_out = np.zeros_like(val_out)
    baseline_out = np.zeros_like(val_out)
    tendency_out = np.zeros_like(val_out)
    profile_out[:, 0, :] = model.prognostic_scaler.denormalize(val_out[:, 0, :])
    baseline_out[:, 0, :] = profile_out[:, 0, :]
    for i in range(1, val_base_out.shape[1]):
        baseline_out[:, i, :] = (
            baseline_out[:, i - 1, :] + timestep_seconds * val_given_tendency[:, i, :]
        )
        new_state = model.model.predict([val_inputs[:, i, :], val_out[:, i - 1, :]])
        tendency_out[:, i, :] = (
            model.tendency_scaler.denormalize(new_state - val_out[:, i - 1, :])
            / timestep_seconds
        )
        val_out[:, i, :] = new_state + timestep_seconds * val_given_tendency[:, i, :]
        val_out[:, i, :] = new_state
        profile_out[:, i, :] = model.prognostic_scaler.denormalize(val_out[:, i, :])

    nz = 79

    def plot_single(predicted, reference, label, ax):
        vmin, vmax = get_vmin_vmax(predicted, reference)
        im = ax[0].pcolormesh(predicted.T, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(f"predicted {label}")
        im = ax[1].pcolormesh(reference.T, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"reference {label}")
        plt.colorbar(im, ax=ax[1])

    antarctica = (
        (arrays.latitude <= -70 * np.pi / 180.0)
        & (arrays.longitude > 15 * np.pi / 180.0)
        & (arrays.longitude < 150 * np.pi / 180.0)
    )
    seconds_in_day = 60 * 60 * 24
    print("antarctica sphum biases")
    print(
        tendency_out[antarctica, 1, :nz].mean(),
        arrays.nudging_tendency[antarctica, 1, :nz].mean(),
    )
    print(
        tendency_out[antarctica, 1, nz - 10 : nz].mean(),
        arrays.nudging_tendency[antarctica, 1, nz - 10 : nz].mean(),
    )
    print(
        tendency_out[antarctica, 1, nz - 1].mean(),
        arrays.nudging_tendency[antarctica, 1, nz - 1].mean(),
    )
    # for i in np.argwhere(antarctica).flatten()[:3]:
    for i in np.random.randint(0, tendency_out.shape[0], size=3):
        fig, ax = plt.subplots(4, 1, figsize=(12, 8))
        # first time is "setup" (ML tendency is 0, model creates the initial state)
        # first nudging tendency is included in ML initial state because of this
        # so compare from second timestep onwards
        print(i)
        print(tendency_out[i, 1:, :nz].shape)
        plot_single(
            tendency_out[i, 1:, :nz] * seconds_in_day,
            arrays.nudging_tendency[i, 1:, :nz] * seconds_in_day,
            "air_temperature (K/day)",
            ax[:2],
        )
        plot_single(
            tendency_out[i, 1:, nz:] * seconds_in_day,
            arrays.nudging_tendency[i, 1:, nz:] * seconds_in_day,
            "specific_humidity (kg/kg/day)",
            ax[2:],
        )
        lat = arrays.latitude[i] * 180.0 / np.pi
        lon = arrays.longitude[i] * 180.0 / np.pi
        plt.suptitle(f"lat: {lat}, lon: {lon}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig, ax = plt.subplots(4, 1, figsize=(12, 8))
        plot_single(
            profile_out[i, 1:, :nz] - model.prognostic_scaler.mean[:nz],
            arrays.prognostic_reference[i, 1:, :nz] - model.prognostic_scaler.mean[:nz],
            "air_temperature minus mean (K)",
            ax[:2],
        )
        plot_single(
            profile_out[i, 1:, nz:] - model.prognostic_scaler.mean[nz:],
            arrays.prognostic_reference[i, 1:, nz:] - model.prognostic_scaler.mean[nz:],
            "specific_humidity (kg/kg)",
            ax[2:],
        )
        plt.suptitle(f"lat: {lat}, lon: {lon}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    print("all times, profiles:")
    r2_air_temperature = 1 - np.std(
        arrays.prognostic_reference[:, 1:, :nz] - profile_out[:, 1:, :nz]
    , axis=(0, 1)) / np.std(arrays.prognostic_reference[:, 1:, :nz])
    r2_specific_humidity = 1 - np.std(
        arrays.prognostic_reference[:, 1:, nz:] - profile_out[:, 1:, nz:]
    , axis=(0, 1)) / np.std(arrays.prognostic_reference[:, 1:, nz:], axis=(0, 1))
    print(f"r2 air_temperature: {r2_air_temperature}")
    print(f"r2 specific_humidity: {r2_specific_humidity}")

    print("all times:")
    r2_air_temperature = 1 - np.std(
        tendency_out[:, 1:, :nz] - arrays.nudging_tendency[:, 1:, :nz]
    ) / np.std(arrays.nudging_tendency[:, 1:, :nz])
    r2_specific_humidity = 1 - np.std(
        tendency_out[:, 1:, nz:] - arrays.nudging_tendency[:, 1:, nz:]
    ) / np.std(arrays.nudging_tendency[:, 1:, nz:])
    print(f"r2 air_temperature: {r2_air_temperature}")
    print(f"r2 specific_humidity: {r2_specific_humidity}")

    print("all times normalized:")
    norm_tendency_out = model.tendency_scaler.normalize(tendency_out)
    norm_nudging_tendency = model.tendency_scaler.normalize(arrays.nudging_tendency)
    r2_air_temperature = 1 - np.std(
        norm_tendency_out[:, 1:, :nz] - norm_nudging_tendency[:, 1:, :nz]
    ) / np.std(norm_nudging_tendency[:, 1:, :nz])
    r2_specific_humidity = 1 - np.std(
        norm_tendency_out[:, 1:, nz:] - norm_nudging_tendency[:, 1:, nz:]
    ) / np.std(norm_nudging_tendency[:, 1:, nz:])
    print(f"r2 air_temperature: {r2_air_temperature}")
    print(f"r2 specific_humidity: {r2_specific_humidity}")

    print("first timestep only:")
    r2_air_temperature = 1 - np.std(
        tendency_out[:, 1, :nz] - arrays.nudging_tendency[:, 1, :nz]
    ) / np.std(arrays.nudging_tendency[:, 1, :nz])
    r2_specific_humidity = 1 - np.std(
        tendency_out[:, 1, nz:] - arrays.nudging_tendency[:, 1, nz:]
    ) / np.std(arrays.nudging_tendency[:, 1, nz:])
    print(f"r2 air_temperature: {r2_air_temperature}")
    print(f"r2 specific_humidity: {r2_specific_humidity}")

    print("first timestep lowest 20 levels only:")
    r2_air_temperature = 1 - np.std(
        tendency_out[:, 1, nz - 20 : nz] - arrays.nudging_tendency[:, 1, nz - 20 : nz]
    ) / np.std(arrays.nudging_tendency[:, 1, nz - 20 : nz])
    r2_specific_humidity = 1 - np.std(
        tendency_out[:, 1, -20:] - arrays.nudging_tendency[:, 1, -20:]
    ) / np.std(arrays.nudging_tendency[:, 1, -20:])
    print(f"r2 air_temperature: {r2_air_temperature}")
    print(f"r2 specific_humidity: {r2_specific_humidity}")

    # plt.figure()
    # print(profile_out[0, :, :79].min(), profile_out[0, :, :79].max())
    # print(profile_out[0, :, 79:].min(), profile_out[0, :, 79:].max())
    # im = plt.pcolormesh(profile_out[0, :, :].T)
    # plt.colorbar(im)
    # plt.show()

    # val_initial_state = val_given_tendency[:, 0, :] * timestep_seconds
    # val_first_output = loaded.model.predict(
    #     [val_inputs[:, 0, :], val_given_tendency[:, 0, :] * timestep_seconds]
    # )
    # val_first_tendency = val_first_output - val_initial_state
    # target_first_tendency = tendency_scaler.normalize(
    #     arrays.nudging_tendency
    # )[:, 0, :]
    # first_tendency_loss = np.mean(
    #     loss(val_first_tendency, target_first_tendency * timestep_seconds)
    # )
    # print(f"first tendency loss: {first_tendency_loss}")
    # print(f"first tendency std: {np.std(target_first_tendency * timestep_seconds)}")

    # n_samples = val_inputs.shape[0] * val_inputs.shape[1]
    # ds_target = prognostic_packer.to_dataset(
    #     prognostic_scaler.denormalize(
    #         val_target_out.reshape([n_samples, val_target_out.shape[2]])
    #     )
    # )
    # ds_norm_target = prognostic_packer.to_dataset(
    #     val_target_out.reshape([n_samples, val_target_out.shape[2]])
    # )
    # ds_input = input_packer.to_dataset(
    #     input_scaler.denormalize(val_inputs.reshape([n_samples, val_inputs.shape[2]]))
    # )
    # ds_output = prognostic_packer.to_dataset(
    #     prognostic_scaler.denormalize(
    #         val_out.reshape([n_samples, profile_out.shape[2]])
    #     )
    # )
    # ds_norm_output = prognostic_packer.to_dataset(
    #     val_out.reshape([n_samples, profile_out.shape[2]])
    # )
    # ds_baseline_out = prognostic_packer.to_dataset(
    #     prognostic_scaler.denormalize(
    #         baseline_out.reshape([n_samples, profile_out.shape[2]])
    #     )
    # )
    # ds_norm_baseline_out = prognostic_packer.to_dataset(
    #     baseline_out.reshape([n_samples, profile_out.shape[2]])
    # )
    # ds_target.to_netcdf("ds_target.nc")
    # ds_norm_target.to_netcdf("ds_norm_target.nc")
    # ds_input.to_netcdf("ds_input.nc")
    # ds_output.to_netcdf("ds_output.nc")
    # ds_norm_output.to_netcdf("ds_norm_output.nc")
    # ds_baseline_out.to_netcdf("ds_baseline_out.nc")
    # ds_norm_baseline_out.to_netcdf("ds_norm_baseline_out.nc")

    # model_out = np.zeros_like(val_target_out)
    # model_offline_out = np.zeros_like(val_target_out)
    # nudging_tendency = tendency_scaler.normalize(validation_arrays.nudging_tendency)
    # before_nudging_state = (
    #     val_target_out[:, :, :] - timestep_seconds * nudging_tendency[:, :, :]
    # )
    # for i in range(val_out.shape[1]):
    #     model_offline_out[:, i, :] = loaded.model.predict(
    #         [val_inputs[:, i, :], val_target_out[:, i, :]]
    #     )
    #     model_out[:, i, :] = loaded.model.predict(
    #         [val_inputs[:, i, :], before_nudging_state[:, i, :]]
    #     )
    # model_tendency = tendency_scaler.denormalize(
    #     (model_out - before_nudging_state) / timestep_seconds
    # )
    # model_offline_tendency = tendency_scaler.denormalize(
    #     (model_offline_out - val_target_out) / timestep_seconds
    # )
    # target_tendency = validation_arrays.nudging_tendency
    # prognostic_packer.to_dataset(
    #     model_tendency.reshape([n_samples, 2 * nz])
    # ).to_netcdf(
    #     "model_tendency.nc"
    # )
    # prognostic_packer.to_dataset(
    #     model_offline_tendency.reshape([n_samples, 2 * nz])
    # ).to_netcdf("model_offline_tendency.nc")
    # prognostic_packer.to_dataset(
    #     target_tendency.reshape([n_samples, 2 * nz])
    # ).to_netcdf("target_tendency.nc")

    # i = 0
    # for i in np.random.randint(0, high=val_base_out.shape[0], size=5):
    #     fig, ax = plt.subplots(3, 1)
    #     vmin = min(val_base_out[i, :, :].min(), val_target_out[i, :, :].min())
    #     vmax = max(val_base_out[i, :, :].max(), val_target_out[i, :, :].max())
    #     im = ax[0].pcolormesh(val_out[i, :, :].T, vmin=vmin, vmax=vmax)
    #     plt.colorbar(im, ax=ax[0])
    #     im = ax[1].pcolormesh(val_target_out[i, :, :].T, vmin=vmin, vmax=vmax)
    #     plt.colorbar(im, ax=ax[1])
    #     im = ax[2].pcolormesh(val_base_out[i, :, :].T, vmin=vmin, vmax=vmax)
    #     plt.colorbar(im, ax=ax[2])
    #     plt.show()
    # print(val_target_out.std(axis=(0, 1)))

    # assert False
