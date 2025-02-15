import hashlib
import os
import glob
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from subprocess import CalledProcessError

import pytest
from google.api_core.exceptions import NotFound
from google.cloud.storage import Blob
from vcm import extract
from vcm.cloud import gcs

TEST_DIR = Path(os.path.abspath(__file__)).parent


@pytest.fixture(scope="function")
def tmpdir():
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield temporary_dir


def _compare_checksums(file_path1: Path, file_path2: Path) -> None:

    with open(file_path1, "rb") as file1:
        with open(file_path2, "rb") as file2:
            file1 = file1.read()
            file2 = file2.read()
            downloaded_checksum = hashlib.md5(file1).hexdigest()
            local_checksum = hashlib.md5(file2).hexdigest()
            assert downloaded_checksum == local_checksum


def test_init_blob_is_blob():
    result = gcs.init_blob("test_bucket", "test_blobdir/test_blob.nc")
    assert isinstance(result, Blob)


def test_init_blob_bucket_and_blob_name():
    result = gcs.init_blob("test_bucket", "test_blobdir/test_blob.nc")
    assert result.bucket.name == "test_bucket"
    assert result.name == "test_blobdir/test_blob.nc"


def test_init_blob_from_gcs_url():
    result = gcs.init_blob_from_gcs_url("gs://test_bucket/test_blobdir/test_blob.nc")
    assert isinstance(result, Blob)
    assert result.bucket.name == "test_bucket"
    assert result.name == "test_blobdir/test_blob.nc"


@pytest.mark.parametrize(
    "gcs_url",
    [
        "gs://vcm-ml-code-testing-data/cloud_gcs/test_datafile.txt",
        "gs://vcm-ml-code-testing-data/cloud_gcs/test_data_array.nc",
    ],
)
def test_files_exist_on_gcs(gcs_url):
    blob = gcs.init_blob_from_gcs_url(gcs_url)
    assert blob.exists()


@pytest.mark.regression
def test_download_blob_to_file(tmpdir):
    txt_filename = "test_datafile.txt"
    gcs_path = "gs://vcm-ml-code-testing-data/cloud_gcs/"

    blob = gcs.init_blob_from_gcs_url(gcs_path + txt_filename)
    outfile_path = gcs.download_blob_to_file(blob, tmpdir, txt_filename)

    assert outfile_path.exists()


@pytest.mark.regression
def test_download_blob_to_file_makes_destination_directories(tmpdir):
    txt_filename = "test_datafile.txt"
    gcs_path = "gs://vcm-ml-code-testing-data/cloud_gcs/"
    nonexistent_path = Path("does/not/exist")

    blob = gcs.init_blob_from_gcs_url(gcs_path + txt_filename)

    non_existent_dir = Path(tmpdir, nonexistent_path)
    assert not non_existent_dir.exists()

    gcs.download_blob_to_file(blob, non_existent_dir, txt_filename)
    assert non_existent_dir.exists()


@pytest.mark.regression
def test_download_glob_to_file_nonexistent_blob(tmpdir):
    nonexistent_gcs_path = (
        "gs://vcm-ml-code-testing-data/non_existent_dir/non_existent_file.lol"
    )
    blob = gcs.init_blob_from_gcs_url(nonexistent_gcs_path)

    with pytest.raises(NotFound):
        gcs.download_blob_to_file(blob, tmpdir, "non_existsent.file")


def test_extract_tarball_default_dir(tmpdir):

    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    working_path = Path(tmpdir, tar_filename)

    tarball_extracted_path = extract.extract_tarball_to_path(working_path)
    assert tarball_extracted_path.exists()
    assert tarball_extracted_path.name == "test_data"


def test_extract_tarball_specified_dir(tmpdir):

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)
    target_output_dirname = "specified"

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    target_path = Path(tmpdir, target_output_dirname)

    tarball_extracted_path = extract.extract_tarball_to_path(
        test_tarball_path, extract_to_dir=target_path
    )
    assert tarball_extracted_path.exists()
    assert tarball_extracted_path.name == target_output_dirname


def test_extract_tarball_check_files_exist(tmpdir):

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    working_path = Path(tmpdir, tar_filename)
    tarball_extracted_path = extract.extract_tarball_to_path(working_path)

    test_data_files = ["test_data_array.nc", "test_datafile.txt"]
    for current_file in test_data_files:
        assert tarball_extracted_path.joinpath(current_file).exists()


def test_extract_tarball_non_existent_tar(tmpdir):
    non_existent_tar = Path(tmpdir, "nonexistent/tarfile.tar")
    with pytest.raises(CalledProcessError):
        extract.extract_tarball_to_path(non_existent_tar)


@pytest.mark.regression
def test_upload_dir_to_gcs(tmpdir):
    src_dir_to_upload = Path(__file__).parent.joinpath("test_data")
    gcs.upload_dir_to_gcs("vcm-ml-code-testing-data", "test_upload", src_dir_to_upload)

    test_files = ["test_datafile.txt", "test_data.tar"]

    for filename in test_files:
        gcs_url = f"gs://vcm-ml-code-testing-data/test_upload/{filename}"
        file_blob = gcs.init_blob_from_gcs_url(gcs_url)
        assert file_blob.exists()

        downloaded_path = gcs.download_blob_to_file(
            file_blob, Path(tmpdir, "test_uploaded"), filename
        )
        local_file = src_dir_to_upload.joinpath(filename)
        _compare_checksums(local_file, downloaded_path)
        file_blob.delete()


@pytest.mark.regression
def test_upload_dir_to_gcs_from_nonexistent_dir(tmpdir):

    nonexistent_dir = Path(tmpdir, "non/existent/dir/")
    with pytest.raises(FileNotFoundError):
        gcs.upload_dir_to_gcs(
            "vcm-ml-code-testing-data", "test_upload", nonexistent_dir
        )


@pytest.mark.regression
def test_upload_dir_to_gcs_dir_is_file():

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            gcs.upload_dir_to_gcs(
                "vcm-ml-code-testing-data", "test_upload", Path(f.name)
            )


@pytest.mark.regression
def test_upload_dir_to_gcs_does_not_upload_subdir(tmpdir):

    x = (1, 2, 3, 4)
    with open(Path(tmpdir, "what_a_pickle.pkl"), "wb") as f:
        pickle.dump(x, f)

    extra_subdir = Path(tmpdir, "extra_dir")
    extra_subdir.mkdir()

    with open(Path(extra_subdir, "extra_pickle.pkl"), "wb") as f:
        pickle.dump(x, f)

    # TODO: use pytest fixture to do setup/teardown of temporary gcs dir

    upload_dir = "transient"
    bucket_name = "vcm-ml-code-testing-data"
    gcs_url_prefix = f"gs://{bucket_name}"
    tmp_gcs_dir = f"test_upload/{upload_dir}"
    tmp_gcs_url = f"{gcs_url_prefix}/{tmp_gcs_dir}"

    gcs.upload_dir_to_gcs(bucket_name, tmp_gcs_dir, Path(tmpdir))

    uploaded_pickle_url = f"{tmp_gcs_url}/what_a_pickle.pkl"
    not_uploaded_pickle_url = f"{tmp_gcs_url}/extra_dir/extra_pickle.pkl"

    # Sleeps added to reduce api request rate by circleci
    time.sleep(0.1)
    pkl_blob = gcs.init_blob_from_gcs_url(uploaded_pickle_url)
    nonexistent_pkl_blob = gcs.init_blob_from_gcs_url(not_uploaded_pickle_url)

    time.sleep(0.1)
    assert pkl_blob.exists()
    time.sleep(0.1)
    pkl_blob.delete()

    assert not nonexistent_pkl_blob.exists()


@pytest.mark.regression
@pytest.mark.parametrize("include_parent", [True, False])
def test_download_all_bucket_files(include_parent):

    bucket_files = ["test_data_array.nc", "test_datafile.txt"]

    gcs_url = "gs://vcm-ml-code-testing-data/cloud_gcs/"

    with tempfile.TemporaryDirectory() as tmpdir:
        gcs.download_all_bucket_files(
            gcs_url, tmpdir, include_parent_in_stem=include_parent
        )

        if include_parent:
            local_dir = os.path.join(tmpdir, "cloud_gcs")
        else:
            local_dir = tmpdir

        for filename in bucket_files:
            local_filepath = os.path.join(local_dir, filename)
            assert os.path.exists(local_filepath)

        local_files = glob.glob(os.path.join(local_dir, "*"))
        assert len(local_files) == len(bucket_files)


@pytest.mark.regression
def test_download_all_bucket_files_nonexistant_gcs_url():

    with pytest.raises(ValueError):
        gcs.download_all_bucket_files(
            "gs://vcm-ml-code-testing-data/non-existent-bucket/doesnt/exist",
            "/tmp/mytemp",
        )
