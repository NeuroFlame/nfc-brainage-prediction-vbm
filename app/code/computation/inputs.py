"""Load site-local feature matrices and ages.

The computation is format-agnostic: ``input_source`` selects VBM NIfTI volumes
(the default for this repository) or the FNC formats it was derived from.
"""

import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .types import SiteInputs

DEFAULT_GICA_DATA_FILE = "coinstac-gica_postprocess_results.mat"
UKBIOBANK_SUBJECT_ID = "eid"
UKBIOBANK_AGE = "age_when_attended_assessment_centre_f21003_2_0"
VBM_SUBJECT_COLUMN = "niftifilename"
VBM_AGE_COLUMN = "age"


def load_inputs(
    data_dir,
    input_source="VBM",
    data_file=None,
    label_file="covariates.csv",
    vbm_downsample_factor=1,
    logger=None,
) -> SiteInputs:
    """Build the feature matrix and age vector for this site."""
    if input_source == "VBM":
        X, y = _load_vbm(data_dir, label_file, vbm_downsample_factor, logger)
    elif input_source == "GICA":
        X, y = _load_gica(data_dir, data_file or DEFAULT_GICA_DATA_FILE, label_file)
    elif input_source == "UKBioBank_Comp2019":
        X, y = _load_ukbiobank(data_dir, data_file, label_file)
    else:
        raise ValueError(
            f"Unknown input_source {input_source!r}; use 'VBM', 'GICA' or "
            "'UKBioBank_Comp2019'"
        )
    return SiteInputs(X=X, y=np.asarray(y, dtype=np.double).reshape(-1))


def read_vbm_image(data_dir, file_name, downsample_factor=1) -> np.ndarray:
    """Load one subject's VBM NIfTI volume as a 3D float32 array.

    The volume is e.g. an SPM ``swc1*.nii[.gz]`` smoothed, warped, modulated
    grey-matter density map. ``downsample_factor`` is the stride applied along
    each spatial axis to reduce the voxel count (2 keeps every other voxel); it
    must be the same for every subject and site so feature vectors line up.
    """
    img = nib.load(os.path.join(data_dir, file_name))
    data = img.get_fdata(dtype=np.float32)

    if downsample_factor and downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]

    return data


def upper_triangle_features(fnc_data: np.ndarray) -> np.ndarray:
    """Flatten each symmetric FNC matrix to its values above the diagonal."""
    rows, cols = np.triu_indices(fnc_data.shape[1], k=1)
    return fnc_data[:, rows, cols].astype(np.double)


def _load_vbm(data_dir, label_file, downsample_factor, logger):
    # Unlike the FNC formats (one file holding every subject), VBM data ships as
    # one NIfTI file per subject, named in the covariates CSV. Each volume is
    # flattened into a feature vector, stacked in covariate-row order.
    covariate_df = pd.read_csv(os.path.join(data_dir, label_file))
    covariate_df.columns = covariate_df.columns.str.strip()

    if (
        VBM_SUBJECT_COLUMN not in covariate_df.columns
        or VBM_AGE_COLUMN not in covariate_df.columns
    ):
        raise KeyError(
            f"Covariates file '{label_file}' must contain '{VBM_SUBJECT_COLUMN}' "
            f"and '{VBM_AGE_COLUMN}' columns; found {list(covariate_df.columns)}."
        )

    covariate_df[VBM_SUBJECT_COLUMN] = (
        covariate_df[VBM_SUBJECT_COLUMN].astype(str).str.strip()
    )

    n_rows = len(covariate_df)
    covariate_df = covariate_df.dropna(
        subset=[VBM_AGE_COLUMN, VBM_SUBJECT_COLUMN]
    ).reset_index(drop=True)
    if len(covariate_df) < n_rows and logger is not None:
        logger.warning(
            f"Dropped {n_rows - len(covariate_df)} row(s) from {label_file} with "
            f"missing '{VBM_SUBJECT_COLUMN}' or '{VBM_AGE_COLUMN}'."
        )

    features = []
    ref_shape = None
    for file_name in covariate_df[VBM_SUBJECT_COLUMN]:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"NIfTI file listed in {label_file} not found: {file_path}"
            )

        volume = read_vbm_image(data_dir, file_name, downsample_factor)
        if ref_shape is None:
            ref_shape = volume.shape
        elif volume.shape != ref_shape:
            raise ValueError(
                f"Volume shape mismatch for {file_name}: expected {ref_shape}, got "
                f"{volume.shape}. All subjects must share the same image grid."
            )
        features.append(volume.reshape(-1))

    X = np.vstack(features)
    y = covariate_df[VBM_AGE_COLUMN].to_numpy(dtype=np.float64)
    return X, y


def _load_gica(data_dir, data_file, label_file):
    # fnc_corrs_all is (subjects, sessions, components, components); use session 1.
    fnc_data = loadmat(os.path.join(data_dir, data_file))["fnc_corrs_all"][:, 0]
    covariate_df = pd.read_csv(os.path.join(data_dir, label_file))

    if len(fnc_data) != len(covariate_df):
        raise ValueError(
            f"Number of subjects in fnc_data ({len(fnc_data)}) and covariate "
            f"data ({len(covariate_df)}) do not match."
        )

    return upper_triangle_features(fnc_data), covariate_df["age"].to_numpy()


def _load_ukbiobank(data_dir, data_file, label_file):
    with h5py.File(os.path.join(data_dir, data_file), "r") as f:
        file_names = _cell_array_strings(f, "fN")[0]
        icn_ins = _matlab_array(f, "icn_ins")
        fnc_data = _matlab_array(f, "corrdata")

    eids = [int(name.split("/")[8].split("_")[0]) for name in file_names]
    df_eid = pd.DataFrame(eids, columns=[UKBIOBANK_SUBJECT_ID])
    df = pd.read_table(os.path.join(data_dir, label_file))
    df_eid_age = df[[UKBIOBANK_SUBJECT_ID, UKBIOBANK_AGE]]
    df_result = pd.concat([df_eid_age, df_eid], axis=1, join="inner").reindex(
        df_eid.index
    )

    # Keep only the intrinsic connectivity networks; MATLAB indices start at 1.
    icn = icn_ins.astype(int).reshape(len(icn_ins)) - 1
    fnc_data = fnc_data[:, icn][:, :, icn]

    return upper_triangle_features(fnc_data), df_result[UKBIOBANK_AGE].to_numpy()


def _cell_array_strings(hdf5_file, key_name):
    return [
        ["".join(map(chr, hdf5_file[ref][:])) for ref in column]
        for column in hdf5_file[key_name]
    ]


def _matlab_array(hdf5_file, key_name):
    # MATLAB v7.3 files store arrays column-major; reverse the axes.
    return np.array(hdf5_file[key_name]).T
