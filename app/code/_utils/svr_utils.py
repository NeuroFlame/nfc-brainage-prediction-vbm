#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import nibabel as nib
except ImportError:
    nib = None

from _utils.preprocessor_utils import get_upper_diagonal_values, split_xy_data
import _utils.preprocessor_utils as preut

def read_fnc_gica(file_dir, file_name):
    data = loadmat(os.path.join(file_dir, file_name))['fnc_corrs_all']
    """
    Data from first session.
    """
    fnc_data = data[:, 0, :, :]

    return fnc_data


def read_vbm_image(file_dir, file_name, downsample_factor=1):
    """
    Load a single-subject VBM NIfTI volume (e.g. an SPM `swc1*.nii[.gz]` smoothed,
    warped, modulated grey-matter density map) as a 3D numpy array.

    :param file_dir: Directory containing the NIfTI file.
    :param file_name: NIfTI filename (`.nii` or `.nii.gz`).
    :param downsample_factor: Stride applied along each spatial axis to reduce the
                              voxel count (e.g. 2 keeps every other voxel). Must be
                              the same for every subject/site so all sites end up
                              with matching feature-vector lengths.
    :return: 3D float32 numpy array of voxel intensities.
    """
    if nib is None:
        raise ImportError(
            "nibabel is required to read VBM NIfTI files. Install with `pip install nibabel`."
        )

    img = nib.load(os.path.join(file_dir, file_name))
    data = img.get_fdata(dtype=np.float32)

    if downsample_factor and downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]

    return data


def read_fnc_ukbiobank(file_dir, file_name):
    with h5py.File(os.path.join(file_dir, file_name), 'r') as f:
        fN_data = preut.get_cell_array_data(f, 'fN')[0]
        icn_ins_data = preut.get_numpy_array(f, 'icn_ins')
        fnc_data = preut.get_numpy_array(f, 'corrdata')

    return fN_data, fnc_data, icn_ins_data


"""
=============================================================================
The below function forms the matrices X, Y.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- input_file : name of the file containing fnc matrices used as features and 
               actual age of the corresponding subjects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension (subjects x features)
 - Y: The response matrix of dimension (subjects,)
=============================================================================
"""


def form_XYMatrices(input_dir, input_source, data_file, label_file, vbm_downsample_factor=1):
    if input_source == "GICA":
        X, y, subj_ref_data = form_XYMatrices_gica(input_dir, data_file, label_file)

    elif input_source == "UKBioBank_Comp2019":
        X, y, subj_ref_data = form_XYMatrices_ukbiobank(input_dir, data_file, label_file,
                                                            extract_icn_features_only=True)

    elif input_source == "VBM":
        X, y, subj_ref_data = form_XYMatrices_vbm(input_dir, label_file,
                                                    downsample_factor=vbm_downsample_factor)

    else:
        raise ValueError(f"Unknown input_source: {input_source!r}")

    return X, y


def form_XYMatrices_gica(input_dir, data_file, label_file):
    fnc_data = read_fnc_gica(input_dir, data_file)
    covariate_df = pd.read_csv(os.path.join(input_dir, label_file))

    assert len(fnc_data) == len(covariate_df), \
        f"Number of subjects in fnc_data({len(fnc_data)}) and covariate data({len(covariate_df)}) do not match."

    y = covariate_df["age"].to_numpy()
    subj_ref_data = covariate_df["filename"].tolist()

    """
    Extract only the FNC matrix corresponding to icn_ins_indexes
    """
    X = preut.get_upper_diagonal_values(fnc_data)

    return X, y, subj_ref_data


def form_XYMatrices_vbm(input_dir, label_file, downsample_factor=1,
                        subject_col="niftifilename", age_col="age"):
    """
    Build the VBM feature matrix from a directory of per-subject NIfTI grey-matter
    density maps plus a covariates CSV.

    Unlike the FNC formats above (a single .mat/.h5 file holding every subject),
    VBM data ships as one NIfTI file per subject, named in the covariates CSV.
    Each subject's volume is loaded, optionally downsampled, and flattened into a
    feature vector; the flattened vectors are stacked into X in covariate-row order.

    :param input_dir: Directory containing the covariates CSV and NIfTI files.
    :param label_file: Covariates CSV filename. Must contain a subject/filename
                       column (default 'niftifilename') and an age column
                       (default 'age').
    :param downsample_factor: Stride passed to read_vbm_image(); must be identical
                              across all sites so feature vectors line up.
    :param subject_col: Covariates column holding each subject's NIfTI filename.
    :param age_col: Covariates column holding the regression target.
    :return: (X, y, subj_ref_data) — X is (subjects x voxels), y is (subjects,).
    """
    covariate_df = pd.read_csv(os.path.join(input_dir, label_file))
    covariate_df.columns = covariate_df.columns.str.strip()

    if subject_col not in covariate_df.columns or age_col not in covariate_df.columns:
        raise KeyError(
            f"Covariates file '{label_file}' must contain '{subject_col}' and '{age_col}' "
            f"columns; found {list(covariate_df.columns)}."
        )

    covariate_df[subject_col] = covariate_df[subject_col].astype(str).str.strip()

    n_rows = len(covariate_df)
    covariate_df = covariate_df.dropna(subset=[age_col, subject_col]).reset_index(drop=True)
    if len(covariate_df) < n_rows:
        print(f"Dropped {n_rows - len(covariate_df)} row(s) from {label_file} with missing "
              f"'{subject_col}' or '{age_col}'.")

    features = []
    ref_shape = None
    for file_name in covariate_df[subject_col]:
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"NIfTI file listed in {label_file} not found: {file_path}"
            )

        volume = read_vbm_image(input_dir, file_name, downsample_factor=downsample_factor)
        if ref_shape is None:
            ref_shape = volume.shape
        elif volume.shape != ref_shape:
            raise ValueError(
                f"Volume shape mismatch for {file_name}: expected {ref_shape}, got "
                f"{volume.shape}. All subjects must share the same image grid."
            )
        features.append(volume.reshape(-1))

    X = np.vstack(features)
    y = covariate_df[age_col].to_numpy(dtype=np.float64)
    subj_ref_data = covariate_df[subject_col].tolist()

    return X, y, subj_ref_data


def form_XYMatrices_ukbiobank(input_dir, data_file, label_file, extract_icn_features_only=True):
    COL_NAME_SUBJECT_ID = "eid"
    COL_NAME_AGE = "age_when_attended_assessment_centre_f21003_2_0"

    def get_eid_age_from_file(file_dir, file_name):
        df = pd.read_table(os.path.join(file_dir, file_name))
        df_eid_age = df[[COL_NAME_SUBJECT_ID, COL_NAME_AGE]]
        """
        Note: Need to perform inner join with eid of this df with filenames in fN variable in mat file
        """
        return df_eid_age

    def get_eid_from_fN(fN_data):
        eid_list = []
        for data in fN_data:
            eid_list.append(int(data.split("/")[8].split("_")[0]))

        df_eid = pd.DataFrame(eid_list, columns=[COL_NAME_SUBJECT_ID])

        return df_eid

    subj_ref_data, fnc_orig_data, icn_ins_data = read_fnc_ukbiobank(input_dir, data_file)
    df_eid = get_eid_from_fN(subj_ref_data)

    df_eid_age = get_eid_age_from_file(input_dir, label_file)

    df_result_eid = pd.concat([df_eid_age, df_eid], axis=1, join="inner").reindex(df_eid.index)
    y = df_result_eid[[COL_NAME_AGE]].to_numpy()

    fnc_data = fnc_orig_data

    """
    Extract only the FNC matrix corresponding to icn_ins_indexes
    """
    if extract_icn_features_only:
        icn_ins_data = icn_ins_data.astype(int).reshape((len(icn_ins_data)))
        icn_ins_data = icn_ins_data - 1  # Note: Matlab indices starts from 1 where as python starts from 0.
        fnc_data = fnc_data[:][:, icn_ins_data][:, :, icn_ins_data]

    X = preut.get_upper_diagonal_values(fnc_data)
    return (X, y, subj_ref_data)


def get_metrics(y_true, y_pred):
    """
    Computes performance evaluation metrics for regression model.
    """
    results = {}
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = math.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)

    return results


if __name__ == "__main__":
    def test_functions():
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],
                      [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 1, 2, 3, 1, 1, 2, 3, 2, 0, 1, 3, 0, 3, 2, 0, 1, 3, 0])
        y = np.array(range(len(X)))

        # preut.get_random_splits(3, X, y, shuffle_data=True, test_size=0.1)
        # preut.generate_k_splits(X, y, y, "", k=3, shuffle_data=True, type="age_range_stratified")
        preut.get_age_range_stratified_splits(2, X, y, shuffle_data=True, test_size=0.5, num_bins=4)

        # GICA
        form_XYMatrices("../../test/input/local0/simulatorRun", "GICA", "coinstac-gica_postprocess_results.mat",
                            "covariates.csv")


    test_functions()
