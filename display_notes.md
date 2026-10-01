# Brain Age Prediction VBM

### Overview

This computation implements the decentralized brain age prediction algorithm described in:

> Basodi S, Raja R, Liu J, Verner E, and Calhoun V. *Decentralized approaches for Brain Age Prediction.* TReNDS Center.

The algorithm uses a two-round decentralized Support Vector Regression (SVR). This version takes VBM (voxel-based morphometry) grey-matter density maps as features — one NIfTI volume per subject, typically the smoothed/warped/modulated grey-matter tissue class output of an SPM segmentation pipeline. Each site contributes local model weights without sharing raw subject data. One designated site acts as the "owner" (holdout) site and serves as an independent evaluator; all other sites train local SVR models and contribute their learned weights to the aggregation step.

**Owner site**: The consortium leader identifies their site with the `consortium_leader_id` setting. The leader must take part in the run with data, and at least one other site must participate to train a local model.

The same federated algorithm was originally validated with FNC (functional network connectivity) matrices as features instead of VBM images; `input_source` still supports `"GICA"` and `"UKBioBank_Comp2019"` for that data format.

**Reference dataset (original FNC/GICA validation):** The paper validated the algorithm on **UKBiobank** resting-state fMRI data from **11,754 subjects** (ages 44–80), preprocessed with FSL/SPM12 and ICA to yield 53 intrinsic connectivity networks (ICNs), producing **1,378 upper-triangular FNC features** per subject. Data was distributed across 6 sites (1 owner + 5 members), with 90% train / 10% test per site. The decentralized model achieved RMSE of ~7.5 years and MAE of ~6.3 years on the test set — performance on par with a centralized model trained on all data pooled together. VBM feature counts and expected error will differ; see Settings Specification below.

### Example Settings

```json
{
    "consortium_leader_id": "site1",
    "input_source": "VBM",
    "vbm_downsample_factor": 2,
    "split_type": "random",
    "test_size": 0.1,
    "shuffle": true,
    "svr_params_local": {
        "epsilon": 0,
        "tol": 0.0001,
        "C": 1,
        "loss": "epsilon_insensitive",
        "fit_intercept": true,
        "intercept_scaling": 1,
        "dual": true,
        "random_state": 0,
        "max_iter": 10000
    },
    "svr_params_owner": {
        "epsilon": 0,
        "tol": 0.0001,
        "C": 1,
        "loss": "epsilon_insensitive",
        "fit_intercept": true,
        "intercept_scaling": 1,
        "dual": true,
        "random_state": 0,
        "max_iter": 10000
    }
}
```

### Settings Specification

| Variable Name | Type | Description | Allowed Options | Default | Required |
| --- | --- | --- | --- | --- | --- |
| `consortium_leader_id` | `string` | The consortium leader's site name (or NeuroFLAME user ID). That site is the owner: it holds out its data and trains the final model. | a participating site | — | ✅ true |
| `input_source` | `string` | Specifies the format/source of the input data. | `"VBM"`, `"GICA"`, `"UKBioBank_Comp2019"` | `"VBM"` | ❌ false |
| `vbm_downsample_factor` | `int` | VBM only. Stride applied to each spatial axis of the NIfTI volumes to reduce the voxel count (e.g. `2` keeps every other voxel per axis). Must be identical across all sites. | `>= 1` | `1` | ❌ false |
| `split_type` | `string` | Method used to split each site's data into train and test sets. | `"random"`, `"age_range_stratified"` | `"random"` | ❌ false |
| `test_size` | `float` | Fraction of subjects reserved for the test set at each site. | `0.0` – `1.0` | `0.1` | ❌ false |
| `shuffle` | `boolean` | Whether to shuffle data before splitting. | `true`, `false` | `true` | ❌ false |
| `svr_params_local` | `dict` | `sklearn.svm.LinearSVR` keyword arguments applied at non-owner sites during round 0 local training. | See sklearn docs | the values in the example above | ❌ false |
| `svr_params_owner` | `dict` | `sklearn.svm.LinearSVR` keyword arguments applied at the owner site during round 1 projected training. | See sklearn docs | the values in the example above | ❌ false |

See `test_data/server/parameters.json` for a complete working example (including the optional `data_file`, `label_file`, and `site_id_name_map`).

### Input Description

Two kinds of input are required at each site, and must be named exactly as below:

1. **NIfTI image files** — one per subject, named as listed in the `niftifilename` column of `covariates.csv`
2. **Covariates/Labels File** (`covariates.csv`)

Both must reside in the same directory.

**NIfTI Image Files**

- **Format**: NIfTI-1 (`.nii` or `.nii.gz`)
- **Contents**: A single 3D grey-matter density volume per subject — typically the smoothed, warped, modulated grey-matter tissue class (`swc1*`) output of an SPM12 VBM pipeline. All subjects across all sites must share the same voxel grid (dimensions, orientation, and voxel size) so that flattened feature vectors line up across sites.
- **Feature extraction**: Each volume is loaded, optionally strided by `vbm_downsample_factor` along each spatial axis, and flattened into a 1D feature vector.

**Covariates / Labels File (`covariates.csv`)**

- **Format**: CSV (Comma-Separated Values)
- **Headers**: Must include a header row. The columns `niftifilename` and `age` are required; `age` is used as the regression target. Extra columns (e.g. `site`, `sex`, `isControl`) are ignored.
- **Rows**: Each row represents one subject. Rows with a missing `niftifilename` or `age` are dropped before training.

**General Structure**:

```
niftifilename,site,isControl,age,sex
subject_001_swc1.nii,IA,FALSE,24.5,M
subject_002_swc1.nii,IA,TRUE,31.2,F
...
```

### Algorithm Description

The key steps of the algorithm include:

1. **Round 0 — Local SVR Training (non-owner sites)**:
    - Each non-owner site loads its NIfTI images and covariates, flattens each (optionally downsampled) volume into a feature vector, and splits data into train and test sets.
    - A `MinMaxScaler + LinearSVR` pipeline is fit on all available local data (train + test combined), maximizing the signal contributed to the federated aggregation.
    - Each site returns its learned weight vector (`w_local`), intercept, and performance metrics (RMSE, MAE) to the server.

2. **Round 0 — Owner Site**:
    - The owner site loads and splits its data but does not train a model. It keeps its train/test split for use in round 1.

3. **Server Aggregation (between rounds)**:
    - The server averages the received local weight vectors into `w_avg` (one value per feature) and broadcasts it to all sites.

4. **Round 1 — Owner Projected SVR**:
    - The owner site projects its cached feature matrix through the averaged local weights: `U = X @ w_avg`, compressing the high-dimensional voxel space into a single federated signal dimension.
    - A second `MinMaxScaler + LinearSVR` pipeline is fit on the projected features `U`, and final performance metrics are computed and saved.

5. **Non-Owner Sites (round 1)**:
    - Non-owner sites receive the aggregated weights but perform no further computation.

### Assumptions

- The NIfTI image files and `covariates.csv` provided by each site are in the same directory and follow the expected format described above.
- Every filename listed in `covariates.csv`'s `niftifilename` column exists in that site's data directory.
- All NIfTI volumes, across every site, share the same voxel grid (same dimensions and orientation) so that flattened/downsampled feature vectors are aligned across sites.
- The `age` column must be present in the covariates CSV and contain numeric values; rows with a missing filename or age are dropped automatically.
- `vbm_downsample_factor` must be set identically for every site — a mismatch produces feature vectors of different lengths, which fails when the server aggregates weight vectors.
- The computation is run in a federated environment where each site contributes valid, non-empty data.
- The owner site must have sufficient data to support a meaningful holdout evaluation (i.e., `test_size` subjects after splitting).

### Output Description

- **Output files**: `local_svr_result.json` (per non-owner site), `owner_svr_result.json` (owner site), and an `index.html` report at every site.
- All files are written to each site's output directory at the end of the run.

The computation outputs both **site-level** and **owner-level** results, which include:

**Non-Owner Sites** (`local_svr_result.json`):

| Field | Description |
| --- | --- |
| `w_local` | Learned SVR weight vector (one value per input feature — a voxel for VBM, an FNC pair for GICA) |
| `intercept_local` | SVR model intercept |
| `n_train_samples_local` | Number of subjects in the training set |
| `n_test_samples_local` | Number of subjects in the test set |
| `rmse_train_local` | Root Mean Square Error on training data |
| `rmse_test_local` | Root Mean Square Error on test data |
| `mae_train_local` | Mean Absolute Error on training data |
| `mae_test_local` | Mean Absolute Error on test data |

**Owner Site** (`owner_svr_result.json`):

| Field | Description |
| --- | --- |
| `w_owner` | SVR weight for the projected feature dimension |
| `intercept_owner` | SVR model intercept |
| `n_train_samples_owner` | Number of subjects in the owner training set |
| `n_test_samples_owner` | Number of subjects in the owner test set |
| `rmse_train_owner` | Root Mean Square Error on owner training data |
| `rmse_test_owner` | Root Mean Square Error on owner test data |
| `mae_train_owner` | Mean Absolute Error on owner training data |
| `mae_test_owner` | Mean Absolute Error on owner test data |
