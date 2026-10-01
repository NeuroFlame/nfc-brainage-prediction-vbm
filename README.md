# Brain Age Prediction VBM

Brain age prediction using decentralized SVR with VBM (voxel-based morphometry) grey-matter density maps as features. Adapted from [NeuroFlame/nfc-brainage-prediction-fnc-gica](https://github.com/NeuroFlame/nfc-brainage-prediction-fnc-gica) (which used FNC/GICA connectivity matrices instead of VBM images), itself ported from [trendscenter/coinstac-brainage-fnc](https://github.com/trendscenter/coinstac-brainage-fnc) to the NeuroFLAME / NVFlare federated learning framework. The federated SVR algorithm is unchanged — only the input data format and feature extraction differ. `input_source` still supports `"GICA"` and `"UKBioBank_Comp2019"` for backward compatibility.

---

## Algorithm

The consortium leader's site is the **owner**: it holds out its data and trains the final model. All other sites are **members**.

1. **Setup** — Every site loads its VBM data, splits it into train/test sets, and reports a random token. The server resolves the owner from `consortium_leader_id` and tells the sites which token is the owner's.
2. **Member training** — Each member trains a `MinMaxScaler + LinearSVR` on all of its data and sends back the learned weight vector. The owner skips this step.
3. **Aggregation** — The server averages the member weight vectors into `w_avg` (one value per feature) and sends it to all sites.
4. **Owner training** — The owner projects its data through the averaged member weights (`U = X @ w_avg`), then trains a second `LinearSVR` on the projected feature.
5. **Outputs** — The owner writes `owner_svr_result.json`; each member writes its `local_svr_result.json`. Every site writes an `index.html` report.

---

## Project Structure

This computation follows the [NeuroFLAME computation boilerplate](https://github.com/NeuroFlame/computation-nvflare-boilerplate) (version recorded in `.neuroflame.json`). Computation code lives in `app/code/computation/`:

```
app/code/computation/
├── spec.py          # Workflow: which steps run, in what order
├── types.py         # Values exchanged between steps
├── inputs.py        # NIfTI + covariates loading (and the FNC formats)
├── local_math.py    # Train/test split, member SVR, owner projected SVR
├── remote_math.py   # Owner resolution, weight averaging
├── results.py       # Output files per site
└── report.py        # index.html report
```

`app/code/framework/`, `app/code/runtime/`, `app/config/`, `system/`, the Dockerfiles, and the scripts are owned by the boilerplate. Update them with the boilerplate's `scripts/migrate_computation.py`, not by hand.

See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for what changed when this computation was migrated to the boilerplate, and how the migration was verified.

---

## Input Data

Each site directory under `test_data/` must contain:

- `covariates.csv` — CSV with at minimum `niftifilename` and `age` columns. Any other columns (e.g. `site`, `sex`, `isControl`) are ignored by the computation.
- One `.nii`/`.nii.gz` file per subject, named exactly as listed in the `niftifilename` column of `covariates.csv`, all sharing the same voxel grid (dimensions and orientation) — e.g. SPM `swc1*` smoothed/warped/modulated grey-matter density maps in MNI space.

Rows in `covariates.csv` with a missing filename or age are dropped automatically.

---

## Computation Parameters

Edit `test_data/server/parameters.json` to configure the run:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `consortium_leader_id` | The consortium leader's site name or NeuroFLAME user ID. That site is the owner. **Required.** | — |
| `data_file` | FNC `.mat` filename. Unused when `input_source` is `"VBM"` (each subject has its own NIfTI file). | `null` |
| `label_file` | Covariates CSV filename | `"covariates.csv"` |
| `input_source` | `"VBM"`, `"GICA"`, or `"UKBioBank_Comp2019"` | `"VBM"` |
| `vbm_downsample_factor` | VBM only — stride applied to each spatial axis of the NIfTI volumes to shrink the voxel count (e.g. `2` keeps every other voxel per axis, an 8x reduction). Must be identical across all sites so feature vectors line up. | `1` |
| `split_type` | `"random"` or `"age_range_stratified"` | `"random"` |
| `test_size` | Fraction of subjects for test set | `0.1` |
| `shuffle` | Shuffle before splitting | `true` |
| `svr_params_local` | `LinearSVR` kwargs for member sites | see file |
| `svr_params_owner` | `LinearSVR` kwargs for the owner site | see file |
| `log_level` | `debug`, `info`, `warning`, `error`, or `critical` | `"info"` |

The test parameters set `consortium_leader_id` to `site1`, and `vbm_downsample_factor` to `2` (rather than the default of `1`) to keep the sample run fast.

---

## Running Locally

Run a local simulation with the bundled test data (requires Docker):

```bash
./run_local_simulation.sh site1,site2
```

Results are written to `test_output/simulate_job/<site>/`. Add `--no-build` to skip rebuilding the image after source-only changes. The bundled `test_data/` ships `site1` (owner) and `site2` (member); add more `siteN` directories under `test_data/` (each with its own `covariates.csv` and NIfTI files) and extend the site list to run with more sites.

Run lint, formatting checks, and unit tests with:

```bash
make check
```

---

## Notes on COINSTAC → NeuroFLAME Translation

| COINSTAC | NeuroFLAME |
|----------|------------|
| `local.py local_0()` | `local_math.prepare_site()` + `local_math.train_member_model()` |
| `remote.py remote_0()` | `remote_math.designate_owner()` + `remote_math.average_member_weights()` |
| `local.py local_1()` | `local_math.train_owner_model()` |
| `remote.py remote_1()` | `remote_math.finish_run()` |
| `state['owner']` / `clientId` check | `consortium_leader_id`, matched through per-site tokens |
| `args['cache']` (disk) | Framework local state (`with_state`), holding train/test indices |
| `compspec.json` inputs | `test_data/server/parameters.json` |
