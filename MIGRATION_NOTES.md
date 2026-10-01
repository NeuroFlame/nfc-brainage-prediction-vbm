# Migration to computation-nvflare-boilerplate

This computation was migrated from the hand-written NVFlare
`Controller`/`Executor`/`Aggregator` architecture (NVFlare 2.4, Python 3.8) to
[`computation-nvflare-boilerplate`](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
`0.1.0` (NVFlare 2.8, Python 3.11). Framework-owned files were applied with the
boilerplate's `scripts/migrate_computation.py --in-place --force` at boilerplate
commit `dbc9503`. The SVR math was ported into `app/code/computation/`.

The port follows
[nfc-brainage-prediction-fnc-gica](https://github.com/NeuroFlame/nfc-brainage-prediction-fnc-gica),
which this repository was forked from before either was migrated: the two
shared an identical controller, aggregator and utilities, and differ only in
how features are loaded and in the report text. The sections below repeat the
design that both now share and note where this repository differs.

## Workflow

The old two tasks map to a `stepped_workflow` with three local/remote pairs
and a final output step:

| Old task | New local step | New remote step |
|---|---|---|
| — | `prepare_site` (with `load_inputs`) | `designate_owner` |
| `GET_LOCAL_SVR_WEIGHTS` | `train_member_model` | `average_member_weights` |
| `ACCEPT_AGGREGATED_WEIGHTS` | `train_owner_model` | `finish_run` |
| — | `build_outputs` (output step) | — |

The extra first round exists because a site needs to know whether it is the
owner before round 0's training (see [Owner site](#owner-site)).

## Owner site

The old executor compared its own NVFlare client name with an `owner_site`
parameter. That parameter was set by a customized `system/entry_provision.py`
to the first provisioned user. The framework does not tell a site its own
identity, and `system/entry_provision.py` is owned by the boilerplate, so the
migration replaced it.

The owner is now the consortium leader's site, named by the required
`consortium_leader_id` parameter (a site display name, or a NeuroFLAME user ID
resolved through `site_id_name_map`):

- In the first round, each site mints a random token and keeps it in local
  state.
- The server resolves the owner's display name to its token and sends the
  token to every site. A site acts as the owner when the token matches its own.
- The run stops with an error if `consortium_leader_id` is missing, if the
  leader's site is not taking part, or if no other site takes part.

## Local state holds indices, not data

The old owner cached its train/test feature matrices in the NVFlare context
between rounds. Local state now holds only the train/test row indices, and
rounds that need the data re-read the site's NIfTI files. A VBM feature matrix
is far too large for the framework's serialized local state (the bundled site2
is 105 subjects × 271,633 voxels at a downsample factor of 2).

## The server sends the mean weight vector, not the stack

This is where this repository differs from the FNC/GICA sibling. The old
aggregator stacked every member's weight vector into an `n_features ×
n_members` matrix and broadcast it; the owner only ever used its row-wise mean.
The server now takes that mean itself (`average_member_weights`) and sends one
`n_features` vector. The owner's result is unchanged, the payload no longer
grows with the number of members, and no site receives another site's
individual weight vector.

The framework limits arrays sent inline to 8 MiB by default. One value per
voxel is about 2.2 MB at the test data's downsample factor of 2, and about
17 MB for an undownsampled 121 × 145 × 121 grid, so `spec.py` raises the limit
to 64 MiB.

## Other intentional differences

- **`owner_site` is no longer read.** Replace it with `consortium_leader_id`.
- **Defaults**: `input_source` defaults to `"VBM"`, `split_type` to `"random"`,
  `test_size` to `0.1`, `shuffle` to `true`, and the SVR parameters to the
  values in the bundled `parameters.json`. Previously, omitting any of these
  raised `KeyError`.
- **Outputs are unchanged**: `owner_svr_result.json` at the owner,
  `local_svr_result.json` at each member, and `index.html` at every site. The
  report is the old one, now built in `app/code/computation/report.py` and
  returned to the framework instead of being written directly.
- **Dropped rows are logged.** The message about covariate rows dropped for a
  missing filename or age goes to the site log instead of standard output.
- **Unused code removed**: k-fold partition generators, `.npy` loading, and the
  `__main__` test stubs in the old utilities. None fed into the output.
- **Dependencies**: boilerplate pins plus nibabel, h5py, scipy, scikit-learn,
  joblib, and threadpoolctl. `h5py` moved from 3.1.0 to 3.9.0 because 3.1.0
  does not install on Python 3.11. `nibabel` is now a hard requirement rather
  than an optional import.

## Verification

- **Numeric parity**: the pre-migration and migrated code were run in the same
  Python environment on the bundled test data (site1 as owner, site2 as member)
  with identical train/test splits, for both `random` and
  `age_range_stratified` splitting. The member result, the owner result, and
  both sites' `index.html` reports are identical.
- A 2-site `./run_local_simulation.sh site1,site2` run completes. site1 (the
  leader in the test parameters) writes the owner outputs and site2 writes the
  member outputs.
- `make check` (ruff, formatting, compile, unit tests) and
  `migrate_computation.py --check` (0 differing paths) pass.

**Not verified**: a run with more than one member site (the bundled test data
has only two sites), and a run on undownsampled volumes.

## Known issues carried over

**High owner error on the bundled test data.** The owner model's test RMSE is
about 11–13 years, while the member model fits its own data almost perfectly
(test RMSE about 0.001 years, because members train on train and test data
together). The pre-migration code gives the same numbers, so the migration did
not cause this.
