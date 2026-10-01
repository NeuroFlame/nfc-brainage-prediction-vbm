"""Exercise the brain age VBM workflow steps on the bundled test data."""

import glob
import json
import os
import unittest

import numpy as np
from computation import inputs, local_math, remote_math, results

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(REPO_ROOT, "test_data")
SITES = ["site1", "site2"]
# 121 x 145 x 121 voxel volumes, keeping every other voxel along each axis.
N_FEATURES = 61 * 73 * 61
# site2 lists 107 subjects; the two without an age are dropped.
N_SUBJECTS = {"site1": 55, "site2": 105}


def load_parameters(**overrides):
    with open(os.path.join(TEST_DATA, "server", "parameters.json")) as f:
        parameters = json.load(f)
    parameters.update(overrides)
    return parameters


def unwrap(value):
    """Split a with_state() result into (payload, state)."""
    if hasattr(value, "state"):
        return value.payload, value.state
    return value, None


def site_dir(site):
    return os.path.join(TEST_DATA, site)


def load_site(site, parameters):
    return inputs.load_inputs(
        site_dir(site),
        input_source=parameters["input_source"],
        data_file=parameters["data_file"],
        label_file=parameters["label_file"],
        vbm_downsample_factor=parameters["vbm_downsample_factor"],
    )


def run_workflow(parameters):
    states, summaries = {}, {}
    for site in SITES:
        site_inputs = load_site(site, parameters)
        summaries[site], states[site] = unwrap(
            local_math.prepare_site(
                site_inputs,
                parameters["split_type"],
                parameters["test_size"],
                parameters["shuffle"],
            )
        )

    designation, remote_state = unwrap(
        remote_math.designate_owner(summaries, parameters)
    )

    trainings = {}
    for site in SITES:
        trainings[site], new_state = unwrap(
            local_math.train_member_model(
                designation, states[site], site_dir(site), parameters
            )
        )
        states[site] = new_state or states[site]

    weights = remote_math.average_member_weights(trainings, remote_state)

    owner_trainings = {}
    for site in SITES:
        owner_trainings[site], new_state = unwrap(
            local_math.train_owner_model(
                weights, states[site], site_dir(site), parameters
            )
        )
        states[site] = new_state or states[site]

    summary = remote_math.finish_run(owner_trainings, remote_state)
    outputs = {site: results.build_outputs(summary, states[site]) for site in SITES}
    return trainings, weights, outputs


class InputsTest(unittest.TestCase):
    def test_vbm_features_are_flattened_downsampled_volumes(self):
        site_inputs = load_site("site1", load_parameters())
        self.assertEqual(site_inputs.X.shape, (N_SUBJECTS["site1"], N_FEATURES))
        self.assertEqual(site_inputs.X.dtype, np.float32)
        self.assertEqual(site_inputs.y.shape, (N_SUBJECTS["site1"],))

    def test_downsample_factor_sets_the_feature_count(self):
        file_name = os.path.basename(
            sorted(glob.glob(os.path.join(site_dir("site1"), "*.nii")))[0]
        )
        volume = inputs.read_vbm_image(site_dir("site1"), file_name)
        self.assertEqual(volume.shape, (121, 145, 121))
        downsampled = inputs.read_vbm_image(site_dir("site1"), file_name, 2)
        self.assertEqual(downsampled.shape, (61, 73, 61))

    def test_unknown_input_source_is_rejected(self):
        with self.assertRaises(ValueError):
            inputs.load_inputs(site_dir("site1"), input_source="other")


class SplitTest(unittest.TestCase):
    def test_splits_partition_all_subjects(self):
        y = np.linspace(20, 80, 40)
        for split_type in ("random", "age_range_stratified"):
            train, test = local_math.split_indices(y, split_type, 0.25, True)
            self.assertEqual(len(test), 10)
            self.assertEqual(sorted(np.concatenate([train, test])), list(range(40)))


class WorkflowTest(unittest.TestCase):
    def test_only_owner_trains_final_model_and_is_excluded_from_members(self):
        parameters = load_parameters(consortium_leader_id="site2")
        trainings, weights, outputs = run_workflow(parameters)

        self.assertTrue(trainings["site2"].is_owner)
        self.assertIsNone(trainings["site2"].result)
        self.assertEqual(weights.w_avg.shape, (N_FEATURES,))
        # With one member, the mean is that member's own weight vector.
        np.testing.assert_array_equal(
            weights.w_avg, np.array(trainings["site1"].result.w_local)
        )

        self.assertEqual(
            sorted(outputs["site2"]), ["index.html", "owner_svr_result.json"]
        )
        owner = outputs["site2"]["owner_svr_result.json"]
        self.assertEqual(
            owner["n_train_samples_owner"] + owner["n_test_samples_owner"],
            N_SUBJECTS["site2"],
        )
        self.assertIsInstance(owner["w_owner"], float)
        self.assertIn("site2", outputs["site2"]["index.html"])

        self.assertEqual(
            sorted(outputs["site1"]), ["index.html", "local_svr_result.json"]
        )
        self.assertEqual(
            len(outputs["site1"]["local_svr_result.json"]["w_local"]), N_FEATURES
        )

    def test_leader_resolved_through_site_id_name_map(self):
        parameters = load_parameters(
            consortium_leader_id="user-1", site_id_name_map={"user-1": "site1"}
        )
        trainings, _, _ = run_workflow(parameters)
        self.assertEqual(
            [site for site, t in trainings.items() if t.is_owner], ["site1"]
        )

    def test_missing_or_absent_leader_fails(self):
        summaries = {site: None for site in SITES}
        for leader in (None, "site9"):
            with self.assertRaises(ValueError):
                remote_math.designate_owner(
                    summaries, load_parameters(consortium_leader_id=leader)
                )


if __name__ == "__main__":
    unittest.main()
