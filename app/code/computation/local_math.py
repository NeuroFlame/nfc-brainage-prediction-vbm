"""Site-side math: data splitting, member SVR training, owner projected SVR."""

import dataclasses
import math
import uuid

import numpy as np
import pandas as pd
from framework import with_state
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from .inputs import load_inputs
from .types import (
    LocalSvrResult,
    LocalTraining,
    LocalWeights,
    OwnerDesignation,
    OwnerSvrResult,
    OwnerTraining,
    SiteInputs,
    SiteState,
    SiteSummary,
)

DEFAULT_SVR_PARAMS = {
    "epsilon": 0,
    "tol": 0.0001,
    "C": 1,
    "loss": "epsilon_insensitive",
    "fit_intercept": True,
    "intercept_scaling": 1,
    "dual": True,
    "random_state": 0,
    "max_iter": 10000,
}


def prepare_site(inputs: SiteInputs, split_type="random", test_size=0.1, shuffle=True):
    """Split this site's subjects into train/test sets and mint its token."""
    train_index, test_index = split_indices(inputs.y, split_type, test_size, shuffle)
    self_token = uuid.uuid4().hex
    summary = SiteSummary(self_token=self_token, n_subjects=len(inputs.y))
    state = SiteState(
        self_token=self_token, train_index=train_index, test_index=test_index
    )
    return with_state(summary, state)


def train_member_model(
    designation: OwnerDesignation, state: SiteState, data_dir, parameters
):
    """Train a local LinearSVR on all member data; the owner skips this round."""
    if state.self_token == designation.owner_token:
        return LocalTraining(is_owner=True)

    X_train, X_test, y_train, y_test = _load_split(data_dir, parameters, state)
    svr_params = parameters.get("svr_params_local", DEFAULT_SVR_PARAMS)

    # Fit on train + test so members contribute the most signal to the federation.
    pipeline = _fit_svr(
        np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), svr_params
    )
    svr_model = pipeline.named_steps["linearsvr"]
    train_metrics = regression_metrics(y_train, pipeline.predict(X_train))
    test_metrics = regression_metrics(y_test, pipeline.predict(X_test))

    result = LocalSvrResult(
        w_local=np.squeeze(svr_model.coef_).tolist(),
        intercept_local=np.atleast_1d(svr_model.intercept_).tolist(),
        n_train_samples_local=len(y_train),
        n_test_samples_local=len(y_test),
        rmse_train_local=train_metrics["rmse"],
        rmse_test_local=test_metrics["rmse"],
        mae_train_local=train_metrics["mae"],
        mae_test_local=test_metrics["mae"],
    )
    return with_state(
        LocalTraining(is_owner=False, result=result),
        dataclasses.replace(state, local_result=result),
    )


def train_owner_model(weights: LocalWeights, state: SiteState, data_dir, parameters):
    """Project owner data through the mean member weights and fit a second SVR.

    The projection U = X @ w_avg compresses the high-dimensional feature space
    into a scalar that captures the shared cross-site signal.
    """
    if state.self_token != weights.owner_token:
        return OwnerTraining(is_owner=False)

    X_train, X_test, y_train, y_test = _load_split(data_dir, parameters, state)
    svr_params = parameters.get("svr_params_owner", DEFAULT_SVR_PARAMS)

    w_avg = weights.w_avg.reshape(-1, 1)
    U_train = np.matmul(X_train, w_avg).astype(np.double)
    U_test = np.matmul(X_test, w_avg).astype(np.double)

    pipeline = _fit_svr(U_train, y_train, svr_params)
    svr_model = pipeline.named_steps["linearsvr"]
    if svr_params.get("fit_intercept", True):
        intercept_owner = float(np.atleast_1d(svr_model.intercept_)[0])
    else:
        intercept_owner = 0.0
    train_metrics = regression_metrics(y_train, pipeline.predict(U_train))
    test_metrics = regression_metrics(y_test, pipeline.predict(U_test))

    result = OwnerSvrResult(
        w_owner=float(np.squeeze(svr_model.coef_)),
        intercept_owner=intercept_owner,
        n_train_samples_owner=len(y_train),
        n_test_samples_owner=len(y_test),
        rmse_train_owner=train_metrics["rmse"],
        rmse_test_owner=test_metrics["rmse"],
        mae_train_owner=train_metrics["mae"],
        mae_test_owner=test_metrics["mae"],
    )
    return with_state(
        OwnerTraining(is_owner=True, result=result),
        dataclasses.replace(state, owner_result=result),
    )


def split_indices(y: np.ndarray, split_type, test_size, shuffle):
    """Return (train_index, test_index) row indices for this site's subjects."""
    if split_type == "age_range_stratified":
        # Stratify on four equal-size age bins.
        age_bins = pd.qcut(y, 4, labels=False)
        splitter = StratifiedShuffleSplit(
            n_splits=1, random_state=42, test_size=test_size
        )
        train_index, test_index = next(splitter.split(np.zeros(len(y)), age_bins))
        return train_index, test_index
    return train_test_split(np.arange(len(y)), test_size=test_size, shuffle=shuffle)


def regression_metrics(y_true, y_pred):
    """Return RMSE and MAE for a set of age predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit_svr(X, y, svr_params):
    pipeline = make_pipeline(preprocessing.MinMaxScaler(), LinearSVR(**svr_params))
    pipeline.fit(X, y)
    return pipeline


def _load_split(data_dir, parameters, state: SiteState):
    inputs = load_inputs(
        data_dir,
        **{
            key: parameters[key]
            for key in (
                "input_source",
                "data_file",
                "label_file",
                "vbm_downsample_factor",
            )
            if key in parameters
        },
    )
    return (
        inputs.X[state.train_index],
        inputs.X[state.test_index],
        inputs.y[state.train_index],
        inputs.y[state.test_index],
    )
