"""Define values exchanged by the brain age VBM computation.

The framework does not tell a site its own identity, so each site mints a
random ``self_token`` in the first round and keeps it in local state. The
server, which sees display names, resolves the owner (the consortium leader's
site) to that site's token and broadcasts it; a site acts as the owner when the
broadcast ``owner_token`` matches its own.

Local state keeps only the train/test row indices, not the feature matrices,
so it stays small for large voxel-level inputs. Later rounds reload the site's inputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SiteInputs:
    """Features and ages loaded from one site's input files."""

    X: np.ndarray
    y: np.ndarray


@dataclass
class LocalSvrResult:
    """Local LinearSVR trained by a member site in round 0."""

    w_local: List[float]
    intercept_local: List[float]
    n_train_samples_local: int
    n_test_samples_local: int
    rmse_train_local: float
    rmse_test_local: float
    mae_train_local: float
    mae_test_local: float


@dataclass
class OwnerSvrResult:
    """Projected LinearSVR trained by the owner site in round 1."""

    w_owner: float
    intercept_owner: float
    n_train_samples_owner: int
    n_test_samples_owner: int
    rmse_train_owner: float
    rmse_test_owner: float
    mae_train_owner: float
    mae_test_owner: float


@dataclass
class SiteState:
    """Local state kept by a site between rounds."""

    self_token: str
    train_index: np.ndarray
    test_index: np.ndarray
    local_result: Optional[LocalSvrResult] = None
    owner_result: Optional[OwnerSvrResult] = None


@dataclass
class SiteSummary:
    """Token and subject count reported by one site in the first round."""

    self_token: str
    n_subjects: int


@dataclass
class OwnerDesignation:
    """Owner token broadcast to all sites."""

    owner_token: str


@dataclass
class LocalTraining:
    """Round 0 result from one site; empty for the owner."""

    is_owner: bool
    result: Optional[LocalSvrResult] = None


@dataclass
class LocalWeights:
    """Mean of the member weight vectors, shape (n_features,)."""

    w_avg: np.ndarray
    owner_token: str


@dataclass
class OwnerTraining:
    """Round 1 result from one site; empty for members."""

    is_owner: bool
    result: Optional[OwnerSvrResult] = None


@dataclass
class RemoteState:
    """Server-side state kept between aggregation rounds."""

    owner_name: str
    owner_token: str
    member_sites: List[str] = field(default_factory=list)
    site_names: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Participants broadcast once the owner has finished.

    ``site_names`` maps each site's token to its display name so a site can
    label its own report.
    """

    owner_site: str
    member_sites: List[str]
    owner_token: str
    site_names: Dict[str, str]
