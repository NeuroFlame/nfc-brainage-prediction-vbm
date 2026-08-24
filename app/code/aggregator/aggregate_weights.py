"""
Pure aggregation logic for the decentralized brain age SVR computation.
Format-agnostic (VBM, GICA, or UKBioBank features) — no NVFlare dependencies.
"""

from typing import List, Dict, Any
import numpy as np


def aggregate_local_svr_weights(site_results: Dict[str, Dict[str, Any]]) -> list:
    """
    Stack weight vectors from all non-owner sites into a single matrix.

    Each site contributes one weight vector (w_local) of shape (n_features,).
    The stacked matrix has shape (n_features, n_sites), which is the format
    expected by the owner site's train_owner_svr() projection step.

    :param site_results: Dict mapping site_name -> local_svr_result dict.
                         Owner site entries (is_owner=True) are excluded automatically.
    :return: w_locals as a list-of-lists, shape (n_features, n_sites).
    """
    weight_vectors = [
        result["w_local"]
        for result in site_results.values()
        if "w_local" in result
    ]

    if not weight_vectors:
        raise ValueError("No weight vectors found — no non-owner site results to aggregate.")

    # Stack rows (each row = one site's weight vector), then transpose to (n_features, n_sites)
    w_locals = np.array(weight_vectors).T
    return w_locals.tolist()
