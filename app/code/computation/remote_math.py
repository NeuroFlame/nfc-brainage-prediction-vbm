"""Server-side aggregation for the brain age VBM computation."""

from typing import Dict

import numpy as np
from framework import with_state

from .types import (
    LocalTraining,
    LocalWeights,
    OwnerDesignation,
    OwnerTraining,
    RemoteState,
    RunSummary,
    SiteSummary,
)


def _resolve_owner_name(parameters, site_names):
    """Return the display name of the owner site: the consortium leader's site.

    The leader sets ``consortium_leader_id`` in the computation parameters to
    their site's display name or user ID; ``site_id_name_map`` resolves an ID
    to the display name.
    """
    leader_id = parameters.get("consortium_leader_id")
    if not leader_id:
        raise ValueError(
            "Set the 'consortium_leader_id' computation parameter to the "
            "consortium leader's site name or user ID; that site trains the "
            "final brain age model"
        )
    owner_name = parameters.get("site_id_name_map", {}).get(leader_id, leader_id)
    if owner_name not in site_names:
        raise ValueError(
            "The consortium leader must take part in the run with data, because "
            "the leader's site trains the final brain age model"
        )
    return owner_name


def designate_owner(site_summaries: Dict[str, SiteSummary], parameters):
    """Resolve the owner site and broadcast its token."""
    site_names = sorted(site_summaries)
    owner_name = _resolve_owner_name(parameters, site_names)
    member_sites = [site for site in site_names if site != owner_name]
    if not member_sites:
        raise ValueError(
            "At least one site other than the consortium leader's is required "
            "to train a local model"
        )

    owner_token = site_summaries[owner_name].self_token
    return with_state(
        OwnerDesignation(owner_token=owner_token),
        RemoteState(
            owner_name=owner_name,
            owner_token=owner_token,
            member_sites=member_sites,
            site_names={
                summary.self_token: site for site, summary in site_summaries.items()
            },
        ),
    )


def average_member_weights(
    site_trainings: Dict[str, LocalTraining], state: RemoteState
):
    """Average the member weight vectors into one (n_features,) vector.

    The owner only ever uses the mean, and with voxel-level features a stack of
    every member's vector grows large, so the mean is taken here.
    """
    missing = [
        site
        for site in state.member_sites
        if site_trainings[site].is_owner or site_trainings[site].result is None
    ]
    if missing:
        raise RuntimeError(f"Member sites {missing} returned no local model")

    w_locals = np.array(
        [site_trainings[site].result.w_local for site in state.member_sites]
    ).T
    return LocalWeights(w_avg=np.mean(w_locals, axis=1), owner_token=state.owner_token)


def finish_run(
    site_trainings: Dict[str, OwnerTraining], state: RemoteState
) -> RunSummary:
    """Confirm the owner trained the final model and name the participants."""
    if not site_trainings[state.owner_name].is_owner:
        raise RuntimeError(f"Owner site {state.owner_name!r} returned no model")
    return RunSummary(
        owner_site=state.owner_name,
        member_sites=list(state.member_sites),
        owner_token=state.owner_token,
        site_names=dict(state.site_names),
    )
