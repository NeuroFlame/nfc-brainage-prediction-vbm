"""Define output files produced by the brain age VBM computation."""

import dataclasses

from .report import build_report
from .types import RunSummary, SiteState


def build_outputs(summary: RunSummary, state: SiteState):
    """Write the owner's final model or this member's local model, plus a report."""
    site_name = summary.site_names.get(state.self_token, "")
    if state.self_token == summary.owner_token:
        owner_result = dataclasses.asdict(state.owner_result)
        return {
            "owner_svr_result.json": owner_result,
            "index.html": build_report(site_name, owner_result=owner_result),
        }

    local_result = dataclasses.asdict(state.local_result)
    return {
        "local_svr_result.json": local_result,
        "index.html": build_report(site_name, local_result=local_result),
    }
