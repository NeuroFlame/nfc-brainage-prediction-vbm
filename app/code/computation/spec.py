"""Declare the brain age VBM computation workflow."""

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
)

from .inputs import load_inputs
from .local_math import prepare_site, train_member_model, train_owner_model
from .remote_math import average_member_weights, designate_owner, finish_run
from .results import build_outputs

# The averaged member weight vector has one value per voxel: about 2.1 million
# (17 MB) for an undownsampled 121 x 145 x 121 VBM grid, above the framework's
# default 8 MiB inline-array limit.
MAX_INLINE_ARRAY_BYTES = 64 * 1024 * 1024

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=prepare_site, input_fn=load_inputs),
        remote_step(fn=designate_owner),
        local_step(fn=train_member_model),
        remote_step(fn=average_member_weights),
        local_step(fn=train_owner_model),
        remote_step(fn=finish_run),
        site_output_step(fn=build_outputs),
    ),
    max_inline_array_bytes=MAX_INLINE_ARRAY_BYTES,
)
