import json
import logging
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from _utils.utils import get_data_directory_path, get_output_directory_path
from .local_svr import load_site_data, train_local_svr, train_owner_svr
from .report_generator import generate_report

TASK_NAME_GET_LOCAL_SVR_WEIGHTS = "GET_LOCAL_SVR_WEIGHTS"
TASK_NAME_ACCEPT_AGGREGATED_WEIGHTS = "ACCEPT_AGGREGATED_WEIGHTS"


class BrainAgeFNCExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        logging.info(f"Task Name: {task_name}")

        if task_name == TASK_NAME_GET_LOCAL_SVR_WEIGHTS:
            computation_parameters = get_computation_parameters(fl_ctx)
            site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME) or fl_ctx.get_identity_name()
            display_name = get_display_name(site_name, computation_parameters)

            data_dir = get_data_directory_path(fl_ctx)
            splits = load_site_data(
                data_dir=data_dir,
                data_file=computation_parameters.get("data_file"),
                label_file=computation_parameters.get("label_file", "covariates.csv"),
                input_source=computation_parameters["input_source"],
                split_type=computation_parameters["split_type"],
                test_size=computation_parameters["test_size"],
                shuffle=computation_parameters["shuffle"],
                vbm_downsample_factor=computation_parameters.get("vbm_downsample_factor", 1),
            )

            if is_owner_site(site_name, computation_parameters):
                # Owner site: cache splits in-memory for round 1, send no weights
                fl_ctx.set_prop("cached_splits", splits, private=True, sticky=True)
                logging.info(f"Owner site '{display_name}': cached train/test splits for round 1.")
                result = Shareable()
                result["is_owner"] = True
                return result

            else:
                # Non-owner site: train local SVR and return weights
                local_result = train_local_svr(
                    X_train=splits["X_train"],
                    X_test=splits["X_test"],
                    y_train=splits["y_train"],
                    y_test=splits["y_test"],
                    svr_params=computation_parameters["svr_params_local"],
                )
                output_dir = get_output_directory_path(fl_ctx)
                save_results_to_file(local_result, "local_svr_result.json", output_dir)
                generate_report(
                    output_dir=output_dir,
                    site_name=display_name,
                    local_result=local_result,
                )
                result = Shareable()
                result["local_svr_result"] = local_result
                return result

        if task_name == TASK_NAME_ACCEPT_AGGREGATED_WEIGHTS:
            computation_parameters = get_computation_parameters(fl_ctx)
            site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME) or fl_ctx.get_identity_name()
            display_name = get_display_name(site_name, computation_parameters)

            if is_owner_site(site_name, computation_parameters):
                w_locals = shareable["w_locals"]
                cached_splits = fl_ctx.get_prop("cached_splits")

                owner_result = train_owner_svr(
                    X_train=cached_splits["X_train"],
                    X_test=cached_splits["X_test"],
                    y_train=cached_splits["y_train"],
                    y_test=cached_splits["y_test"],
                    w_locals=w_locals,
                    svr_params=computation_parameters["svr_params_owner"],
                )
                output_dir = get_output_directory_path(fl_ctx)
                save_results_to_file(owner_result, "owner_svr_result.json", output_dir)
                generate_report(
                    output_dir=output_dir,
                    site_name=display_name,
                    owner_result=owner_result,
                )
                result = Shareable()
                result["owner_svr_result"] = owner_result
                return result

            else:
                # Non-owner sites: nothing to do in round 1
                return Shareable()


def is_owner_site(site_name: str, computation_parameters: dict) -> bool:
    """
    Check whether the current site is the owner site.

    In NeuroFLAME, site_name is the MongoDB ObjectId. The owner_site value
    may be an ID (set by entry_provision.py) or a human-readable name (set
    manually). We resolve both directions via site_id_name_map.
    """
    owner_site = computation_parameters.get("owner_site", "")
    site_id_name_map = computation_parameters.get("site_id_name_map", {})

    if site_name == owner_site:
        return True

    resolved_name = site_id_name_map.get(site_name, "")
    if resolved_name and resolved_name == owner_site:
        return True

    name_to_id = {v: k for k, v in site_id_name_map.items()}
    resolved_id = name_to_id.get(site_name, "")
    if resolved_id and resolved_id == owner_site:
        return True

    return False


def get_display_name(site_name: str, computation_parameters: dict) -> str:
    """Resolve the human-readable display name for a site."""
    site_id_name_map = computation_parameters.get("site_id_name_map", {})
    return site_id_name_map.get(site_name, site_name)


def get_computation_parameters(fl_ctx: FLContext) -> dict:
    return fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS", {})


def save_results_to_file(results: dict, file_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to: {os.path.join(output_dir, file_name)}")
    except Exception as e:
        raise RuntimeError(f"Failed to save results to {file_name}: {e}")
