"""
Generates index.html for Brain Age Prediction VBM results.

Algorithm reference:
  Basodi S, Raja R, Liu J, Verner E, and Calhoun V.
  "Decentralized approaches for Brain Age Prediction." TReNDS Center.
"""

import os


def generate_report(
    output_dir: str,
    site_name: str,
    local_result: dict = None,
    owner_result: dict = None,
):
    if owner_result:
        html = _build_owner_report(owner_result, site_name)
    elif local_result:
        html = _build_local_report(local_result, site_name)
    else:
        html = _build_empty_report()

    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------------------
# Scientific context (shared)
# ---------------------------------------------------------------------------

SCIENTIFIC_CONTEXT = """
    <h2>Scientific Context</h2>
    <p>This computation implements the decentralized brain age prediction algorithm from:</p>
    <blockquote>
      Basodi S, Raja R, Liu J, Verner E, and Calhoun V.
      <em>Decentralized approaches for Brain Age Prediction.</em> TReNDS Center.
    </blockquote>
    <p>The algorithm was originally validated on <strong>UKBiobank</strong> resting-state fMRI data from
    <strong>11,754 subjects</strong> (ages 44–80 years), preprocessed with FSL/SPM12 and group ICA
    to yield 53 intrinsic connectivity networks. The 1,378 upper-triangular FNC values per subject
    were used as features. Data was distributed across 6 sites (1 owner + 5 members),
    90% train / 10% test, repeated 5 times.</p>
    <table>
      <thead>
        <tr><th>Model</th><th>RMSE train</th><th>RMSE test</th><th>MAE train</th><th>MAE test</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Decentralized (this algorithm, paper)</td>
          <td>7.50 ± 0.08</td><td>7.49 ± 0.07</td>
          <td>6.29 ± 0.08</td><td>6.28 ± 0.07</td>
        </tr>
        <tr>
          <td>Centralized (all data pooled, paper)</td>
          <td>7.21 ± 0.01</td><td>7.86 ± 0.08</td>
          <td>5.72 ± 0.01</td><td>6.54 ± 0.06</td>
        </tr>
      </tbody>
    </table>
    <p class="note">This reference table is from the paper's original FNC/GICA connectivity-matrix
    validation run, not VBM voxel data — the algorithm is identical, but the feature type, dataset
    size, and age range differ, so treat it as context rather than a direct benchmark for this run.
    Results from your run will vary based on dataset size, age range, voxel downsampling, number of
    member sites, and the random train/test split.</p>
"""


# ---------------------------------------------------------------------------
# Owner report
# ---------------------------------------------------------------------------

def _build_owner_report(r: dict, site_name: str) -> str:
    rmse_train = r.get("rmse_train_owner", 0)
    rmse_test  = r.get("rmse_test_owner", 0)
    mae_train  = r.get("mae_train_owner", 0)
    mae_test   = r.get("mae_test_owner", 0)
    n_train    = r.get("n_train_samples_owner", 0)
    n_test     = r.get("n_test_samples_owner", 0)
    w_owner    = r.get("w_owner", 0)
    intercept  = r.get("intercept_owner", 0)

    gap_rmse = rmse_test - rmse_train
    gap_mae  = mae_test  - mae_train

    rmse_bar_train = _bar(rmse_train, max(rmse_train, rmse_test) or 1)
    rmse_bar_test  = _bar(rmse_test,  max(rmse_train, rmse_test) or 1)
    mae_bar_train  = _bar(mae_train,  max(mae_train, mae_test) or 1)
    mae_bar_test   = _bar(mae_test,   max(mae_train, mae_test) or 1)

    return _wrap(f"""
    <h1>Brain Age Prediction VBM</h1>
    <p class="subtitle">Owner Site Results &mdash; {site_name}</p>
    <hr>

    <div class="card-row">
      <div class="card accent">
        <div class="card-label">MAE (test)</div>
        <div class="card-value">{mae_test:.3f} <span class="unit">years</span></div>
        <div class="card-sub">Mean absolute brain age error</div>
      </div>
      <div class="card accent">
        <div class="card-label">RMSE (test)</div>
        <div class="card-value">{rmse_test:.3f} <span class="unit">years</span></div>
        <div class="card-sub">Root mean square error</div>
      </div>
      <div class="card">
        <div class="card-label">Subjects</div>
        <div class="card-value">{n_train + n_test}</div>
        <div class="card-sub">{n_train} train &nbsp;/&nbsp; {n_test} test</div>
      </div>
    </div>

    <h2>Model Performance</h2>
    <p>The owner site projected its cached VBM feature matrix through the federated weight average
    <code>U&nbsp;=&nbsp;X&nbsp;&middot;&nbsp;w&#772;</code> and trained a final LinearSVR on the
    compressed features. Errors are in <strong>years of brain age</strong>.</p>

    <table>
      <thead>
        <tr><th>Metric</th><th>This run (train)</th><th>This run (test)</th><th>Gap</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>RMSE (years)</td>
          <td>{rmse_train:.4f}{rmse_bar_train}</td>
          <td>{rmse_test:.4f}{rmse_bar_test}</td>
          <td class="{_gap_class(gap_rmse)}">{gap_rmse:+.4f}</td>
        </tr>
        <tr>
          <td>MAE (years)</td>
          <td>{mae_train:.4f}{mae_bar_train}</td>
          <td>{mae_test:.4f}{mae_bar_test}</td>
          <td class="{_gap_class(gap_mae)}">{gap_mae:+.4f}</td>
        </tr>
        <tr>
          <td>Subjects</td>
          <td>{n_train}</td>
          <td>{n_test}</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>
    <p class="note">Variation between runs is expected — the train/test split is random and the test
    set is only {n_test} subject(s).</p>

    <h2>Model Parameters</h2>
    <table>
      <thead><tr><th>Parameter</th><th>This run</th><th>Description</th></tr></thead>
      <tbody>
        <tr>
          <td>SVR weight (w_owner)</td>
          <td>{_fmt(w_owner)}</td>
          <td>Scalar weight for the projected feature dimension U</td>
        </tr>
        <tr>
          <td>Intercept</td>
          <td>{_fmt(intercept)}</td>
          <td>SVR model intercept (bias term)</td>
        </tr>
      </tbody>
    </table>

    <h2>Algorithm Summary</h2>
    <table>
      <thead><tr><th>Round</th><th>Role</th><th>Action</th></tr></thead>
      <tbody>
        <tr><td>Round 0</td><td>Member sites</td>
            <td>Train local LinearSVR on VBM features; send weight vector to server</td></tr>
        <tr><td>Round 0</td><td><strong>This site (owner)</strong></td>
            <td>Cache train/test split for holdout evaluation</td></tr>
        <tr><td>Aggregation</td><td>Server</td>
            <td>Stack local weight vectors; compute mean w&#772; and broadcast</td></tr>
        <tr><td>Round 1</td><td><strong>This site (owner)</strong></td>
            <td>Project: U&nbsp;=&nbsp;X&nbsp;&middot;&nbsp;w&#772;; train final SVR on U; report results</td></tr>
      </tbody>
    </table>

    <hr>
    {SCIENTIFIC_CONTEXT}
    """)


# ---------------------------------------------------------------------------
# Member report
# ---------------------------------------------------------------------------

def _build_local_report(r: dict, site_name: str) -> str:
    rmse_train = r.get("rmse_train_local", 0)
    rmse_test  = r.get("rmse_test_local", 0)
    mae_train  = r.get("mae_train_local", 0)
    mae_test   = r.get("mae_test_local", 0)
    n_train    = r.get("n_train_samples_local", 0)
    n_test     = r.get("n_test_samples_local", 0)
    n_features = len(r.get("w_local", []))
    w_local    = r.get("w_local", [])

    gap_rmse = rmse_test - rmse_train
    gap_mae  = mae_test  - mae_train

    rmse_bar_train = _bar(rmse_train, max(rmse_train, rmse_test) or 1)
    rmse_bar_test  = _bar(rmse_test,  max(rmse_train, rmse_test) or 1)
    mae_bar_train  = _bar(mae_train,  max(mae_train, mae_test) or 1)
    mae_bar_test   = _bar(mae_test,   max(mae_train, mae_test) or 1)

    return _wrap(f"""
    <h1>Brain Age Prediction VBM</h1>
    <p class="subtitle">Member Site Results &mdash; {site_name}</p>
    <hr>

    <div class="card-row">
      <div class="card accent">
        <div class="card-label">MAE (test)</div>
        <div class="card-value">{mae_test:.6f} <span class="unit">years</span></div>
        <div class="card-sub">Mean absolute brain age error</div>
      </div>
      <div class="card accent">
        <div class="card-label">RMSE (test)</div>
        <div class="card-value">{rmse_test:.6f} <span class="unit">years</span></div>
        <div class="card-sub">Root mean square error</div>
      </div>
      <div class="card">
        <div class="card-label">Input Features</div>
        <div class="card-value">{n_features}</div>
        <div class="card-sub">Flattened VBM voxel values</div>
      </div>
      <div class="card">
        <div class="card-label">Subjects</div>
        <div class="card-value">{n_train + n_test}</div>
        <div class="card-sub">{n_train} train &nbsp;/&nbsp; {n_test} test</div>
      </div>
    </div>

    <h2>Local Model Performance</h2>
    <p>This site trained a local <code>MinMaxScaler + LinearSVR</code> on all available VBM data and
    contributed the learned weight vector to the federated aggregation. Errors are in
    <strong>years of brain age</strong>.</p>

    <table>
      <thead>
        <tr><th>Metric</th><th>This run (train)</th><th>This run (test)</th><th>Gap</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>RMSE (years)</td>
          <td>{rmse_train:.6f}{rmse_bar_train}</td>
          <td>{rmse_test:.6f}{rmse_bar_test}</td>
          <td class="{_gap_class(gap_rmse)}">{gap_rmse:+.6f}</td>
        </tr>
        <tr>
          <td>MAE (years)</td>
          <td>{mae_train:.6f}{mae_bar_train}</td>
          <td>{mae_test:.6f}{mae_bar_test}</td>
          <td class="{_gap_class(gap_mae)}">{gap_mae:+.6f}</td>
        </tr>
        <tr>
          <td>Subjects</td>
          <td>{n_train}</td>
          <td>{n_test}</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>

    <h2>Contribution to Federation</h2>
    <table>
      <thead><tr><th>Item</th><th>This run</th></tr></thead>
      <tbody>
        <tr><td>Weight vector length</td><td>{n_features} coefficients</td></tr>
        <tr><td>Weight vector mean</td><td>{_fmt(_safe_mean(w_local))}</td></tr>
        <tr><td>Weight vector min</td><td>{_fmt(_safe_min(w_local))}</td></tr>
        <tr><td>Weight vector max</td><td>{_fmt(_safe_max(w_local))}</td></tr>
        <tr><td>w_local[0]</td><td>{_fmt(w_local[0] if w_local else 0)}</td></tr>
        <tr><td>Intercept</td><td>{_fmt(_safe_first(r.get("intercept_local")))}</td></tr>
      </tbody>
    </table>
    <p class="note">Small differences in weight values between runs are due to the random train/test
    split — the SVR is fit on all data (train+test merged) so this should be minimal.</p>

    <h2>Algorithm Summary</h2>
    <table>
      <thead><tr><th>Round</th><th>Role</th><th>Action</th></tr></thead>
      <tbody>
        <tr><td>Round 0</td><td><strong>This site (member)</strong></td>
            <td>Train local LinearSVR on VBM features; send w_local to server</td></tr>
        <tr><td>Round 0</td><td>Owner site</td>
            <td>Cache holdout train/test split for final evaluation</td></tr>
        <tr><td>Aggregation</td><td>Server</td>
            <td>Stack member weight vectors; compute mean w&#772; for owner projection</td></tr>
        <tr><td>Round 1</td><td>Owner site</td>
            <td>Project: U&nbsp;=&nbsp;X&nbsp;&middot;&nbsp;w&#772;; train final SVR on U; report results</td></tr>
      </tbody>
    </table>

    <hr>
    {SCIENTIFIC_CONTEXT}
    """)


def _build_empty_report() -> str:
    return _wrap("<h1>Brain Age Prediction VBM</h1><p>No results available.</p>")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals=6):
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _gap_class(gap: float) -> str:
    if abs(gap) < 0.5:
        return "gap-good"
    elif abs(gap) < 2.0:
        return "gap-warn"
    return "gap-bad"


def _bar(value: float, max_value: float) -> str:
    if max_value == 0:
        return ""
    pct = min(100, int(100 * value / max_value))
    return f'<div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>'


def _safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def _safe_min(lst):
    return min(lst) if lst else 0.0

def _safe_max(lst):
    return max(lst) if lst else 0.0

def _safe_first(val):
    if isinstance(val, list):
        return val[0] if val else 0.0
    return val if val is not None else 0.0


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

def _wrap(body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
  <head>
    <title>Brain Age Prediction VBM &mdash; Results</title>
    <style>
      *, *::before, *::after {{ box-sizing: border-box; }}
      body {{ font-family: sans-serif; color: #222; margin: 0; padding: 32px 40px; background: #f9f9f9; }}
      h1 {{ font-size: 1.6em; margin: 0 0 4px 0; color: #1a1a2e; }}
      p.subtitle {{ margin: 0 0 16px 0; color: #555; font-size: 1em; }}
      h2 {{ font-size: 1.1em; color: #16213e; margin: 32px 0 8px 0; padding-bottom: 4px;
            border-bottom: 2px solid #009879; display: inline-block; }}
      hr {{ border: none; border-top: 1px solid #ddd; margin: 32px 0 24px 0; }}
      p {{ color: #555; font-size: 0.9em; margin: 0 0 12px 0; line-height: 1.5; }}
      p.note {{ font-style: italic; color: #888; font-size: 0.85em; margin-top: 4px; }}
      blockquote {{ border-left: 3px solid #009879; margin: 8px 0 16px 0;
                   padding: 6px 16px; color: #555; font-size: 0.9em; }}
      code {{ background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 0.88em; }}
      .card-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0 28px 0; }}
      .card {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px;
               padding: 16px 20px; min-width: 160px; flex: 1; }}
      .card.accent {{ border-top: 4px solid #009879; }}
      .card-label {{ font-size: 0.78em; color: #888; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 6px; }}
      .card-value {{ font-size: 1.6em; font-weight: bold; color: #1a1a2e; line-height: 1.1; }}
      .card-value .unit {{ font-size: 0.5em; font-weight: normal; color: #888; }}
      .card-sub {{ font-size: 0.78em; color: #aaa; margin-top: 4px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 0.9em;
               margin: 12px 0 24px 0; background: white; }}
      table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
      table th, table td {{ padding: 11px 15px; white-space: nowrap; }}
      table tbody tr {{ border-bottom: 1px solid #e8e8e8; }}
      table tbody tr:nth-of-type(even) {{ background-color: #f5f5f5; }}
      table td:first-child {{ font-weight: bold; background-color: white; color: #333; }}
      table tbody tr:nth-of-type(even) td:first-child {{ background-color: #f5f5f5; }}
      td.ref {{ color: #888; font-style: italic; }}
      .bar {{ height: 6px; background: #e8e8e8; border-radius: 3px; margin-top: 5px; width: 120px; }}
      .bar-fill {{ height: 100%; background: #009879; border-radius: 3px; }}
      .gap-good {{ color: #009879; font-weight: bold; }}
      .gap-warn {{ color: #e67e22; font-weight: bold; }}
      .gap-bad  {{ color: #c0392b; font-weight: bold; }}
    </style>
  </head>
  <body>{body}</body>
</html>"""
