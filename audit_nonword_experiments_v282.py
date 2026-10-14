"""Audit completed nonword experiments without altering their forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from next_session_forecast_lab_v271 import ROOT, sha
from nonword_specialist_challenger_v281 import complete_rows, selected
from next_session_expectation_contract_v271 import now, load_registered


def run(output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite experiment audit")
    paths = {"SPECIALIST": ROOT / "results/v281/nonword_specialist_20261002_v1",
             "PAIRWISE": ROOT / "results/v282/nonword_pairwise_20261002_v1"}
    reports = {name: json.loads((path / "report.json").read_text(encoding="utf-8"))
               for name, path in paths.items()}
    checks, temporal = {}, 0
    for name, report in reports.items():
        for path, wanted in report["source_hashes"].items():
            actual = sha(ROOT / path)
            if actual != wanted:
                raise ValueError("research source hash mismatch: " + path)
            checks[path] = actual
        filename = "specialist_predictions_LABEL_ONLY.csv" if name == "SPECIALIST" else "ranker_predictions_LABEL_ONLY.csv"
        if sha(paths[name] / filename) != report["prediction_sha256"]:
            raise ValueError("prediction output hash mismatch: " + name)
        for log in report["selection_logs"]:
            if not (log["max_train_label_date"] < log["validation_start"]
                    and log["max_validation_label_date"] < log["cutoff"]
                    and log["max_refit_label_date"] < log["cutoff"]):
                raise ValueError("noncausal selection/refit")
            temporal += 1
    if reports["SPECIALIST"]["specification"]["fields"] != reports["PAIRWISE"]["specification"]["fields"]:
        raise ValueError("feature-list comparison mismatch")
    pred = pd.read_csv(paths["SPECIALIST"] / "specialist_predictions_LABEL_ONLY.csv", dtype={"code": str})
    diagnostics = {}
    groups = list(pred.groupby("fold")) + [("combined_pre_september", pred[pred.fold.ne("SEPTEMBER")])]
    for fold, frame in groups:
        full = complete_rows(frame)
        domain = full[full.daily_true_one_word.eq(0)]
        possible = domain.groupby("date").target.max()
        picks = selected(full).set_index("date")
        diagnostics[fold] = dict(full_candidate_rows=len(full), nonword_rows=len(domain),
            promotions=int(domain.target.sum()), all_negative_binary_baseline=float(domain.target.eq(0).mean()),
            dates=len(possible), no_promoter_dates=int(possible.eq(0).sum()),
            any_promoter_dates=int(possible.sum()),
            hindsight_oracle_upper_bound_NOT_A_FORECAST=float(possible.mean()),
            specialist_wrong_despite_existing_promoter_dates=int((picks.target.eq(0) & possible.eq(1)).sum()))
    frozen_path = ROOT / "data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
    frozen = load_registered(frozen_path)
    report = dict(version="v282-nonword-result-audit-1", recorded_at=now().isoformat(),
        status="AUDIT_PASSED_RESEARCH_MODELS_REJECTED", unique_research_source_hashes=len(checks),
        strictly_prior_monthly_selection_logs=temporal, aligned_candidate_rows=reports["PAIRWISE"]["denominator_alignment_rows"],
        source_hashes=checks, retrospective_diagnostics=diagnostics,
        specialist=reports["SPECIALIST"]["summaries"], pairwise=reports["PAIRWISE"]["summaries"],
        frozen_forward_plan_hash=frozen["sha256"], frozen_forward_plan_preserved=True,
        decision=dict(default_model_changed=False, probability_display_allowed=False, goal_verified=False,
            reason="Specialist and pairwise full-date Top1 do not beat the existing control consistently or reach 60%; gated specialist also fails",
            hindsight_diagnostics_usable_as_features=False),
        attribution_limit="Pairwise changes loss and logistic regularization/family grid; comparison does not isolate a causal loss-only effect")
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {k: report[k] for k in ("status", "unique_research_source_hashes", "strictly_prior_monthly_selection_logs",
                                  "aligned_candidate_rows", "frozen_forward_plan_preserved", "decision")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=True))
