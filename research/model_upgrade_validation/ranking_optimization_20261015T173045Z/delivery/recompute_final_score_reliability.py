"""Read-only diagnostics for the complete seven-head research comparison."""
from pathlib import Path
import sys
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import relay_score_reliability as diagnostic
import short_term_thinking_dataset_builder as binding

ROOT = Path(__file__).resolve().parents[1]


def run():
    source = ROOT / "source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv"
    output = ROOT / "score_reliability/final26439"
    if output.exists():
        raise FileExistsError("keep the previous diagnostic; use a new output directory")
    before = binding.sha(source)
    frame = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    if len(frame) != 26439 or frame.groupby("arm").size().ne(3777).any():
        raise ValueError("complete fixed seven-head comparison required")
    periods = {
        "2025H2": frame.date.between("2025-07-01", "2025-12-31"),
        "2026JanAug": frame.date.between("2026-01-01", "2026-08-31"),
        "2026Sep": frame.date.between("2026-09-01", "2026-09-30"),
        "2026Oct": frame.date.between("2026-10-01", "2026-10-15"),
        "HISTORICAL_POOLED": frame.date.lt("2026-10-01"),
    }
    summaries, bins = [], []
    for name, mask in periods.items():
        summary, cells = diagnostic.reliability(frame.loc[mask])
        summary["period"] = name
        cells["period"] = name
        summaries.append(summary)
        bins.append(cells)
    metrics = pd.concat(summaries, ignore_index=True)
    details = pd.concat(bins, ignore_index=True)
    prior_path = ROOT / "score_reliability/all_score_reliability_metrics_UNCALIBRATED.csv"
    prior = pd.read_csv(prior_path)
    key = ["period", "arm"]
    unchanged = metrics.merge(prior[key], on=key, how="inner").set_index(key).sort_index()
    pd.testing.assert_frame_equal(
        unchanged[prior.columns.drop(key)],
        prior.set_index(key).sort_index(),
        check_exact=False, atol=1e-14, rtol=0,
    )
    if binding.sha(source) != before:
        raise ValueError("the frozen score table changed")
    output.mkdir(parents=True)
    metrics.to_csv(output / "raw_score_reliability_UNCALIBRATED.csv", index=False, encoding="utf-8-sig")
    details.to_csv(output / "fixed_decile_bins_NOT_CALIBRATED_PROBABILITIES.csv", index=False, encoding="utf-8-sig")
    sources = [source, prior_path, Path(diagnostic.__file__), Path(__file__), *output.glob("*.csv")]
    binding.write_new(output / "read_only_diagnostic_receipt.json", {
        "formed_at": binding.now(), "input_rows": len(frame), "arms": 7,
        "new_fits": 0, "calibrator_trained": False, "event_probability_validated": False,
        "input_score_or_direction_changed": False,
        "prior_six_head_diagnostic_rows_unchanged": len(unchanged),
        "pairwise_absolute_event_metrics": "NOT_APPLICABLE",
        "unknown_outcomes_never_replaced_with_zero": True,
        "retrospective_historical_diagnostic_not_blind": True,
        "score_interpretation": "Raw internal classifier score, never a verified event probability",
        "source_bindings": {str(p.resolve()): binding.sha(p) for p in sources},
    })
    print({"diagnostic_rows": len(metrics), "bins": len(details), "unchanged_prior_rows": len(unchanged), "new_fits": 0})


if __name__ == "__main__":
    run()
