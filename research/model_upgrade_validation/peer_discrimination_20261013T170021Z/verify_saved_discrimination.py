"""Independently verify saved facts and all conditional pair counts; never fit."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(WORKSPACE))
import model_upgrade_history as history


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_same(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


def main():
    plan = read(ROOT / "diagnosis_plan_BEFORE_PAIR_MEASUREMENT.json")
    report = read(ROOT / "source_and_discrimination_qualification.json")
    for path, digest in plan["source_bindings"].items():
        assert sha(path) == digest, path
    for path, digest in report["output_hashes"].items():
        assert sha(ROOT / path) == digest, path
    paths = plan["source_bindings"]
    prepared_path = next(p for p in paths if p.endswith("prepared_source_reconstruction.joblib"))
    source = joblib.load(prepared_path)
    original = source["relay"]
    facts = pd.read_csv(ROOT / "full9245_peer_context_NO_OUTCOMES.csv", dtype={"code": str})
    facts["code"] = facts.code.str.zfill(6)
    assert len(facts) == len(original) == 9245
    assert not facts.duplicated(["date", "code"]).any()
    assert not set(facts.columns) & {"target", "target_date", "truth", "prediction", "promoted"}
    assert list(zip(facts.date, facts.code)) == list(zip(original.date, original.code))
    fields = plan["fixed_candidate_list"]
    for field in fields[:9]:
        numeric_same(facts[field], pd.to_numeric(original[field], errors="coerce"))
    phase = read(next(p for p in paths if p.endswith("source_phase_context.json")))
    phase_map = {(x["date"], x["code"]): x["phase_feature"] or "UNKNOWN" for x in phase["rows"]}
    assert facts.source_phase.tolist() == [phase_map[k] for k in zip(facts.date, facts.code)]
    members = read(ROOT / "original_complete_height_members.json")["groups"]
    member_map = {(x["date"], int(x["boards"])): x for x in members}
    group_count = 0
    for key, group in facts.groupby(["date", "boards"], sort=False):
        recorded = member_map[(key[0], int(key[1]))]
        assert sorted(group.code) == sorted(recorded["codes"])
        assert len(group) == recorded["original_count"]
        assert (group.same_height_rivals == len(group) - 1).all()
        assert (group.original_peer_count == len(group) - 1).all()
        for raw, derived, ratio in [
            ("float_cap_log", "peer_log_cap_difference", False),
            ("signal_amount", "peer_amount_ratio", True),
            ("amplitude", "peer_amplitude_difference", False),
        ]:
            values = group[raw].to_numpy(float)
            complete = np.isfinite(values).all() and (not ratio or (values > 0).all())
            assert recorded["complete_field_scope"][raw] == bool(np.isfinite(values).all())
            expected = np.full(len(group), np.nan)
            if len(group) > 1 and complete:
                for i, value in enumerate(values):
                    median = np.median(np.delete(values, i))
                    expected[i] = value / median if ratio else value - median
            numeric_same(group[derived], expected)
        group_count += 1
    assert group_count == len(members) == 2455
    stats = pd.read_csv(ROOT / "full1080_pre_month_conditional_discrimination.csv")
    cells = pd.read_csv(ROOT / "all_conditional_cells.csv")
    assert len(stats) == 1080 and len(cells) == 54672
    cell_keys = ["month", "period", "proxy_scope", "field", "boards", "source_phase", "legacy_proxy_cell"]
    assert not cells.duplicated(cell_keys).any()
    saved_cells = cells.set_index(cell_keys).to_dict("index")
    saved_stats = stats.set_index(cell_keys[:4]).to_dict("index")
    fact_index = facts.set_index(["date", "code"])
    checked_cells = checked_stats = 0
    for window in report["training_windows"]:
        train = history.training_rows(original, window["cutoff"], "relay")
        assert history._frame_digest(train, source["relay_fields"]) == window["original_member_digest"]
        dates = sorted(train.date.unique())
        assert len(dates) == 252 and len(train) == window["original_n"]
        assert train.target_date.max() == window["latest_target"] < window["cutoff"]
        selected = fact_index.loc[list(zip(train.date, train.code))].reset_index()
        selected["target"] = train.target.to_numpy()
        for period, selected_dates in [("EARLIER189", dates[:-63]), ("RECENT63", dates[-63:])]:
            part = selected[selected.date.isin(selected_dates)]
            for scope in ["ALL", "PROXY0", "PROXY1"]:
                scoped = part if scope == "ALL" else part[part.legacy_proxy_cell == scope]
                groups = list(scoped.groupby(["boards", "source_phase", "legacy_proxy_cell"], dropna=False))
                for field in fields:
                    full_pairs = known_pairs = wins = ties = known_n = support_cells = constant_cells = 0
                    for condition, group in groups:
                        targets = group.target.to_numpy(int)
                        values = group[field].to_numpy(float)
                        positive = np.sort(values[(targets == 1) & np.isfinite(values)])
                        negative = np.sort(values[(targets == 0) & np.isfinite(values)])
                        possible = int((targets == 1).sum() * (targets == 0).sum())
                        usable = len(positive) * len(negative)
                        # Different algorithm from the measurement's pair-difference matrix.
                        less = np.searchsorted(negative, positive, side="left")
                        equal = np.searchsorted(negative, positive, side="right") - less
                        win = int(less.sum())
                        tie = int(equal.sum())
                        known = int(np.isfinite(values).sum())
                        key = (window["month"], period, scope, field, *condition)
                        actual = saved_cells[key]
                        for col, expected in {
                            "original_n": len(group), "original_positive": int((targets == 1).sum()),
                            "original_negative": int((targets == 0).sum()), "known_field_n": known,
                            "full_pair_count": possible, "known_pair_count": usable,
                            "wins_increasing_field": win, "ties": tie,
                            "signal_days": group.date.nunique(),
                        }.items():
                            assert actual[col] == expected, (key, col)
                        numeric_same([actual["concordance"]], [(win + .5 * tie) / usable if usable else np.nan])
                        full_pairs += possible
                        known_pairs += usable
                        wins += win
                        ties += tie
                        known_n += known
                        support_cells += int(usable > 0)
                        constant_cells += int(usable > 0 and np.unique(values[np.isfinite(values)]).size == 1)
                        checked_cells += 1
                    actual = saved_stats[(window["month"], period, scope, field)]
                    for col, expected in {
                        "original_n": len(scoped), "field_known_n": known_n,
                        "all_label_pairs_within_cells": full_pairs, "known_field_pairs": known_pairs,
                        "wins_increasing_field": wins, "ties": ties,
                        "support_cells_with_both_classes_and_field": support_cells,
                        "constant_value_eligible_cells": constant_cells,
                    }.items():
                        assert actual[col] == expected, (window["month"], period, scope, field, col)
                    numeric_same(
                        [actual["field_coverage"], actual["pair_coverage"], actual["concordance"]],
                        [known_n / len(scoped) if len(scoped) else np.nan,
                         known_pairs / full_pairs if full_pairs else np.nan,
                         (wins + .5 * ties) / known_pairs if known_pairs else np.nan],
                    )
                    checked_stats += 1
    assert checked_cells == len(cells) and checked_stats == len(stats)
    assert sha(prepared_path) == plan["source_bindings"][prepared_path]
    result = {
        "formed_at": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "source_hashes_checked": len(paths), "full_original_rows": len(facts),
        "original_peer_groups": group_count, "pre_month_windows": 15,
        "all_summary_rows_checked": checked_stats, "all_conditional_cells_checked": checked_cells,
        "pair_recalculation": "sorted_negative_search_not_pair_difference_matrix",
        "no_target_fields_in_feature_file": True, "original_members_labels_fields_unchanged": True,
        "source_phase_is_archived_algorithm_not_independent_gold": True,
        "research_fits": 0, "new_mature_forward_samples": 0, "prediction_gain_verified": False,
        "verifier_sha256": sha(__file__), "checked_output_hashes": report["output_hashes"],
    }
    with (ROOT / "independent_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
