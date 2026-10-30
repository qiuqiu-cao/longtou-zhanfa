"""Target-free whole-pool and temporal-prefix qualification; never fit models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_long_high_features as source
import first_board_event_path_features as features

OUT = Path(__file__).resolve().parent
FACTS = base.ROOT / "research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"


def main():
    protocol = OUT / "EVENT_PATH_QUALIFICATION_BEFORE_COMPUTE.json"
    if protocol.exists():
        raise FileExistsError("Preserve each actual qualification attempt; no overwrite")
    original = stage.VARIANTS["STAGE_CYCLE_GENE"]
    columns = list(dict.fromkeys(features.KEYS + ["stage", "close"] + original))
    facts = pd.read_csv(FACTS, usecols=columns, dtype={"code": str}, float_precision="round_trip")
    facts = facts[facts.stage.eq(1)].reset_index(drop=True)
    assert len(facts) == 28503 and facts.signal_date.nunique() == 668
    keys = facts[features.KEYS]
    base.write_json(protocol, dict(started_at=base.now(), fits=0, labels_read=False,
        source_sha256=base.sha(features.__file__), raw_sources={str(p.resolve()): base.sha(p) for p in base.INPUTS.values()},
        facts_sha256=base.sha(FACTS), requested_rows=len(keys), window_official_sessions=20,
        event_definition="Latest prior nominal 10% exact-cent close-limit / touch-without-close separately, over S-20..S-1",
        path_definition="Event->S-1 low drawdown, worst close return including first correction day, median volume/event;S volume/correction and close/event high",
        absence="Full21 valid consecutive traded bars required;found=0 means observed no event,missing window=NaN;no correction days leaves path variables NaN",
        limits="Unadjusted nominal price basis; imported pre_close is previous raw close;not official ex-rights truth or fund intent",
        eligibility="Design qualification only; no agreed model arms or training permission"))
    bars = source.load_source()
    actual = features.from_bars(bars, keys)
    full = facts[features.KEYS + ["close"]].merge(actual, on=features.KEYS, validate="one_to_one")
    observed = full.event_path_source_close.notna()
    assert np.array_equal(np.rint(full.loc[observed, "close"] * 100), np.rint(full.loc[observed, "event_path_source_close"] * 100))
    assert full[features.KEYS].equals(keys)
    destination = OUT / "COMPLETE28503_EVENT_PATH_FEATURES_NO_TARGETS.csv"
    actual.to_csv(destination, index=False, encoding="utf-8-sig")
    prefix_checks = []
    for cutoff in ("2025-06-30", "2025-12-31"):
        prefix = features.from_bars(bars[bars.date.le(cutoff)], keys[keys.signal_date.le(cutoff)])
        expected = actual[actual.signal_date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)
        prefix_checks.append(dict(cutoff=cutoff, rows=len(prefix), all_features_and_diagnostics_equal=True))
    summaries = {}
    masks = dict(development=facts.signal_date.lt("2025-01-01"),
        validation_2025H1=facts.signal_date.between("2025-01-01", "2025-06-30"),
        evaluation_14_months=facts.signal_date.between("2025-07-01", "2026-08-31"))
    fields = features.FIELDS["sealed"] + features.FIELDS["probe"]
    for name, mask in masks.items():
        part = actual[mask]
        item = dict(rows=len(part), dates=part.signal_date.nunique(),
            full_window_rows=int(part.event_path_full_window_observed.sum()),
            full_window_fraction=float(part.event_path_full_window_observed.mean()), kinds={})
        for kind in features.KINDS:
            own = features.FIELDS[kind]
            found = part[own[0]].eq(1)
            complete = part[own].notna().all(axis=1)
            item["kinds"][kind] = dict(event_found_rows=int(found.sum()),
                observed_no_event_rows=int(part[own[0]].eq(0).sum()),
                source_window_unknown_rows=int(part[own[0]].isna().sum()),
                event_dates=part.loc[found, "signal_date"].nunique(), complete_path_rows=int(complete.sum()),
                complete_path_dates=part.loc[complete, "signal_date"].nunique(),
                field_observed_rows={field:int(part[field].notna().sum()) for field in own})
        summaries[name] = item
    duplicates = []
    for field in fields:
        for old in original:
            if np.allclose(actual[field].to_numpy(float), facts[old].to_numpy(float), atol=0, rtol=0, equal_nan=True):
                duplicates.append([field, old])
    base.write_json(OUT / "EVENT_PATH_QUALIFICATION.json", dict(completed_at=base.now(), status="COMPLETE_DESIGN_QUALIFICATION_NO_FITS",
        whole_pool_rows=len(actual), dates=actual.signal_date.nunique(), features=fields,
        protocol_sha256=base.sha(protocol), feature_file_sha256=base.sha(destination), feature_source_sha256=base.sha(features.__file__),
        all_signal_prices_match=True, exact_original_feature_duplicates=duplicates,
        prefix_checks=prefix_checks, statistics=summaries, fits=0, labels_read=False,
        outcome_association_tested=False, deployment_allowed=False))
    print(json.dumps(dict(status="COMPLETE", statistics=summaries, prefix_checks=prefix_checks, exact_duplicates=duplicates), ensure_ascii=False))


if __name__ == "__main__":
    main()
