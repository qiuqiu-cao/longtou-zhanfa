"""Read-only positive report cross-check, not a replacement target generator."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage

OUT = Path(__file__).resolve().parent
FACTS = base.ROOT / "research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
MANIFEST = base.ROOT / "research/first_board_joint_design_20261020T185417Z/ALL668_SOURCE_RECEIPTS.json"


def cents(value):
    try:
        n = Decimal(str(value))*100
        return int(n) if n.is_finite() and n > 0 and n == n.to_integral_value() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def main():
    protocol = OUT / "LABEL_CROSSCHECK_BEFORE_COMPUTE.json"
    if protocol.exists():
        raise FileExistsError("Preserve original audit attempts")
    before = {str(p):base.sha(p) for p in [FACTS, MANIFEST, *base.INPUTS.values()]}
    base.write_json(protocol, dict(started_at=base.now(), fits=0, protected_hashes=before,
        scope="All28503originalfirst-boardkeys;only668already archived dated report envelopes;no new requests",
        interpretation="Report appearance at exact raw close is third-party positive corroboration;absence UNKNOWN,not negative. Integer15 provisional board-level meaning,not exchange truth.",
        target_action="No overwriting old labels;all unknowns retained;conflicts pending verification,not proven ex-rights errors"))
    facts = pd.read_csv(FACTS, usecols=["signal_date", "code", "stage", "close", "label_end", stage.TARGET],
                        dtype={"code":str}, float_precision="round_trip")
    facts = facts[facts.stage.eq(1)].reset_index(drop=True)
    assert len(facts) == 28503
    sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
    next_day = dict(zip(sessions[:-1], sessions[1:]))
    assert facts.label_end.eq(facts.signal_date.map(next_day)).all()
    manifests = json.loads(MANIFEST.read_text("utf-8"))
    reports = {}
    dates = []
    for item in manifests:
        if not item.get("qualified_date_envelope"):
            dates.append(dict(date=item["date"], status="UNAVAILABLE"))
            continue
        path = Path(item["source_path"])
        assert base.sha(path) == item["sha256"]
        payload = json.loads(path.read_bytes())
        assert str(payload["errcode"]) == "0" and payload["info"][1] == item["date"]
        raw = payload["info"][0]
        counts = Counter(str(row[0]).zfill(6) for row in raw if isinstance(row, list) and len(row) >= 23)
        for i, row in enumerate(raw):
            if not isinstance(row, list) or len(row) < 23:
                continue
            code = str(row[0]).zfill(6)
            if not code.isdigit() or len(code) != 6 or counts[code] != 1:
                continue
            reports[item["date"],code] = dict(provider_close=row[21], provider_field15=row[15],
                provider_pct=row[22], source_path=str(path), source_sha256=item["sha256"],
                pointer=f"/info/0/{i}", received_at=item.get("received_at"))
        dates.append(dict(date=item["date"], status="DATED_ENVELOPE_HASH_OK", records=len(raw)))
    prices = pd.concat([pd.read_csv(base.INPUTS[name], usecols=["date", "code", "close"], dtype={"code":str},
                    float_precision="round_trip") for name in ("bars", "recovered")], ignore_index=True)
    prices["date"] = prices.date.astype(str).str[:10]
    assert not prices.duplicated(["date", "code"]).any()
    closes = dict(zip(zip(prices.date,prices.code.str.zfill(6)), prices.close))
    records = []
    for row in facts.itertuples(index=False):
        signal = reports.get((row.signal_date,row.code), {})
        target = reports.get((row.label_end,row.code), {})
        target_close = closes.get((row.label_end,row.code), np.nan)
        same_signal = cents(row.close) is not None and cents(row.close) == cents(signal.get("provider_close"))
        same_target = cents(target_close) is not None and cents(target_close) == cents(target.get("provider_close"))
        y = getattr(row, stage.TARGET)
        if not target:
            category = "NO_TARGET_REPORT_UNKNOWN_NOT_NEGATIVE"
        elif not same_target:
            category = "TARGET_REPORT_PRICE_CONFLICT_OR_NO_RAW_CLOSE_PENDING"
        elif pd.isna(y):
            category = "ORIGINAL_TARGET_UNKNOWN_WITH_POSITIVE_REPORT_NO_REFILL"
        elif y == 1:
            category = "ORIGINAL_POSITIVE_CORROBORATED_BY_REPORT_CLOSE"
        else:
            category = "ORIGINAL_NEGATIVE_VS_POSITIVE_REPORT_PENDING"
        record = dict(signal_date=row.signal_date, code=row.code, label_end=row.label_end,
            original_target_LABEL_ONLY=y, original_signal_close=row.close, original_target_raw_close=target_close,
            signal_report_present=bool(signal), signal_report_price_equal=same_signal,
            signal_report_field15=signal.get("provider_field15"),
            target_report_present=bool(target), target_report_price_equal=same_target,
            target_report_field15=target.get("provider_field15"),
            target_report_close=target.get("provider_close"), target_report_pct=target.get("provider_pct"),
            target_source_sha256=target.get("source_sha256"), target_source_pointer=target.get("pointer"),
            target_source_path=target.get("source_path"), target_source_received_at=target.get("received_at"),
            crosscheck_category=category)
        records.append(record)
    result = pd.DataFrame(records)
    assert len(result) == 28503 and result.original_target_LABEL_ONLY.isna().sum() == 126
    result.to_csv(OUT / "COMPLETE28503_LABEL_REPORT_CROSSCHECK.csv", index=False, encoding="utf-8-sig")
    conflicts = result[result.crosscheck_category.isin(["TARGET_REPORT_PRICE_CONFLICT_OR_NO_RAW_CLOSE_PENDING", "ORIGINAL_NEGATIVE_VS_POSITIVE_REPORT_PENDING"])]
    conflicts.to_csv(OUT / "ALL_PENDING_LABEL_REPORT_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    assert before == {str(p):base.sha(p) for p in [FACTS, MANIFEST, *base.INPUTS.values()]}
    pd.DataFrame(dates).to_csv(OUT / "ALL668_RECHECKED_SOURCE_DATES.csv", index=False)
    report = dict(completed_at=base.now(), fits=0, status="READ_ONLY_POSITIVE_CROSSCHECK_COMPLETE",
        original_rows=len(result), original_unknown126_preserved=True, original_hashes_unchanged=True,
        report_rows=len(reports), dated_sources=len(dates), categories=result.crosscheck_category.value_counts().to_dict(),
        source_presence_by_original_target=result.assign(target_group=result.original_target_LABEL_ONLY.fillna(-1)).groupby("target_group").agg(
            rows=("code","size"), report_present=("target_report_present","sum"), matching_price=("target_report_price_equal","sum")).reset_index().to_dict("records"),
        pending_difference_rows=len(conflicts), target_rewrites=0, confirmed_corporate_action_errors=0,
        interpretation="No report does not establish a negative. Report conflicts do not establish false labels;integer15 remains provisional. No outcome or price corrections made.")
    base.write_json(OUT / "LABEL_REPORT_CROSSCHECK.json", report)
    print(json.dumps(base.clean(report), ensure_ascii=True))


if __name__ == "__main__":
    main()
