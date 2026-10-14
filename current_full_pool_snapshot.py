"""Standardize one archived closing-limit and failed-board pool.

Provider verification, full-market completeness, and point-in-time security
eligibility are deliberately separate.  Missing official security evidence is
kept unavailable rather than inferred from a non-ST-looking name.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
COL = {
    "code": "代码", "name": "名称", "change": "涨跌幅", "last": "最新价",
    "amount": "成交额", "float_cap": "流通市值", "total_cap": "总市值",
    "turnover": "换手率", "first_seal": "首次封板时间", "failed_count": "炸板次数",
    "limit_stat": "涨停统计", "industry": "所属行业", "boards": "连板数",
    "limit_price": "涨停价", "amplitude": "振幅", "sealed_funds": "封板资金",
    "last_seal": "最后封板时间",
}


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _normalize_master(master):
    required = {"code", "valid_from", "valid_to", "exchange", "board", "listing_status",
                "st_status", "price_limit_regime", "source_type", "source_ref", "data_status"}
    missing = required - set(master)
    if missing:
        raise ValueError("security master missing fields: " + ",".join(sorted(missing)))
    result = master.copy()
    result["code"] = result.code.astype(str).str.zfill(6)
    result["valid_from"] = result.valid_from.astype(str)
    result["valid_to"] = result.valid_to.astype(str)
    return result


def _security(master, code, date):
    matched = master[(master.code == code) & (master.valid_from <= date) & (master.valid_to >= date)]
    if len(matched) != 1:
        return {
            "security_status": "unavailable", "security_evidence_status": "missing_or_ambiguous",
            "is_st": None, "is_delisting_period": None, "no_price_limit": None,
            "security_source": None, "security_source_type": None,
        }
    row = matched.iloc[0]
    verified_source = (_truth(row.get("source_verified"))
                       or "official" in str(row.source_type).lower())
    verified = (verified_source and str(row.data_status).startswith("verified")
                and row.board == "MAINBOARD" and row.exchange in {"SSE", "SZSE"})
    if not verified:
        return {
            "security_status": "unavailable", "security_evidence_status": "source_not_verified",
            "is_st": None, "is_delisting_period": None, "no_price_limit": None,
            "security_source": str(row.source_ref), "security_source_type": str(row.source_type),
        }
    listed = row.listing_status == "LISTED"
    normal = row.st_status == "NORMAL"
    normal_limit = row.price_limit_regime == "MAINBOARD_10"
    return {
        "security_status": "verified", "security_evidence_status": "official_interval_match",
        "is_st": not normal, "is_delisting_period": not listed,
        "no_price_limit": not normal_limit,
        "security_source": str(row.source_ref), "security_source_type": str(row.source_type),
    }


def _required(frame, names, label):
    missing = [COL[name] for name in names if COL[name] not in frame]
    if missing:
        raise ValueError(label + " missing fields: " + ",".join(missing))


def build(limit_up, failed, manifest, security_master, signal_date):
    signal_compact = signal_date.replace("-", "")
    if str(manifest.get("date")) != signal_compact:
        raise ValueError("manifest date mismatch")
    for label, frame, manifest_key in (("limit_up", limit_up, "limit_up"),
                                       ("failed_board", failed, "failed_board")):
        if manifest.get("datasets", {}).get(manifest_key, {}).get("status") != "verified":
            raise ValueError(label + " provider status not verified")
        if manifest["datasets"][manifest_key].get("rows") != len(frame):
            raise ValueError(label + " manifest row count mismatch")
    _required(limit_up, ["code", "name", "change", "last", "amount", "float_cap", "turnover",
                         "first_seal", "failed_count", "limit_stat", "industry", "boards"], "limit_up")
    _required(failed, ["code", "name", "change", "last", "amount", "float_cap", "turnover",
                       "first_seal", "failed_count", "limit_stat", "industry", "limit_price"], "failed_board")
    master = _normalize_master(security_master)
    collected = str(manifest.get("collected_at", ""))
    if not collected:
        raise ValueError("manifest collection time missing")
    if "+" not in collected and not collected.endswith("Z"):
        collected += "+08:00"
    rows = []
    for pool, frame in (("CLOSING_LIMIT_UP", limit_up), ("FAILED_BOARD_OBSERVATION", failed)):
        sealed = pool == "CLOSING_LIMIT_UP"
        for record in frame.to_dict("records"):
            code = str(record[COL["code"]]).zfill(6)
            mainboard = code.startswith(MAINBOARD_PREFIXES)
            security = _security(master, code, signal_date) if mainboard else {
                "security_status": "verified", "security_evidence_status": "prefix_out_of_scope",
                "is_st": None, "is_delisting_period": None, "no_price_limit": None,
                "security_source": None, "security_source_type": None,
            }
            boards = int(record[COL["boards"]]) if sealed and pd.notna(record[COL["boards"]]) else 0
            row = {
                "candidate_id": signal_date + ":" + code + ":" + pool,
                "signal_date": signal_date, "code": code, "name": record[COL["name"]],
                "boards": boards, "board_group": ("FIRST_BOARD" if boards == 1 else
                                                    "CONSECUTIVE_BOARD" if boards >= 2 else "FAILED_BOARD"),
                "pool_assignment": pool, "is_mainboard": bool(mainboard),
                "is_limit_up": bool(sealed), "touched_limit": True, "touch_status": "verified",
                "limit_up_status": "verified", "identity_status": "verified" if sealed else "not_applicable",
                "snapshot_kind": "closing", "observed_at": signal_date + "T15:00:00+08:00",
                "available_at": collected, "provider_data_status": "verified",
                "source": manifest["datasets"]["limit_up" if sealed else "failed_board"]["source"],
                "change_pct": record[COL["change"]], "last_price": record[COL["last"]],
                "amount": record[COL["amount"]], "float_cap": record[COL["float_cap"]],
                "turnover": record[COL["turnover"]], "first_seal_time": record[COL["first_seal"]],
                "failed_seal_count": record[COL["failed_count"]],
                "limit_statistics": record[COL["limit_stat"]], "industry_provider": record[COL["industry"]],
                **security,
            }
            row["eligibility_status"] = (
                "OUT_OF_SCOPE_NON_MAINBOARD" if not mainboard else
                "VERIFIED_ELIGIBLE" if security["security_status"] == "verified"
                and security["is_st"] is False and security["is_delisting_period"] is False
                and security["no_price_limit"] is False else "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE"
            )
            rows.append(row)
    output = pd.DataFrame(rows)
    if output.candidate_id.duplicated().any():
        raise ValueError("duplicate standardized candidate id")
    closing = output[output.pool_assignment.eq("CLOSING_LIMIT_UP")]
    main_closing = closing[closing.is_mainboard]
    failed_main = output[output.pool_assignment.eq("FAILED_BOARD_OBSERVATION") & output.is_mainboard]
    counts = main_closing.boards.value_counts().sort_index().to_dict()
    closing_unavailable = int(main_closing.eligibility_status.eq(
        "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE").sum())
    failed_verified = int(failed_main.eligibility_status.eq("VERIFIED_ELIGIBLE").sum())
    failed_unavailable = int(failed_main.eligibility_status.eq(
        "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE").sum())
    security_complete = closing_unavailable == 0 and failed_unavailable == 0
    limitations = [
        "the archive was collected after the signal-date close but before the next trading decision",
        "provider pool receipt does not independently prove full-market completeness",
        "no next-session auction, opening, theme, announcement, or execution evidence exists yet",
    ]
    if closing_unavailable:
        limitations.insert(2, "point-in-time security eligibility is incomplete for closing-limit candidates")
    if failed_unavailable:
        limitations.insert(2, "point-in-time security eligibility is incomplete for failed-board observations")
    report = {
        "status": ("CURRENT_POOL_SECURITY_VERIFIED_COMPLETENESS_NOT_INDEPENDENTLY_VERIFIED"
                   if security_complete else "CURRENT_POOL_STANDARDIZED_NOT_FULLY_VERIFIED"),
        "signal_date": signal_date, "available_at": collected,
        "raw_closing_limit_rows": int(len(limit_up)),
        "mainboard_closing_limit_rows": int(len(main_closing)),
        "mainboard_first_board_rows": int(main_closing.boards.eq(1).sum()),
        "mainboard_consecutive_board_rows": int(main_closing.boards.ge(2).sum()),
        "mainboard_board_level_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "non_mainboard_closing_rows": int((~closing.is_mainboard).sum()),
        "mainboard_failed_board_rows": int(len(failed_main)),
        "verified_eligible_closing_rows": int(main_closing.eligibility_status.eq("VERIFIED_ELIGIBLE").sum()),
        "security_unavailable_or_ineligible_closing_rows": closing_unavailable,
        "verified_eligible_failed_board_rows": failed_verified,
        "security_unavailable_or_ineligible_failed_board_rows": failed_unavailable,
        "point_in_time_security_scope_complete": security_complete,
        "main_and_failed_pools_disjoint": not bool(set(main_closing.code) & set(failed_main.code)),
        "provider_pool_status": "verified", "full_market_completeness": "not_independently_verified",
        "entry_authorized": False,
        "limitations": limitations,
    }
    return output, report


def run(limit_path, failed_path, manifest_path, masters, signal_date, output_dir):
    paths = [Path(limit_path), Path(failed_path), Path(manifest_path), *map(Path, masters)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable current-pool snapshot")
    master = pd.concat([pd.read_csv(path, dtype={"code": str}) for path in paths[3:]], ignore_index=True)
    output, report = build(pd.read_csv(paths[0], dtype={COL["code"]: str}),
                           pd.read_csv(paths[1], dtype={COL["code"]: str}),
                           json.loads(paths[2].read_text(encoding="utf-8")), master, signal_date)
    target.mkdir(parents=True, exist_ok=False)
    output.to_csv(target / "all_closing_and_failed_candidates.csv", index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths + [Path(__file__)]}
    with (target / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-up", required=True)
    parser.add_argument("--failed-board", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--security-master", action="append", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.limit_up, args.failed_board, args.manifest, args.security_master,
                         args.signal_date, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
