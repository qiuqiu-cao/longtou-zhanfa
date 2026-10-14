#!/usr/bin/env python3
"""Point-in-time security eligibility master and survivorship-bias audit.

Historical OHLC being verified does not prove that the historical security
universe, ST status, listing state, or price-limit regime is complete.  This
module keeps those claims separate and refuses to treat a missing interval as a
normal mainboard stock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from leader_system_v3 import zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "data" / "v4" / "security_master_intervals.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
OUTPUT = ROOT / "results" / "v4" / "security_master"
VERSION = "v4-security-master-1"

MASTER_COLUMNS = (
    "code", "name", "valid_from", "valid_to", "exchange", "board",
    "listing_status", "st_status", "price_limit_regime", "listing_date",
    "delisting_date", "source_type", "source_ref", "source_timestamp", "data_status",
)
TRUSTED_SOURCE_TYPES = {"sse_official", "szse_official", "company_announcement", "verified_broker_terminal"}
VERIFIED_STATUSES = {"verified", "verified_import", "verified_terminal_export"}
EXCHANGES = {"SSE", "SZSE"}
BOARDS = {"MAINBOARD", "CHINEXT", "STAR", "BSE", "OTHER"}
LISTING_STATUSES = {"LISTED", "SUSPENDED", "DELISTED"}
ST_STATUSES = {"NORMAL", "ST", "*ST", "DELISTING_RISK"}
LIMIT_REGIMES = {"MAINBOARD_10", "ST_5", "NO_LIMIT", "OTHER", "NONE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_mainboard(code: str) -> bool:
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def interval_id(row: pd.Series) -> str:
    payload = "\x1f".join(str(row.get(field, "")).strip() for field in MASTER_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _date_string(series: pd.Series, allow_blank: bool = False) -> pd.Series:
    text = series.astype("string").str.strip()
    blank = text.isna() | text.eq("") | text.str.lower().eq("nan")
    parsed = pd.to_datetime(text.mask(blank), errors="coerce")
    if ((~blank) & parsed.isna()).any() or (not allow_blank and blank.any()):
        raise ValueError("security master contains invalid required date")
    return parsed.dt.strftime("%Y-%m-%d")


def validate_master(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MASTER_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"security master missing columns: {missing}")
    checked = frame.loc[:, MASTER_COLUMNS].copy()
    checked["code"] = checked["code"].map(zfill_code)
    for field in ["name", "exchange", "board", "listing_status", "st_status", "price_limit_regime",
                  "source_type", "source_ref", "source_timestamp", "data_status"]:
        checked[field] = checked[field].astype("string").str.strip()
        if checked[field].isna().any() or checked[field].eq("").any():
            raise ValueError(f"security master field {field} cannot be blank")
    for field in ["exchange", "board", "listing_status", "st_status", "price_limit_regime"]:
        checked[field] = checked[field].str.upper()
    checked["source_type"] = checked["source_type"].str.lower()
    checked["data_status"] = checked["data_status"].str.lower()
    if (~checked["exchange"].isin(EXCHANGES)).any():
        raise ValueError("invalid exchange")
    if (~checked["board"].isin(BOARDS)).any():
        raise ValueError("invalid board")
    if (~checked["listing_status"].isin(LISTING_STATUSES)).any():
        raise ValueError("invalid listing_status")
    if (~checked["st_status"].isin(ST_STATUSES)).any():
        raise ValueError("invalid st_status")
    if (~checked["price_limit_regime"].isin(LIMIT_REGIMES)).any():
        raise ValueError("invalid price_limit_regime")
    checked["valid_from"] = _date_string(checked["valid_from"])
    checked["valid_to"] = _date_string(checked["valid_to"], allow_blank=True)
    checked["listing_date"] = _date_string(checked["listing_date"])
    checked["delisting_date"] = _date_string(checked["delisting_date"], allow_blank=True)
    checked["source_timestamp"] = checked["source_timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat() if not pd.isna(pd.to_datetime(value, errors="coerce")) else None
    )
    if checked["source_timestamp"].isna().any():
        raise ValueError("source_timestamp must be parseable")
    start = pd.to_datetime(checked["valid_from"])
    end = pd.to_datetime(checked["valid_to"])
    listing = pd.to_datetime(checked["listing_date"])
    delisting = pd.to_datetime(checked["delisting_date"])
    if (start < listing).any():
        raise ValueError("valid_from cannot precede listing_date")
    if (end.notna() & (end < start)).any():
        raise ValueError("valid_to cannot precede valid_from")
    if (delisting.notna() & (delisting < listing)).any():
        raise ValueError("delisting_date cannot precede listing_date")
    if (checked["listing_status"].eq("DELISTED") & delisting.isna()).any():
        raise ValueError("DELISTED interval requires delisting_date")
    st_named = checked["name"].str.upper().str.contains(r"(?:^|\*)ST", regex=True)
    if (st_named & checked["st_status"].eq("NORMAL")).any():
        raise ValueError("ST name conflicts with NORMAL st_status")
    if (checked["st_status"].isin(["ST", "*ST"]) & ~checked["price_limit_regime"].isin(["ST_5", "NO_LIMIT"])).any():
        raise ValueError("ST interval must use ST_5 or explicit NO_LIMIT regime")
    if (checked["board"].eq("MAINBOARD") & ~checked["code"].map(expected_mainboard)).any():
        raise ValueError("MAINBOARD interval has incompatible code prefix")
    if (checked["board"].ne("MAINBOARD") & checked["price_limit_regime"].eq("MAINBOARD_10")).any():
        raise ValueError("MAINBOARD_10 cannot be assigned to another board")
    checked["source_verified"] = checked["source_type"].isin(TRUSTED_SOURCE_TYPES) & checked["data_status"].isin(VERIFIED_STATUSES)
    checked["interval_id"] = checked.apply(interval_id, axis=1)
    if checked["interval_id"].duplicated().any():
        raise ValueError("security master contains exact duplicate intervals")
    for code, part in checked.groupby("code", sort=False):
        ordered = part.assign(_start=pd.to_datetime(part["valid_from"]), _end=pd.to_datetime(part["valid_to"])).sort_values("_start")
        previous_end = None
        open_ended_seen = False
        for _, row in ordered.iterrows():
            if open_ended_seen:
                raise ValueError(f"security master has interval after open-ended interval for {code}")
            if previous_end is not None and row["_start"] <= previous_end:
                raise ValueError(f"security master intervals overlap for {code}")
            if pd.isna(row["_end"]):
                open_ended_seen = True
            else:
                previous_end = row["_end"]
    return checked.sort_values(["code", "valid_from"]).reset_index(drop=True)


def import_master(source: Path, master: Path = DEFAULT_MASTER) -> dict[str, Any]:
    incoming = validate_master(pd.read_csv(source, dtype={"code": "string"}))
    if master.exists() and master.stat().st_size:
        existing = validate_master(pd.read_csv(master, dtype={"code": "string"}))
        existing_ids = set(existing["interval_id"].astype(str))
        duplicates = sorted(existing_ids.intersection(incoming["interval_id"].astype(str)))
        if duplicates:
            raise ValueError(f"immutable interval_ids already exist: {duplicates[:5]}")
        keys = ["code", "valid_from"]
        old_keys = set(map(tuple, existing[keys].astype(str).to_numpy()))
        conflicts = [tuple(row) for row in incoming[keys].astype(str).to_numpy() if tuple(row) in old_keys]
        if conflicts:
            raise ValueError(f"security interval start already registered: {conflicts[:5]}")
        combined = validate_master(pd.concat([existing.loc[:, MASTER_COLUMNS], incoming.loc[:, MASTER_COLUMNS]], ignore_index=True))
    else:
        combined = incoming
    master.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(master, index=False, encoding="utf-8-sig")
    return {
        "version": VERSION,
        "imported": int(len(incoming)),
        "verified_imported": int(incoming["source_verified"].sum()),
        "archive_rows": int(len(combined)),
        "master": str(master.resolve()),
        "sha256": sha256_file(master),
    }


def audit_candidates(candidates: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    date_column = "date" if "date" in candidates.columns else "signal_date" if "signal_date" in candidates.columns else None
    if date_column is None:
        raise ValueError("candidate table requires date or signal_date")
    name_column = "name" if "name" in candidates.columns else "stock_name" if "stock_name" in candidates.columns else None
    if "code" not in candidates.columns or "boards" not in candidates.columns:
        raise ValueError("candidate table requires code and boards")
    candidates["code"] = candidates["code"].map(zfill_code)
    candidates["_date"] = pd.to_datetime(candidates[date_column], errors="coerce")
    if candidates["_date"].isna().any():
        raise ValueError("candidate table contains invalid dates")
    candidates["_canonical_date"] = candidates[date_column].astype(str)
    candidates["_canonical_name"] = candidates[name_column].astype(str) if name_column else ""
    if master.empty:
        result = candidates[["_canonical_date", "code", "_canonical_name", "boards"]].copy()
        result.columns = ["date", "code", "name", "boards"]
        result["security_status"] = "UNVERIFIED_POINT_IN_TIME_STATUS"
        result["status_verified"] = False
        result["historically_eligible"] = False
        result["board"] = ""
        result["listing_status"] = ""
        result["st_status"] = ""
        result["price_limit_regime"] = ""
        result["interval_id"] = ""
        result["status_reason"] = "no covering point-in-time security interval"
        return result
    master = master.copy()
    master["_start"] = pd.to_datetime(master["valid_from"])
    master["_end"] = pd.to_datetime(master["valid_to"]).fillna(pd.Timestamp("2262-04-11"))
    by_code = {code: part for code, part in master.groupby("code", sort=False)}
    rows = []
    for _, candidate in candidates.iterrows():
        base = {"date": str(candidate["_canonical_date"]), "code": candidate["code"],
                "name": candidate["_canonical_name"], "boards": candidate["boards"]}
        part = by_code.get(candidate["code"])
        if part is None:
            rows.append({**base, "security_status": "UNVERIFIED_POINT_IN_TIME_STATUS", "status_verified": False,
                         "historically_eligible": False, "board": "", "listing_status": "", "st_status": "",
                         "price_limit_regime": "", "interval_id": "", "status_reason": "code absent from security master"})
            continue
        matches = part[(part["_start"] <= candidate["_date"]) & (part["_end"] >= candidate["_date"])]
        if len(matches) != 1:
            rows.append({**base, "security_status": "UNVERIFIED_POINT_IN_TIME_STATUS", "status_verified": False,
                         "historically_eligible": False, "board": "", "listing_status": "", "st_status": "",
                         "price_limit_regime": "", "interval_id": "", "status_reason": "no unique covering interval"})
            continue
        interval = matches.iloc[0]
        verified = bool(interval["source_verified"])
        if not verified:
            status, reason = "UNVERIFIED_SOURCE", "covering interval lacks trusted verified source"
        elif interval["board"] != "MAINBOARD":
            status, reason = "EXCLUDED_NON_MAINBOARD", f"board={interval['board']}"
        elif interval["listing_status"] != "LISTED":
            status, reason = "EXCLUDED_NOT_LISTED", f"listing_status={interval['listing_status']}"
        elif interval["st_status"] != "NORMAL":
            status, reason = "EXCLUDED_ST_OR_DELISTING_RISK", f"st_status={interval['st_status']}"
        elif interval["price_limit_regime"] == "NO_LIMIT":
            status, reason = "EXCLUDED_NO_PRICE_LIMIT", "listing/new-share no-limit interval"
        elif interval["price_limit_regime"] != "MAINBOARD_10":
            status, reason = "EXCLUDED_OTHER_LIMIT_REGIME", f"price_limit_regime={interval['price_limit_regime']}"
        else:
            status, reason = "VERIFIED_ELIGIBLE", "verified listed normal mainboard 10% regime"
        rows.append({
            **base,
            "security_status": status,
            "status_verified": verified,
            "historically_eligible": status == "VERIFIED_ELIGIBLE",
            "board": interval["board"],
            "listing_status": interval["listing_status"],
            "st_status": interval["st_status"],
            "price_limit_regime": interval["price_limit_regime"],
            "interval_id": interval["interval_id"],
            "status_reason": reason,
        })
    return pd.DataFrame(rows)


def make_report(audit: pd.DataFrame, metadata: dict[str, Any]) -> str:
    counts = audit["security_status"].value_counts().to_dict()
    lines = [
        "# V4 时点证券资格与幸存者偏差审计",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 候选观察：{metadata['candidate_rows']} 条",
        f"- 时点状态已核验：{metadata['verified_rows']} 条（{metadata['verified_coverage']:.1%}）",
        f"- 历史资格可确认：{metadata['eligible_rows']} 条（{metadata['eligible_coverage']:.1%}）",
        f"- 独立代码：{metadata['candidate_codes']}；缺失主表代码：{metadata['missing_codes']}",
        f"- 历史候选母池完整：`{metadata['candidate_universe_complete']}`",
        "- 注意：行情字段已核验不等于历史证券池、ST状态或涨跌幅制度已核验。",
        "",
        "| 状态 | 观察数 |",
        "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "`VERIFIED_ELIGIBLE` 只是时点证券资格的必要条件，不足以单独证明回测无幸存者偏差。",
        "历史候选母池还必须包含当时已上市、后来退市的证券；缺失时点区间不能按NORMAL处理。",
        "在母池或状态覆盖不足前，历史回测继续保持诊断资格。",
        "",
    ])
    return "\n".join(lines)


def run_audit(
    candidates_path: Path = DEFAULT_CANDIDATES,
    master_path: Path = DEFAULT_MASTER,
    output: Path = OUTPUT,
    end_date: str | None = None,
) -> dict[str, Any]:
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    if end_date is not None:
        date_column = "date" if "date" in candidates.columns else "signal_date" if "signal_date" in candidates.columns else None
        if date_column is None:
            raise ValueError("candidate table requires date or signal_date for end-date cutoff")
        cutoff = pd.Timestamp(end_date).normalize()
        dates = pd.to_datetime(candidates[date_column], errors="coerce").dt.normalize()
        if dates.isna().any():
            raise ValueError("candidate table contains invalid dates")
        candidates = candidates.loc[dates.le(cutoff)].copy()
    if "universe_complete" in candidates.columns:
        candidate_universe_complete = bool(
            candidates["universe_complete"].astype("string").str.lower().isin(["true", "1", "yes"]).all()
        )
    else:
        candidate_universe_complete = False
    if master_path.exists() and master_path.stat().st_size:
        master = validate_master(pd.read_csv(master_path, dtype={"code": "string"}))
    else:
        master = pd.DataFrame()
    audit = audit_candidates(candidates, master)
    verified = int(audit["status_verified"].sum())
    eligible = int(audit["historically_eligible"].sum())
    missing_codes = int(audit.loc[audit["security_status"].eq("UNVERIFIED_POINT_IN_TIME_STATUS"), "code"].nunique())
    metadata = {
        "version": VERSION,
        "status": "POINT_IN_TIME_COMPLETE" if (
            verified == len(audit) and eligible == len(audit) and candidate_universe_complete
        ) else "SURVIVORSHIP_BIAS_UNRESOLVED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_rows": int(len(audit)),
        "candidate_codes": int(audit["code"].nunique()),
        "verified_rows": verified,
        "verified_coverage": verified / len(audit) if len(audit) else 0.0,
        "eligible_rows": eligible,
        "eligible_coverage": eligible / len(audit) if len(audit) else 0.0,
        "missing_codes": missing_codes,
        "candidate_universe_complete": candidate_universe_complete,
        "status_counts": {str(k): int(v) for k, v in audit["security_status"].value_counts().items()},
        "candidate_sha256": sha256_file(candidates_path),
        "master_sha256": sha256_file(master_path) if master_path.exists() else None,
        "assurance": "LOCAL_HASHED_POINT_IN_TIME_IMPORT",
        "deployment_effect": "historical gate remains failed unless both point-in-time status and historical universe coverage are complete",
        "historical_end_cutoff": (pd.Timestamp(end_date).strftime("%Y-%m-%d") if end_date else None),
    }
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "security_eligibility_audit.csv"
    json_path = output / "security_eligibility_audit.json"
    md_path = output / "security_eligibility_audit.md"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(make_report(audit, metadata), encoding="utf-8")
    metadata["files"] = {
        "audit": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
        "report": {"path": str(md_path.resolve()), "sha256": sha256_file(md_path)},
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(json_path.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit point-in-time A-share listing/ST/price-limit eligibility")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("source", type=Path)
    imported = sub.add_parser("import")
    imported.add_argument("source", type=Path)
    imported.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    audit = sub.add_parser("audit")
    audit.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    audit.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    audit.add_argument("--output", type=Path, default=OUTPUT)
    audit.add_argument("--end-date", help="optional inclusive historical cutoff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        checked = validate_master(pd.read_csv(args.source, dtype={"code": "string"}))
        result = {"status": "PASS", "rows": len(checked), "verified_rows": int(checked["source_verified"].sum())}
    elif args.command == "import":
        result = import_master(args.source, args.master)
    else:
        result = run_audit(args.candidates, args.master, args.output, args.end_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
