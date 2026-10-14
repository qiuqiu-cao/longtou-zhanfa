"""Build an immutable terminal-export request pack for missing five-minute cases."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


VERSION = "v49-minute-request-pack-2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_requests(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [(str(item[0]), str(item[1])) for item in payload]
    if (len(cases) != len(set(cases))
            or any(not code.startswith(("sh.", "sz.")) or len(code) != 9 for code, _ in cases)):
        raise ValueError("invalid or duplicate request identities")
    return cases


def successful_keys(roots: Iterable[Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for root in roots:
        manifests = [*root.glob("minute_*/manifest.json"), *root.glob("batch_*/manifest.json")]
        if (root / "manifest.json").exists():
            manifests.append(root / "manifest.json")
        for manifest_path in sorted(set(manifests)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for case in manifest.get("cases", []):
                if case.get("error_code") != "0" or case.get("rows") != 48:
                    continue
                raw = manifest_path.parent / f"{case['code']}_{case['date']}.json"
                if not raw.exists() or sha256_file(raw) != case.get("raw_sha256"):
                    raise ValueError(f"archive hash mismatch: {raw}")
                keys.add((str(case["code"]), str(case["date"])))
    return keys


def build(requests_path: Path, cohort_path: Path, archive_roots: list[Path], output: Path,
          batch_size: int = 100, replay_signals_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    requests = read_requests(requests_path)
    cohort = pd.read_csv(cohort_path, dtype={"code": str, "entry_date": str,
                                             "provider_code": str, "signal_date": str})
    cohort["code"] = cohort["code"].str.zfill(6)
    lookup = {(row.provider_code, row.entry_date): row for row in cohort.itertuples(index=False)}
    reason_by_case: dict[tuple[str, str], str] = {}
    if replay_signals_path is None:
        completed = successful_keys(archive_roots)
        remaining = [case for case in requests if case not in completed]
        requested_total = len(requests)
        for case in remaining:
            reason_by_case[case] = "MISSING_ARCHIVE"
    else:
        replay = json.loads(replay_signals_path.read_text(encoding="utf-8"))
        if not isinstance(replay.get("missing"), list):
            raise ValueError("replay signal file must contain a missing list")
        by_security_day = {(row.code, row.entry_date): row for row in cohort.itertuples(index=False)}
        for item in replay["missing"]:
            key = (str(item.get("code", "")).zfill(6), str(item.get("entry_date", "")))
            row = by_security_day.get(key)
            if row is None:
                raise ValueError(f"replay missing identity is not present in cohort: {key}")
            case = (str(row.provider_code), str(row.entry_date))
            if case in reason_by_case:
                raise ValueError(f"duplicate replay missing identity: {case}")
            reason_by_case[case] = str(item.get("state") or "UNSPECIFIED_REMEDIATION")
        remaining = list(reason_by_case)
        requested_total = len(cohort)

    rows = []
    for index, (provider_code, day) in enumerate(remaining):
        row = lookup.get((provider_code, day))
        if row is None:
            raise ValueError(f"request is not present in cohort: {(provider_code, day)}")
        rows.append({
            "batch": index // batch_size + 1,
            "provider_code": provider_code,
            "code": provider_code[3:],
            "exchange": "SH" if provider_code.startswith("sh.") else "SZ",
            "signal_date": row.signal_date,
            "entry_date": day,
            "boards": row.boards,
            "board_group": row.board_group,
            "remediation_reason": reason_by_case[(provider_code, day)],
        })
    result = pd.DataFrame(rows, columns=["batch", "provider_code", "code", "exchange",
                                         "signal_date", "entry_date", "boards", "board_group",
                                         "remediation_reason"])
    if (replay_signals_path is None and not result.empty
            and not result["board_group"].eq("FIRST_BOARD").all()):
        raise ValueError("V49 missing pack unexpectedly contains non-first-board rows")

    output.mkdir(parents=True)
    result.to_csv(output / "remaining_requests.csv", index=False, encoding="utf-8-sig")
    (output / "remaining_requests.json").write_text(
        json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    by_date = (result.groupby("entry_date", as_index=False).size().rename(columns={"size": "missing_cases"})
               if not result.empty else pd.DataFrame(columns=["entry_date", "missing_cases"]))
    by_date.to_csv(output / "by_date.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["code", "date", "time", "open", "high", "low", "close",
                          "volume", "amount"]).to_csv(output / "import_template.csv", index=False,
                                                       encoding="utf-8-sig")
    readme = f"""# 历史5分钟数据补齐请求包

- 候选总数：{requested_total}
- 已通过当前回放口径：{requested_total - len(remaining)}
- 仍缺：{len(remaining)}
- 缺失板组：{result['board_group'].value_counts().to_dict() if not result.empty else {}}。
- 修复原因：{result['remediation_reason'].value_counts().to_dict() if not result.empty else {}}。

导出未复权5分钟K线，必须包含 `code,date,time,open,high,low,close,volume,amount`。
时间必须声明为K线开始时间或结束时间；成交量必须声明为股或手；成交额必须声明为元或万元。
不要补写停牌/无成交K线，不要把缺失值填0。每个股票交易日应有48根5分钟K线。

示例导入：

`python import_minute_csv_v47.py --input D:\\your_data\\historical_5min.csv --output data\\v49\\imported_5min_batch1 --source 同花顺导出 --interval 5 --timestamp-side end --volume-unit shares --amount-unit yuan --adjustment unadjusted --requests {str((output / 'remaining_requests.json').resolve())}`

导入结果仍标记为 `USER_IMPORTED_NOT_INDEPENDENTLY_VERIFIED`，不会被宣称为真实排队成交。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    manifest = {
        "version": VERSION,
        "status": "DATA_REQUEST_ONLY",
        "created_at": datetime.now().astimezone().isoformat(),
        "requested": requested_total,
        "already_complete": requested_total - len(remaining),
        "remaining": len(remaining),
        "remaining_board_group_counts": result["board_group"].value_counts().to_dict(),
        "date_count": int(result["entry_date"].nunique()) if not result.empty else 0,
        "batch_size": batch_size,
        "batch_count": int(result["batch"].max()) if not result.empty else 0,
        "source_requests_sha256": sha256_file(requests_path),
        "source_cohort_sha256": sha256_file(cohort_path),
        "archive_roots": [str(path.resolve()) for path in archive_roots],
        "artifact_sha256": hashes,
        "claim": "request coverage only; no market data or fills created",
    }
    if replay_signals_path is not None:
        manifest["source_replay_signals_sha256"] = sha256_file(replay_signals_path)
        manifest["remediation_reason_counts"] = result["remediation_reason"].value_counts().to_dict()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--replay-signals", type=Path,
                        help="optional frozen replay signal file; request every unresolved case, including contradictions")
    args = parser.parse_args()
    print(json.dumps(build(args.requests, args.cohort, args.archive_root, args.output,
                           args.batch_size, args.replay_signals), ensure_ascii=False, indent=2))
