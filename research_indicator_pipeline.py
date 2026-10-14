"""Source-bound research observations; never write production plans or scores.

One immutable output contains the full intake, independent peer freeze, seal
observations and next-session confirmation. Historical data remain diagnostic.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("explicit timestamp timezone required")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def formation(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def load_intake(directory):
    """Keep all original rows and validate their complete ex-self peer lists."""
    directory = Path(directory).resolve()
    path, meta_path = directory / "candidate_theme_intake.csv", directory / "manifest.json"
    meta = read(meta_path)
    if meta.get("artifact_hashes", {}).get(path.name) != sha(path):
        raise ValueError("intake hash does not match original manifest")
    if meta.get("artifact_kind") == "SIGNAL_DAY_RESEARCH_CAPTURE":
        import capture_signal_theme_inputs as capture
        checked = capture.verify_archive(directory)
        meta = {**meta, "provider_pool_count": meta.get("provider_closing_rows"),
                "prepared_by_cutoff": checked["prepared_by_cutoff"]}
    if stamp(meta["recorded_at"]) > stamp(now()):
        raise ValueError("future intake receipt")
    if max(stamp(formation(path)), stamp(formation(meta_path))) > stamp(now()):
        raise ValueError("intake files were formed in the future")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != meta.get("provider_pool_count"):
        raise ValueError("entire original provider intake required")
    by_code = {}
    for row in rows:
        code = row.get("code", "")
        if len(code) != 6 or not code.isdigit() or code in by_code:
            raise ValueError("invalid or duplicate intake code")
        if row.get("signal_date") != meta.get("signal_date"):
            raise ValueError("intake signal date differs from receipt")
        boards = int(row["boards"])
        if boards < 1 or str(boards) != row["boards"]:
            raise ValueError("invalid original board height")
        by_code[code] = row
    for code, row in by_code.items():
        expected = sorted(c for c, peer in by_code.items()
                          if c != code and peer["boards"] == row["boards"])
        if json.loads(row["provider_same_height_codes"]) != expected:
            raise ValueError("peer denominator changed or incomplete")
    return path, meta_path, meta, rows


def registered_scope(path, meta, rows):
    if path is None:
        return {"status": "NO_REGISTERED_FORECAST_ATTACHED", "codes": []}
    path = Path(path).resolve()
    value = read(path)
    plan = value.get("plan", {})
    if (plan.get("signal_date") != meta["signal_date"]
            or plan.get("target_date") != meta["target_date"]):
        raise ValueError("original registry signal/target mismatch")
    candidates = plan.get("candidates", [])
    codes = [row["code"] for row in candidates]
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("original registry empty or duplicated")
    by_code = {row["code"]: row for row in rows}
    for candidate in candidates:
        code = candidate["code"]
        if code not in by_code or int(by_code[code]["boards"]) != candidate["boards"]:
            raise ValueError("original candidate/board missing from intake")
    if set(codes) != {row["code"] for row in rows if int(row["boards"]) >= 2}:
        raise ValueError("all original relay candidates must be retained")
    return {"status": "ORIGINAL_REGISTRY_UNCHANGED", "path": str(path),
            "sha256": sha(path), "codes": codes, "candidate_count": len(codes)}


def prepare_roster(intake_directory, calendar_path, registered=None):
    from research.indicator_observations.open_strength import digest, freeze_intake
    intake, manifest, meta, rows = load_intake(intake_directory)
    calendar = read(calendar_path)
    original = registered_scope(registered, meta, rows)
    frozen = freeze_intake(intake, meta["signal_date"], meta["target_date"], calendar,
                           candidate_codes=[row["code"] for row in rows])
    frozen["intake_manifest_path"] = str(manifest)
    frozen["intake_manifest_sha256"] = sha(manifest)
    frozen["intake_recorded_at"] = meta["recorded_at"]
    frozen["calendar_path"] = str(Path(calendar_path).resolve())
    frozen["calendar_file_sha256"] = sha(calendar_path)
    frozen["registered_forecast"] = original
    frozen["information_mode"] = "INDEPENDENT_RESEARCH_PEERS_FROZEN_AT_ACTUAL_TIME"
    frozen["original_signal_cutoff_eligible"] = bool(meta.get("prepared_by_cutoff"))
    frozen["reviewed_theme_rows"] = meta.get("reviewed_theme_rows")
    frozen["provider_scope_only"] = True
    frozen["production_forecast_modified"] = False
    frozen["new_settled_forward_samples"] = 0
    frozen["original_group_sha256"] = digest({k: v for k, v in frozen.items()
                                               if k != "original_group_sha256"})
    return frozen


def load_roster(path, expected_hash, calendar_path):
    from research.indicator_observations.open_strength import validate_frozen
    if not expected_hash or sha(path) != expected_hash:
        raise ValueError("independent roster hash required and must match")
    frozen = read(path)
    validate_frozen(frozen, read(calendar_path))
    if sha(calendar_path) != frozen["calendar_file_sha256"]:
        raise ValueError("explicit calendar differs from original frozen calendar")
    intake, manifest, meta, rows = load_intake(Path(frozen["original_roster_path"]).parent)
    if (sha(intake) != frozen["original_roster_sha256"]
            or sha(manifest) != frozen["intake_manifest_sha256"]):
        raise ValueError("original intake or manifest changed after peer freeze")
    if max(stamp(formation(intake)), stamp(formation(manifest))) > stamp(frozen["frozen_at"]):
        raise ValueError("original intake files were formed after peer freeze")
    by_code = {row["code"]: row for row in rows}
    groups = frozen.get("groups", [])
    codes = [group.get("code") for group in groups]
    if (len(codes) != len(set(codes)) or set(codes) != set(by_code)
            or frozen.get("original_roster_codes") != sorted(by_code)
            or frozen.get("original_roster_count") != len(by_code)
            or frozen["signal_date"] != meta["signal_date"] or frozen["target_date"] != meta["target_date"]):
        raise ValueError("frozen groups do not preserve the complete original intake")
    for group in groups:
        row = by_code[group["code"]]
        if (group["boards"] != int(row["boards"]) or group.get("name") != row["name"]
                or group["original_peer_codes"] != json.loads(row["provider_same_height_codes"])):
            raise ValueError("frozen group differs from original source facts")
    if frozen.get("registered_forecast", {}).get("path"):
        old = frozen["registered_forecast"]
        if sha(old["path"]) != old["sha256"]:
            raise ValueError("original registered forecast changed")
        checked = registered_scope(old["path"], meta, rows)
        if checked != old:
            raise ValueError("original registered scope changed")
    if stamp(frozen["frozen_at"]) > stamp(now()) or stamp(formation(path)) > stamp(now()):
        raise ValueError("peer freeze file is from the future")
    return frozen


def seal_observations(frozen, csv_path, report_path, derived_at, signal_archive=None):
    from research.indicator_observations.seal_quality import (
        compute_seal_quality, load_v275_observations)
    if signal_archive:
        from research.indicator_observations.seal_quality import load_signal_capture_observations
        observations = load_signal_capture_observations(signal_archive, signal_date=frozen["signal_date"])
    elif csv_path:
        observations = load_v275_observations(csv_path, report_path,
                                             project_root=ROOT, signal_date=frozen["signal_date"])
    else:
        observations = {}
    derived_at = now()
    source_path = Path(frozen["original_roster_path"])
    source_bytes = source_path.read_bytes()
    output = []
    for row in frozen["groups"]:
        group = {"signal_date": frozen["signal_date"], "boards": row["boards"],
                 "codes": sorted([row["code"], *row["original_peer_codes"]]),
                 "frozen_at": frozen["frozen_at"], "source_path": str(source_path),
                 "source_sha256": frozen["original_roster_sha256"],
                 "source_bytes": source_bytes, "source_format": "INTAKE_CSV",
                 "archive_formed_at": formation(source_path)}
        value = compute_seal_quality(
            row["code"], group, observations, signal_date=frozen["signal_date"],
            target_date=frozen["target_date"],
            knowledge_cutoff=frozen["signal_date"] + "T16:30:00+08:00",
            derived_at=derived_at, allow_retrospective=True)
        value["registered_forecast_candidate"] = row["code"] in frozen["registered_forecast"]["codes"]
        output.append(value)
    return output


def prior_closes(archive, signal_date):
    """Reconstruct original V293 daily bytes, using only signal-day closes."""
    import public_outcome_sources_v293 as sources
    manifest = read(Path(archive) / "manifest.json")
    if manifest.get("artifact_kind") == "SIGNAL_DAY_RESEARCH_CAPTURE":
        import capture_signal_theme_inputs as capture
        capture.verify_archive(archive)
        if manifest["signal_date"] != signal_date:
            raise ValueError("signal archive previous close date mismatch")
        daily, unavailable = {}, list(manifest["failures"])
        start = (datetime.strptime(signal_date, "%Y-%m-%d") - timedelta(days=25)).strftime("%Y-%m-%d")
        for entry in manifest["requests"]:
            if entry["kind"] != "daily" or entry.get("status") != "RAW_JSON_RECEIVED":
                continue
            code = entry["code"]
            symbol = ("sh" if code.startswith("6") else "sz") + code
            if (entry["url"] != sources.DAY_URL or entry["parameters"] !=
                    {"param": f"{symbol},day,{start},{signal_date},40,"}):
                raise ValueError("signal archive daily request identity changed")
            try:
                daily[code] = sources.parse_daily(read(entry["raw_path"]), code, start, signal_date)
            except (ValueError, KeyError, TypeError) as exc:
                unavailable.append({"code": code, "reason": str(exc)})
    else:
        manifest, _, daily, unavailable = sources.reconstruct(archive)
    entries = {entry["code"]: entry for entry in manifest["requests"]
               if entry["kind"] == "daily" and entry.get("status") == "RAW_JSON_RECEIVED"}
    rows = []
    for code, value in daily.items():
        day = value.get("days", {}).get(signal_date)
        if day is None or value.get("adjusted") is not False:
            continue
        entry = entries[code]
        rows.append({"code": code, "date": signal_date, "close": day["close"] / 100,
                     "adjustment": "none", "price_unit": "yuan", "independent": True,
                     "source": "TENCENT_UNADJUSTED_DAILY_ORIGINAL_V293",
                     "raw_path": entry["raw_path"], "raw_sha256": entry["raw_sha256"],
                     "available_at": max(stamp(entry["available_at"]), stamp(formation(entry["raw_path"])),
                                         stamp(now())).isoformat(),
                     "binding_status": "RAW_FIELDS_REBUILT"})
    return rows, {"archive": str(Path(archive).resolve()), "manifest_sha256": sha(Path(archive) / "manifest.json"),
                  "usable_signal_closes": len(rows), "original_source_failures": unavailable}


def write_report(output, frozen, seals, opened, metadata):
    """Independent observation appendix, with no ranking or trading signals."""
    seal_by_code = {row["code"]: row for row in seals}
    open_by_code = {row["code"]: row for row in opened}
    combined = []
    columns = ["code", "name", "boards", "registered_forecast_candidate",
               "first_seal_percentile_same_height_ex_self", "failed_seal_count_reported",
               "last_seal_time_reported", "open_duration_trading_minutes",
               "seal_stable_duration_trading_minutes", "seal_status", "seal_point_in_time_eligible",
               "own_return_0935_pct", "peer_return_0935_median_pct", "relative_strength_0935_pp",
               "original_peer_count", "observed_peer_count", "missing_peer_codes", "open_status"]
    for group in frozen["groups"]:
        code = group["code"]
        seal, opening = seal_by_code[code], open_by_code[code]
        row = {"code": code, "name": group.get("name"), "boards": group["boards"],
               "registered_forecast_candidate": code in frozen["registered_forecast"]["codes"],
               "seal_status": seal["status"], "seal_point_in_time_eligible": seal["point_in_time_eligible"],
               "open_status": opening["status"]}
        for field in columns[4:9]:
            row[field] = seal.get(field)
        for field in columns[11:17]:
            row[field] = opening.get(field)
        row["missing_peer_codes"] = json.dumps(opening["missing_peer_codes"])
        combined.append(row)
    with (output / "all_candidate_observations.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined)
    summary = {"status": "RESEARCH_CALCULATORS_COMPLETE_EFFECT_NOT_EVALUATED",
               "recorded_at": now(), "signal_date": frozen["signal_date"],
               "target_date": frozen["target_date"], "research_roster_count": len(frozen["groups"]),
               "original_registered_count": len(frozen["registered_forecast"]["codes"]),
               "new_settled_forward_samples": 0, "accuracy_change": None,
               "production_model_changed": False, "historical_replay_is_forward": False,
               "numeric_experiment_started": False, "theme_specific_members_verified": frozen["reviewed_theme_rows"],
               "theme_status": ("WAIT_COMPLETE_REVIEWED_PIT_MEMBERSHIP_AND_COMMON_DRIVER" if frozen["reviewed_theme_rows"] == 0
                                else "SEE_ORIGINAL_BOUND_THEME_INPUTS_TRAINING_GATE_UNCHANGED"),
               "seal_percentile_available_diagnostic_rows": sum(row["first_seal_percentile_same_height_ex_self"] is not None for row in seals),
               "seal_original_count_available_rows": sum(row["failed_seal_count_reported"] is not None for row in seals),
               "seal_point_in_time_eligible_rows": sum(row["point_in_time_eligible"] for row in seals),
               "open_known_strength_rows": sum(row["relative_strength_0935_pp"] is not None for row in opened),
               "open_waiting_rows": sum(row["status"] == "WAIT_TARGET_OBSERVATION" for row in opened),
               "source_metadata": metadata,
               "artifact_hashes": {name: sha(output / name) for name in
                                   ["research_roster.json", "seal_observations.json", "open_observations.json", "all_candidate_observations.csv"]},
               "program_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                                  [Path(__file__), ROOT / "research/indicator_observations/seal_quality.py",
                                   ROOT / "research/indicator_observations/open_strength.py"]}}
    dump(output / "report.json", summary)
    lines = ["# 三类指标研究链路交付", "",
             f"实际生成时间：{summary['recorded_at']}。信号日{frozen['signal_date']}，目标日{frozen['target_date']}。",
             "", f"完整研究池{summary['research_roster_count']}只；原登记{summary['original_registered_count']}只结算分母保持。",
             "封板字段由原响应重建，旧档或新派生均不能倒填原信号日。精确事件时长不足保持未知。",
             "09:35字段只用于目标日独立确认；未知和未发生不算失败，不修改原分数或前5排序。",
             "真实题材沿用原完整细分成员、共同催化及时点门槛；核验状态见原输入档案，已有固定C1/C2不变。",
             "本次完成程序与输入留档，未训练或检验预测增量。新增已结算真实前向样本0，准确率变化不可计算。",
             "", f"封板同组分位可显示{summary['seal_percentile_available_diagnostic_rows']}行（旧档诊断）；原报告开板次数{summary['seal_original_count_available_rows']}行；完整盘后时点可用{summary['seal_point_in_time_eligible_rows']}行。",
             f"09:35有效完整组强度{summary['open_known_strength_rows']}行，待发生{summary['open_waiting_rows']}行。",
             "", "运行方法与后续命令见collection_plan.json；所有产物仅当前独立研究附页。", "",
             "|原登记股票|板位|首封同组分位（越大越晚）|原报告开板次数|最后封板|09:35|",
             "|---|---:|---:|---:|---|---|"]
    for row in combined:
        if row["registered_forecast_candidate"]:
            percentile = row["first_seal_percentile_same_height_ex_self"]
            fraction = f"{percentile:.3f}" if percentile is not None else "未知"
            count = row['failed_seal_count_reported'] if row['failed_seal_count_reported'] is not None else '未知'
            strength = row['relative_strength_0935_pp']
            opening = (f"{strength:.2f}个百分点" if strength is not None else
                       "待发生" if row["open_status"] == "WAIT_TARGET_OBSERVATION" else "未知")
            lines.append(f"|{row['name']} {row['code']}|{row['boards']}|{fraction}|{count}|{row['last_seal_time_reported'] or '未知'}|{opening}|")
    (output / "交付说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(args):
    from research.indicator_observations.open_strength import derive_open_strength, load_bound_observations
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("unique new output required; prior archives cannot be overwritten")
    if args.roster:
        frozen = load_roster(args.roster, args.roster_sha256, args.calendar)
    else:
        frozen = prepare_roster(args.intake_directory, args.calendar, args.registered)
    output.mkdir(parents=True)
    dump(output / "research_roster.json", frozen)
    derived = now()
    seals = seal_observations(frozen, args.seal_csv, args.seal_report, derived,
                              signal_archive=args.signal_archive)
    previous, metadata = prior_closes(args.daily_archive, frozen["signal_date"]) if args.daily_archive else ([], {})
    minute_bundle = load_bound_observations(args.minute_rows) if args.minute_rows else {"records": [], "failures": []}
    minutes = minute_bundle["records"]
    metadata["minute_reconstruction_errors"] = minute_bundle["failures"]
    opened = derive_open_strength(frozen, previous, minutes, read(args.calendar),
                                  knowledge_cutoff=now())
    dump(output / "seal_observations.json", seals)
    dump(output / "open_observations.json", opened)
    dump(output / "collection_plan.json", {
        "research_only": True, "no_forecast_backfill": True,
        "theme_capture_entry": str(ROOT / "capture_signal_theme_inputs.py"),
        "theme_allowed_window": "actual signal date15:00--16:30 Asia/Shanghai; no backdating",
        "theme_existing_parameters": str(ROOT / "research/theme_same_height/signal_capture_repair_20261003T085622520708Z/future_capture_parameters.json"),
        "open_entry": str(Path(__file__)), "open_command": "collect-open",
        "roster_path": str(output / "research_roster.json"), "roster_sha256": sha(output / "research_roster.json"),
        "calendar_path": str(Path(args.calendar).resolve()), "calendar_sha256": sha(args.calendar),
        "target_date": frozen["target_date"], "minutes_required": "actual target-day09:35; explicitly certified bar-end semantics; one attempt per original code",
        "missing_policy": "retain every original candidate and peer; unknown main metric when any required quote missing",
        "promotion_rule": "separate preregistered comparison only after eligible observations; no production integration here"})
    return write_report(output, frozen, seals, opened, metadata)


def collect_signal(args):
    """Existing signal capture plus the new appendix, one actual-day command."""
    import capture_signal_theme_inputs as capture
    capture.validate_window(args.signal, read(args.calendar), now())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("unique new output required")
    output.mkdir(parents=True)
    source_directory = output / "signal_sources"
    result = capture.collect(args.signal, args.calendar, args.protocol, source_directory,
                             args.prior_roster, args.prior_roster_sha256,
                             args.theme_evidence, args.theme_bindings)
    if not (source_directory / "candidate_theme_intake.csv").is_file():
        status = {"status": "SIGNAL_CAPTURE_WITH_GAPS_NO_VALID_NEW_INTAKE", "source_directory": str(source_directory),
                  "actual_source_receipt": result, "new_settled_forward_samples": 0,
                  "model_changed": False, "prediction_modified": False}
        dump(output / "report.json", status)
        return status
    observation = argparse.Namespace(output=str(output / "observations"), roster=None,
        roster_sha256=None, intake_directory=str(source_directory), registered=None,
        calendar=args.calendar, seal_csv=None, seal_report=None,
        signal_archive=str(source_directory), daily_archive=str(source_directory), minute_rows=None)
    return run(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    observation = commands.add_parser("run")
    group = observation.add_mutually_exclusive_group(required=True)
    group.add_argument("--intake-directory")
    group.add_argument("--roster")
    observation.add_argument("--roster-sha256")
    observation.add_argument("--registered")
    observation.add_argument("--calendar", required=True)
    observation.add_argument("--seal-csv")
    observation.add_argument("--seal-report")
    observation.add_argument("--signal-archive")
    observation.add_argument("--daily-archive")
    observation.add_argument("--minute-rows")
    observation.add_argument("--output", required=True)
    collect = commands.add_parser("collect-open")
    for name in ["roster", "roster-sha256", "calendar", "output"]:
        collect.add_argument("--" + name, required=True)
    signal = commands.add_parser("collect-signal")
    for name in ["signal", "calendar", "protocol", "output"]:
        signal.add_argument("--" + name, required=True)
    for name in ["prior-roster", "prior-roster-sha256", "theme-evidence", "theme-bindings"]:
        signal.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args)
    elif args.command == "collect-open":
        from research.indicator_observations.open_strength import collect_target_minutes
        frozen = load_roster(args.roster, args.roster_sha256, args.calendar)
        result = collect_target_minutes(frozen, read(args.calendar), args.output)
    else:
        result = collect_signal(args)
    print(json.dumps({key: result.get(key) for key in
                      ["status", "research_roster_count", "original_registered_count", "new_settled_forward_samples"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
