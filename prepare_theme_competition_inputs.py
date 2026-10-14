"""Prepare immutable, source-bound inputs for the existing V150/V114/V151 tools.

Provider categories remain descriptive. Unknown securities and memberships are
retained, and no verified security/full-market status is synthesized. This is a
research input adapter; it neither freezes predictions nor upgrades a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_specific_theme_roles_v150 as roles_v150
import build_current_competition_snapshot_v114 as competition_v114
import theme_competition_archive_v151 as archive_v151
import public_outcome_sources_v293 as outcome_v293
import point_context_v295 as context_v295
import point_context_reconciled_v295 as reconciled_v295


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stamp(value):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("an exact timestamp with timezone is required")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("an exact timestamp with timezone is required")
    return result.tz_convert("Asia/Shanghai")


def file_time(path):
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")


def validate_theme_sources(evidence_path, binding_path, signal_date, cutoff, clock):
    evidence_path, binding_path = Path(evidence_path), Path(binding_path)
    evidence, binding = read(evidence_path), read(binding_path)
    if evidence.get("signal_date") != signal_date or binding.get("signal_date") != signal_date:
        raise ValueError("theme evidence/binding signal date mismatch")
    roles_v150.validate_evidence(evidence, cutoff.isoformat())
    if (binding.get("review_status") != "REVIEWED_SOURCE_TEXT"
            or not str(binding.get("reviewer", "")).strip()
            or not str(binding.get("review_notes", "")).strip()):
        raise ValueError("source text must actually be reviewed; an empty template is insufficient")
    if binding.get("evidence_sha256") != sha(evidence_path):
        raise ValueError("reviewed theme evidence hash mismatch")
    reviewed, recorded = stamp(binding.get("reviewed_at")), stamp(binding.get("recorded_at"))
    if not reviewed <= recorded <= clock <= cutoff:
        raise ValueError("theme review/build was not available by the fixed cutoff")
    if max(file_time(evidence_path), file_time(binding_path)) > cutoff:
        raise ValueError("theme artifacts were formed after the fixed cutoff")
    groups = binding.get("groups")
    if not isinstance(groups, dict) or set(groups) != {g["theme_group_id"] for g in evidence["groups"]}:
        raise ValueError("each primary theme requires its own source binding")
    hashes = {str(p.resolve()): sha(p) for p in (evidence_path, binding_path)}
    for group in evidence["groups"]:
        refs = groups[group["theme_group_id"]]
        if set(refs) != {"membership", "driver", "assessment"}:
            raise ValueError("membership, driver and assessment sources must be bound separately")
        for kind, ref in refs.items():
            path = Path(ref["raw_path"])
            if not path.is_absolute() or not path.is_file() or sha(path) != ref["raw_sha256"]:
                raise ValueError("theme source bytes missing or hash mismatch")
            captured = stamp(ref["captured_at"])
            if not captured <= reviewed or file_time(path) > reviewed:
                raise ValueError("source capture/file formation follows its review")
            # Excerpts are locators in reviewed UTF-8 source text, not automatic
            # semantic verification. Tier/claim assertions still require review.
            text = path.read_text(encoding="utf-8-sig")
            excerpt = ref.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip() or excerpt not in text:
                raise ValueError("reviewed source excerpt not found in bound raw text")
            if kind in {"membership", "driver"}:
                field = "membership_published_at" if kind == "membership" else "driver_first_available_at"
                url_field = "membership_source_url" if kind == "membership" else "driver_source_url"
                if ref.get("url") != group[url_field]:
                    raise ValueError("source URL differs from reviewed theme evidence")
                if ref.get("publication_time_basis") != "REVIEWED_EXACT_TIMESTAMP":
                    raise ValueError("publication time needs explicit reviewed support")
                if stamp(ref.get("published_at")) != stamp(group[field]) or stamp(group[field]) > captured:
                    raise ValueError("source publication/capture clocks inconsistent")
                if kind == "membership" and not all(str(c).zfill(6) in excerpt for c in group["codes"]):
                    raise ValueError("membership excerpt must explicitly contain every declared code")
            elif stamp(group["expectation_assessed_at"]) > captured:
                raise ValueError("expectation assessment was not available in the captured record")
            hashes[str(path.resolve())] = sha(path)
    return evidence, hashes


def frames(pool, signal_date, prepared_at, cutoff, context=None, evidence=None):
    if pool.get("date") != signal_date or not pool.get("capture_complete_within_provider_response"):
        raise ValueError("exact-date complete provider closing response required")
    rows = pool.get("rows")
    if not rows or len({row["code"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique provider closing roster required")
    available = max(stamp(pool["available_at"]), stamp(prepared_at)).isoformat()
    observed = signal_date + "T15:00:00+08:00"
    if stamp(observed) > stamp(available):
        raise ValueError("closing observation is later than its availability")
    categories = {}
    if context is not None:
        scope = context["scope"]
        if scope.get("price_signal_date") != signal_date or set(scope["codes"]) != {r["code"] for r in rows}:
            raise ValueError("context date or declared roster differs from closing pool")
        items = context["candidates"]
        if (len(items) != len(rows) or len({r["code"] for r in items}) != len(items)
                or {r["code"] for r in items} != {r["code"] for r in rows}):
            raise ValueError("context must preserve the entire unique candidate roster")
        categories = {r["code"]: r.get("categories") for r in items}
    base, intake = [], []
    for row in sorted(rows, key=lambda r: r["code"]):
        code, boards = row["code"], row["boards"]
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or type(boards) is not int or boards < 1:
            raise ValueError("invalid code or board height")
        base.append({
            "candidate_id": f"{signal_date}:{code}", "signal_date": signal_date,
            "code": code, "name": row["name"], "boards": boards,
            "board_group": str(boards) if boards < 5 else "5+",
            "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
            "is_st": None, "is_delisting_period": None,
            "security_status": "UNVERIFIED_NAME_FILTER_ONLY",
            "limit_up_status": "PROVIDER_REPORTED_INDEPENDENT_VERIFICATION_REQUIRED",
            "identity_status": "UNVERIFIED_PROVIDER_CODE_NAME_ONLY",
            "observed_at": observed, "available_at": available,
            "amount": row.get("amount"), "turnover": row.get("turnover_pct"),
            "first_seal_time": row.get("seal_first"), "failed_seal_count": row.get("reported_open_count"),
        })
        category = categories.get(code)
        category_time = stamp(category["available_at"]) if category else None
        usable = bool(category_time is not None and category_time <= cutoff
                      and category_time.date().isoformat() == signal_date
                      and (cutoff-category_time).total_seconds() <= 86400)
        intake.append({
            "signal_date": signal_date, "code": code, "name": row["name"], "boards": boards,
            "is_relay_candidate": boards >= 2,
            "provider_categories_json": json.dumps(category, ensure_ascii=False) if category else None,
            "provider_category_usable_for_description": usable,
            "provider_categories_verify_active_theme": False,
            "provider_same_height_codes": json.dumps(sorted(r["code"] for r in rows if r["boards"] == boards and r["code"] != code)),
            "provider_higher_height_codes": json.dumps(sorted(r["code"] for r in rows if r["boards"] > boards)),
            "model_score": None, "model_score_source": None,
        })
    pool_frame = pd.DataFrame(base)
    pool_manifest = {
        "signal_date": signal_date, "mainboard_closing_limit_rows": len(base),
        "complete_within_archived_roster": True,
        "full_market_completeness": "UNVERIFIED_PROVIDER_RESPONSE_ONLY",
        "scope": "PROVIDER_MAINBOARD_PREFIX_NAME_NON_ST_CLOSING_LIMIT_POOL",
        "amount_unit": "provider_amount_native", "turnover_unit": "provider_turnover_pct",
        "first_seal_time_unit": "provider_HHMMSS",
        "security_status_verified": False, "independent_limit_prices_verified": False,
        "recorded_at": prepared_at, "available_at": available,
    }
    verified_roles = (roles_v150.build(pool_frame, evidence, cutoff.isoformat()) if evidence is not None
                      else pd.DataFrame(columns=roles_v150.OUTPUT_COLUMNS))
    if len(verified_roles):
        # The complete derived record becomes available when this adapter runs.
        verified_roles["evidence_published_at"] = verified_roles.evidence_published_at.map(
            lambda value: max(stamp(value), stamp(prepared_at)).isoformat())
    all_roles = pd.DataFrame({"signal_date": signal_date, "code": pool_frame.code}).merge(
        verified_roles.drop(columns="signal_date"), on="code", how="left", validate="one_to_one")
    all_roles["theme_status"] = all_roles.theme_status.fillna("UNAVAILABLE")
    theme_manifest = {"signal_date": signal_date, "candidate_rows": len(all_roles),
                      "recorded_at": prepared_at, "source_text_review_required": True}
    if stamp(available) <= cutoff and all_roles.theme_status.isin(competition_v114.VERIFIED_THEME_STATUSES).any():
        competition, competition_manifest = competition_v114.build(
            pool_frame, pool_manifest, all_roles, theme_manifest, cutoff.isoformat())
    else:
        competition = pd.DataFrame()
        competition_manifest = {"signal_date": signal_date, "closing_candidate_count": len(base),
                                "verified_theme_candidate_count": 0,
                                "status": ("NOT_AVAILABLE_BY_FIXED_CUTOFF" if stamp(available) > cutoff
                                           else "NO_VERIFIED_THEME_ROWS"),
                                "unknown_candidate_codes": pool_frame.code.tolist(), "coverage": 0.0}
    if len(competition):
        # V114 computes the existing roles/rival ordering. The provider pool
        # has not passed independent limit/security verification, so its
        # verified status must not be inherited by this descriptive adapter.
        competition["data_status"] = "UNVERIFIED_PROVIDER_CLOSING_STRUCTURE"
        competition["evidence_status"] = "REVIEWED_THEME_WITH_PROVIDER_REPORTED_POOL"
        competition["source"] = "source_bound_reviewed_themes+provider_reported_closing_pool"
        competition_manifest["status"] = "DESCRIPTIVE_PROVIDER_SCOPE_COMPETITION"
        competition_manifest["independent_closing_verification"] = False
    if competition.empty:
        competition = pd.DataFrame(columns=["signal_date", "code"] + archive_v151.FACTOR_COLUMNS)
    factors = pool_frame.merge(competition[["code"] + archive_v151.FACTOR_COLUMNS], on="code", how="left", validate="one_to_one")
    factors["theme_factor_status"] = factors.theme_group_id.map(lambda x: "UNAVAILABLE" if pd.isna(x) else "REVIEWED_WITHIN_PROVIDER_SCOPE")
    factors["theme_membership_known"] = factors.theme_group_id.notna().astype("boolean")
    factors.loc[factors.theme_group_id.isna(), "theme_membership_known"] = pd.NA
    factors["missing_theme_is_negative"] = False
    factors["eligible_for_leader_factor_training"] = False
    factors["entry_authorized"] = False
    return pool_frame, pool_manifest, all_roles, theme_manifest, pd.DataFrame(intake), competition, competition_manifest, factors


def prepare(pool_archive, signal_date, protocol_path, output, context_archive=None,
            theme_evidence=None, theme_bindings=None):
    output, pool_archive, protocol_path = Path(output), Path(pool_archive), Path(protocol_path)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable research inputs")
    prepared_at = now()
    clock = stamp(prepared_at)
    protocol = read(protocol_path)
    if (protocol.get("prospective_information_mode") != "B_FIXED_EVENING"
            or protocol.get("prospective_knowledge_cutoff_local") != "16:30:00 Asia/Shanghai"):
        raise ValueError("unsupported preregistered information mode/cutoff")
    registered = stamp(protocol["registered_at"])
    if registered > clock:
        raise ValueError("protocol was not registered before preparation")
    cutoff = stamp(signal_date + "T16:30:00+08:00")
    manifest, pools, _daily, failures = outcome_v293.reconstruct(pool_archive)
    if stamp(manifest["recorded_at"]) > clock:
        raise ValueError("source archive formation follows preparation")
    calendar = manifest["calendar"]
    if calendar.get("status") != "verified" or signal_date not in calendar["sessions"]:
        raise ValueError("signal date absent from the verified trading calendar")
    target_date = outcome_v293.next_session(calendar, signal_date)
    pool = pools.get((signal_date, "limit_up"))
    if pool is None:
        raise ValueError("requested signal-day closing pool unavailable")
    # V293 already verifies every raw receipt and reconstructs normalized bytes.
    hashes = {str(p.resolve()): sha(p) for p in [protocol_path, pool_archive/"manifest.json", pool_archive/"normalized.json"]}
    context = None
    if context_archive:
        context_archive = Path(context_archive)
        cm = read(context_archive/"manifest.json")
        context = (reconciled_v295.reconstruct(context_archive) if cm["version"] == reconciled_v295.VERSION
                   else context_v295.reconstruct(context_archive))
        if (Path(context["scope"]["source_path"]).resolve() != pool_archive.resolve()
                or context["scope"]["manifest_sha256"] != sha(pool_archive/"manifest.json")):
            raise ValueError("context is not bound to this exact pool source archive")
        for p in [context_archive/"manifest.json", context_archive/"context.json"]:
            hashes[str(p.resolve())] = sha(p)
    if bool(theme_evidence) != bool(theme_bindings):
        raise ValueError("theme evidence and reviewed source bindings must be provided together")
    evidence = None
    # Actual preparation clock is read after source reconstruction.
    prepared_at = now()
    clock = stamp(prepared_at)
    if theme_evidence:
        if registered >= stamp(signal_date + "T09:15:00+08:00"):
            raise ValueError("prospective protocol must be registered before the signal session")
        evidence, evidence_hashes = validate_theme_sources(theme_evidence, theme_bindings, signal_date, cutoff, clock)
        hashes.update(evidence_hashes)
    values = frames(pool, signal_date, prepared_at, cutoff, context, evidence)
    pool_frame, pm, all_roles, tm, intake, competition, comp_manifest, factors = values
    output.mkdir(parents=True, exist_ok=False)
    def dump(name, value):
        with (output/name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    for name, frame in [("closing_pool.csv", pool_frame), ("candidate_theme_roles.csv", all_roles),
                        ("candidate_theme_intake.csv", intake), ("candidate_competition_snapshot.csv", competition),
                        ("candidate_theme_competition.csv", factors)]:
        frame.to_csv(output/name, index=False, encoding="utf-8-sig")
    dump("closing_pool_manifest.json", pm)
    dump("theme_manifest.json", tm)
    dump("competition_manifest.json", comp_manifest)
    dump("specific_theme_evidence.draft.json", {"schema_version": roles_v150.VERSION, "signal_date": signal_date, "groups": []})
    dump("theme_source_bindings.draft.json", {
        "signal_date": signal_date, "recorded_at": None, "reviewed_at": None,
        "review_status": "UNREVIEWED", "reviewer": None, "review_notes": None,
        "evidence_sha256": None, "groups": {},
    })
    completed_at = now()
    if theme_evidence and stamp(completed_at) > cutoff:
        dump("failure.json", {"status": "INCOMPLETE_LATE_DERIVED_RECORD", "recorded_at": completed_at,
                              "knowledge_cutoff": cutoff.isoformat(), "prediction_use_allowed": False})
        raise ValueError("complete derived archive was formed after the fixed cutoff")
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    for module in [roles_v150, competition_v114, archive_v151, outcome_v293, context_v295, reconciled_v295]:
        hashes[str(Path(module.__file__).resolve())] = sha(module.__file__)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    receipt = {
        "status": "SOURCE_BOUND_RESEARCH_INPUTS_PREPARED", "recorded_at": completed_at,
        "available_at": completed_at, "preparation_started_at": prepared_at,
        "signal_date": signal_date, "target_date": target_date,
        "knowledge_cutoff": cutoff.isoformat(), "information_mode": "B_FIXED_EVENING",
        "prepared_by_cutoff": stamp(completed_at) <= cutoff,
        "protocol_registered_before_signal": registered < stamp(signal_date + "T09:15:00+08:00"),
        "provider_pool_count": len(pool_frame), "relay_candidate_count": int((pool_frame.boards >= 2).sum()),
        "reviewed_theme_rows": int(all_roles.theme_status.isin(competition_v114.VERIFIED_THEME_STATUSES).sum()),
        "unknown_theme_codes": factors.loc[factors.theme_membership_known.isna(), "code"].tolist(),
        "official_full_market": False, "v151_ready": False,
        "v151_blockers": ["exchange security/identity status unverified", "independent closing limit-price verification missing", "full-market completeness unverified"],
        "original_source_failures": failures, "source_hashes": hashes, "artifact_hashes": artifacts,
        "new_settled_forward_samples": 0, "performance_experiment_started": False,
        "model_upgraded": False, "old_prediction_modified": False,
        "limitation": "Publication assertions and factual tiers require actual source review. Categories never supply verified active themes. Derived availability includes actual preparation time. This adapter does not infer securities verification or satisfy V151's training gate.",
    }
    dump("manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pool-archive", "signal-date", "protocol", "output"]:
        parser.add_argument("--"+name, required=True)
    for name in ["context-archive", "theme-evidence", "theme-bindings"]:
        parser.add_argument("--"+name)
    args = parser.parse_args()
    result = prepare(args.pool_archive, args.signal_date, args.protocol, args.output,
                     args.context_archive, args.theme_evidence, args.theme_bindings)
    print(json.dumps({k: result[k] for k in ["status", "recorded_at", "signal_date", "target_date", "provider_pool_count", "relay_candidate_count", "reviewed_theme_rows", "prepared_by_cutoff", "v151_ready"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
