"""Source-bound short-term theme analysis; no stock/model prediction changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamps required")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(name):
    name = str(name).strip()
    return name[:-2] if name.endswith("概念") else name


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bind_pair(obj, path_key, hash_key, binding):
    """Optional bindings are mandatory when either member is supplied."""
    if path_key not in obj and hash_key not in obj:
        return None
    if not obj.get(path_key) or not obj.get(hash_key):
        raise ValueError("incomplete source binding: " + path_key)
    path = Path(obj[path_key]).resolve()
    expected = str(obj[hash_key]).lower()
    if not path.is_file() or sha(path) != expected:
        raise ValueError("source hash mismatch: " + path_key)
    binding[str(path)] = expected
    return path


def analyse(theme_report, narrative_inputs, policy):
    """Project signal-day facts and fixed source claims, without accessing labels.

    File binding is checked by run(); this pure function supports synthetic tests.
    Source claims establish narrative occurrence, not consensus/business truth.
    """
    signal = theme_report["signal_date"]
    if narrative_inputs["signal_date"] != signal:
        raise ValueError("narrative and theme signal dates differ")
    cutoff = stamp(narrative_inputs["knowledge_cutoff"])
    if stamp(theme_report["knowledge_cutoff"]) > cutoff:
        raise ValueError("theme input follows analysis cutoff")
    if stamp(policy["defined_at"]) > cutoff:
        raise ValueError("policy follows analysis cutoff")
    if cutoff < stamp(signal + "T15:00:00+08:00"):
        raise ValueError("completed signal session required")
    if theme_report.get("official_full_market") is not False:
        raise ValueError("provider scope must remain explicit")
    aliases = {}
    for canonical, names in policy.get("aliases", {}).items():
        for name in [canonical] + names:
            key = normalize(name)
            if key in aliases and aliases[key] != normalize(canonical):
                raise ValueError("ambiguous theme alias")
            aliases[key] = normalize(canonical)
    canon = lambda name: aliases.get(normalize(name), normalize(name))
    excluded = {normalize(x) for x in policy.get("excluded_names", [])}
    patterns = policy.get("excluded_patterns", [])
    families = policy.get("families", {})
    family_names = {canon(v["canonical"]): v for v in families.values()}
    positive_tokens = policy["positive_tokens"]
    negative_tokens = policy["negative_tokens"]
    if not positive_tokens or not negative_tokens:
        raise ValueError("fixed positive and negative evidence tokens required")
    if any(type(policy[k]) is not int or policy[k] < 1
           for k in ("min_relay", "min_first", "min_ladder_prefix")):
        raise ValueError("fixed positive structure thresholds required")
    claims = []
    active = {}
    for source in narrative_inputs["sources"]:
        obtained, published = stamp(source["obtained_at"]), stamp(source["published_at"])
        if max(obtained, published) > cutoff or published > obtained:
            raise ValueError("future or inconsistent narrative timestamps")
        if published.astimezone(timezone(timedelta(hours=8))).date().isoformat() != signal:
            raise ValueError("active narrative must be published on signal date")
        for claim in source.get("claims", []):
            name, excerpt = canon(claim["theme_name"]), claim["excerpt"]
            if claim["stance"] not in ("ACTIVE", "WEAK", "BACKGROUND", "CAUTION"):
                raise ValueError("unrecognized narrative stance")
            if not isinstance(excerpt, str) or not excerpt.strip():
                raise ValueError("empty narrative excerpt")
            known_terms = {name, normalize(claim["theme_name"])}
            known_terms.update(k for k, v in aliases.items() if v == name)
            if not any(term and term in excerpt for term in known_terms):
                raise ValueError("claim theme is absent from excerpt")
            item = {"theme_name": name, "stance": claim["stance"], "excerpt": excerpt,
                    "source_id": source["source_id"], "url": source["url"],
                    "published_at": source["published_at"], "obtained_at": source["obtained_at"],
                    "text_path": source["text_path"], "text_sha256": source["text_sha256"],
                    "offset": claim.get("offset")}
            claims.append(item)
            if claim["stance"] == "ACTIVE":
                if (not any(token in excerpt for token in positive_tokens)
                        or any(token in excerpt for token in negative_tokens)
                        or any(re.search(r"(?:并未|尚未|没有|并非|未曾|未|不).{0,8}" + re.escape(token), excerpt)
                               for token in positive_tokens)):
                    raise ValueError("ACTIVE excerpt lacks unnegated positive evidence")
                active.setdefault(name, []).append(item)
                for parent, spec in family_names.items():
                    keyword = spec.get("match_active_keyword")
                    if keyword and keyword in name:
                        active.setdefault(parent, []).append(item)
    facts = {}
    for row in theme_report["all_candidates"]:
        code = row["code"]
        if code in facts:
            raise ValueError("duplicate candidate")
        kind = row["pool_kind"]
        boards = row.get("boards")
        if kind not in ("limit_up", "failed_board"):
            raise ValueError("unknown pool kind")
        if kind == "limit_up" and (type(boards) is not int or boards < 1):
            raise ValueError("invalid closing board height")
        if row.get("category_status") not in ("UNKNOWN", "PROVIDER_CURRENT_CATEGORY_ONLY"):
            raise ValueError("explicit category known/unknown status required")
        facts[code] = {"code": code, "name": row.get("name"), "pool_kind": kind,
                       "boards": boards, "category_status": row.get("category_status"),
                       "category_missing_reason": row.get("category_missing_reason"),
                       "original_base": row.get("original_base")}
    merged, rejected = {}, []
    for group in theme_report["groups"]:
        name = canon(group["group_name"])
        lane = group["evidence_lane"]
        if lane == "PROVIDER_SEMANTIC_FAMILY":
            spec = next((s for s in families.values() if s.get("group_id") == group["group_id"]
                         and s.get("source_lane") == lane), None)
            if spec is None:
                rejected.append({"name": name, "reason": "UNREGISTERED_SEMANTIC_FAMILY"})
                continue
            name = canon(spec["canonical"])
        if name in excluded or any(re.search(p, name) for p in patterns):
            rejected.append({"name": name, "reason": "GENERIC_OR_NON_THEME_CATEGORY"})
            continue
        if name not in active:
            rejected.append({"name": name, "reason": "NO_ACTIVE_SIGNAL_DAY_NARRATIVE"})
            continue
        codes = set(group["member_codes"])
        if not codes <= facts.keys():
            raise ValueError("group contains extra candidate")
        item = merged.setdefault(name, {"theme_name": name, "codes": set(), "lanes": set()})
        item["codes"].update(codes)
        item["lanes"].add(lane)
    candidates = []
    parent_of = {}
    for parent, spec in family_names.items():
        for child in spec.get("children", []):
            parent_of[canon(child)] = parent
    identical = {}
    for name, group in sorted(merged.items()):
        key = tuple(sorted(group["codes"]))
        if key in identical:
            identical[key]["equivalent_names"].append(name)
            identical[key]["narrative_evidence"].extend(active[name])
            continue
        closing = [facts[c] for c in key if facts[c]["pool_kind"] == "limit_up"]
        levels = sorted({r["boards"] for r in closing})
        prefix = 0
        while prefix + 1 in levels:
            prefix += 1
        first = [r["code"] for r in closing if r["boards"] == 1]
        relay = [r["code"] for r in closing if r["boards"] >= 2]
        failed = [c for c in key if facts[c]["pool_kind"] == "failed_board"]
        item = {"theme_name": name, "equivalent_names": [name], "member_codes": list(key),
                "evidence_lanes": sorted(group["lanes"]), "first_board_codes": first,
                "relay_codes": relay, "known_failed_codes": failed,
                "first_board_count": len(first), "relay_count": len(relay),
                "board_levels": levels, "ladder_prefix_length": prefix,
                "failed_membership_coverage_complete": not any(r["pool_kind"] == "failed_board"
                    and r["category_status"] == "UNKNOWN" for r in facts.values()),
                "narrative_evidence": active[name], "parent_theme": parent_of.get(name),
                "original_negative_relay_codes": [c for c in relay
                    if (facts[c]["original_base"] or {}).get("predicted_positive") is False]}
        item["structure_qualified"] = (len(relay) >= policy["min_relay"]
            and len(first) >= policy["min_first"] and prefix >= policy["min_ladder_prefix"])
        identical[key] = item
        candidates.append(item)
    # Parent relations are only honored for a declared, present, containing family.
    for item in candidates:
        parent = next((g for g in candidates if item.get("parent_theme") in g["equivalent_names"]), None)
        if parent and not set(item["member_codes"]) <= set(parent["member_codes"]):
            raise ValueError("declared family does not contain child members")
        if parent is None or parent is item:
            item["parent_theme"] = None
    roots = [g for g in candidates if g["parent_theme"] is None]
    dimensions = ("relay_count", "first_board_count", "ladder_prefix_length")
    def dominates(a, b):
        return all(a[d] >= b[d] for d in dimensions) and any(a[d] > b[d] for d in dimensions)
    for item in roots:
        item["dominated_by"] = [g["theme_name"] for g in roots if g is not item and dominates(g, item)]
    frontier = [g for g in roots if not g["dominated_by"]]
    qualified = [g for g in frontier if g["structure_qualified"]]
    closing_unknown = sorted(c for c, r in facts.items() if r["pool_kind"] == "limit_up"
                             and r["category_status"] == "UNKNOWN")
    unmatched = sorted(set(active) - set(merged))
    if closing_unknown or not qualified:
        status, mainline = "UNKNOWN", None
    elif len(frontier) == 1 and len(qualified) == 1:
        status, mainline = "INFERRED_SHORT_TERM_MAINLINE", qualified[0]["theme_name"]
    else:
        status, mainline = "COMPETING", None
    focus = [g for g in candidates if mainline and g["parent_theme"] == mainline]
    accompanying = [g["theme_name"] for g in roots if g["theme_name"] != mainline
                    and g not in frontier]
    warnings = [{"kind": "SOURCE_CAUTION", "evidence": c} for c in claims if c["stance"] == "CAUTION"]
    warnings.append({"kind": "NO_SAME_SCOPE_PREVIOUS_THEME_RECORD",
                     "detail": "本输入只有当日梯队，不能计算相对上日强度退化"})
    if any(r["pool_kind"] == "failed_board" and r["category_status"] == "UNKNOWN" for r in facts.values()):
        warnings.append({"kind": "UNKNOWN_FAILED_MEMBERSHIP", "detail": "炸板归属不完整，不能计算完整封板率"})
    return {"status": status, "signal_date": signal, "knowledge_cutoff": narrative_inputs["knowledge_cutoff"],
            "policy_version": policy["version"], "mainline": mainline,
            "competitive_lines": [g["theme_name"] for g in frontier if g["theme_name"] != mainline],
            "accompanying_lines": accompanying, "focus_subthemes": focus,
            "warnings": warnings, "strength_degradation_assessed": False,
            "pool_counts": {k: theme_report.get("pool_counts", {}).get(k) for k in ("closing", "failed", "relay", "first_board")},
            "original_base_prediction_binding": theme_report.get("original_base_prediction_binding"),
            "candidates": candidates, "excluded_groups": rejected,
            "unresolved": {"closing_category_unknown_codes": closing_unknown,
                "failed_category_unknown_codes": sorted(c for c, r in facts.items()
                    if r["pool_kind"] == "failed_board" and r["category_status"] == "UNKNOWN"),
                "active_narratives_without_pool_membership": unmatched},
            "all_candidates": [facts[c] for c in sorted(facts)], "narrative_claims": claims,
            "narrative_source_metadata": [{key: s[key] for key in (
                "source_id", "url", "original_url", "title", "publisher", "author", "source_type", "source_kind",
                "original_verified", "published_at", "obtained_at", "text_path", "text_sha256",
                "raw_html_path", "raw_html_sha256", "manifest_path", "manifest_sha256") if key in s}
                for s in narrative_inputs["sources"]],
            "scope": theme_report["scope"], "official_full_market": False,
            "model_or_prediction_changed": False, "prediction_effect_verified": False,
            "new_forward_samples": 0, "automatic_entry_allowed": False,
            "limits": ["结论仅为供应商归档涨停/炸板池内短线梯队结构推断，不是全市场趋势主线认证",
                "原正文只证明限定来源叙事出现，不证明市场共识、真实业务或资金归因",
                "未知炸板归属不记零；本层不计算封板率或预测晋级概率",
                "当日识别回放不等于独立前向准确率改善；次日持续性待发生"]}


def run(theme_report, narratives, policy, output):
    paths = [Path(theme_report).resolve(), Path(narratives).resolve(), Path(policy).resolve()]
    theme, inputs, rules = map(_read, paths)
    cutoff = stamp(inputs["knowledge_cutoff"])
    if cutoff > datetime.now(timezone.utc):
        raise ValueError("analysis cutoff is in future")
    if stamp(theme.get("generated_at", theme["knowledge_cutoff"])) > cutoff:
        raise ValueError("theme report formed after cutoff")
    sources = theme.get("source_hashes")
    if not sources:
        raise ValueError("source-bound theme report required")
    binding = {str(p): sha(p) for p in paths}
    binding[str(Path(__file__).resolve())] = sha(__file__)
    bound_policy = _bind_pair(inputs, "policy_path", "policy_sha256", binding)
    if bound_policy is not None and bound_policy != paths[2]:
        raise ValueError("requested policy differs from narrative policy binding")
    receipt = inputs.get("original_text_hash_receipt")
    receipt_entries = None
    if receipt is not None:
        receipt_path = _bind_pair(receipt, "path", "sha256", binding)
        if receipt_path is None:
            raise ValueError("empty original text hash receipt")
        receipt_entries = _read(receipt_path)
        if not isinstance(receipt_entries, list):
            raise ValueError("original text receipt must contain artifact entries")
    for path, expected in sources.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError("theme source hash mismatch")
        binding[str(Path(path).resolve())] = expected
    for source in inputs["sources"]:
        if "source_kind" in source and source["source_kind"] != "VERIFIED_ORIGINAL_AUTHOR_TEXT":
            raise ValueError("unverified original source kind")
        path = Path(source["text_path"]).resolve()
        if not path.is_file() or sha(path) != source["text_sha256"]:
            raise ValueError("narrative text hash mismatch")
        content = path.read_text(encoding="utf-8")
        for claim in source.get("claims", []):
            excerpt = claim["excerpt"]
            offset = claim.get("offset")
            if excerpt not in content or (offset is not None and content[offset:offset+len(excerpt)] != excerpt):
                raise ValueError("narrative excerpt absent or offset differs")
        binding[str(path)] = source["text_sha256"]
        raw = _bind_pair(source, "raw_html_path", "raw_html_sha256", binding)
        manifest_path = _bind_pair(source, "manifest_path", "manifest_sha256", binding)
        if manifest_path is not None:
            manifest = _read(manifest_path)
            matches = [r for r in manifest.get("http_requests", []) if r.get("url") == source["url"]
                and r.get("status") == 200 and stamp(r["retrieved"]) == stamp(source["obtained_at"])
                and stamp(r["published"]) == stamp(source["published_at"])
                and (raw is None or str(r.get("sha256", "")).lower() == source["raw_html_sha256"].lower())]
            if not matches:
                raise ValueError("original source does not match request manifest")
        if receipt_entries is not None:
            for artifact in [path, raw, manifest_path]:
                if artifact is not None and not any(r.get("name") == artifact.name
                    and str(r.get("sha256", "")).lower() == sha(artifact) for r in receipt_entries):
                    raise ValueError("original source is absent from hash receipt")
    definition_time = stamp(inputs.get("formed_at", inputs.get("defined_at", inputs["knowledge_cutoff"])))
    if (definition_time > cutoff or any(stamp(s["obtained_at"]) > definition_time for s in inputs["sources"])):
        raise ValueError("narrative definition predates source acquisition or follows cutoff")
    if stamp(rules["defined_at"]) > definition_time:
        raise ValueError("narrative definition predates policy")
    result = analyse(theme, inputs, rules)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["source_hashes"] = binding
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme-report", required=True)
    parser.add_argument("--narratives", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.theme_report, args.narratives, args.policy, args.output)
    print(json.dumps({"status": result["status"], "mainline": result["mainline"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
