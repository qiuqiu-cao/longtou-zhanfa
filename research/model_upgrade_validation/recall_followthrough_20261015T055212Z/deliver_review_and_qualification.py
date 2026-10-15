"""One bounded review delivery: immutable old calls, source qualification, no fits."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from relay_cohort_output import summarize
from relay_error_analysis import components, RELAY_FIELDS
import public_outcome_sources_v293 as outcome_sources
import public_outcome_bridge_v293 as outcome_bridge

OUT = Path(__file__).resolve().parent
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    with (OUT/name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
def binding(*paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}

old15path = ROOT/"results/deep_review_20261014T115540Z/15日补充评分同身位输出.json"
scorepath = ROOT/"results/score_update_20261014T1153Z/supplemental_BASE_scores.json"
featurepath = ROOT/"results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json"
old15, scores, features = map(read, (old15path, scorepath, featurepath))
original_members = {r["code"]: {**r, "boards": g["boards"]} for g in old15["groups"] for r in g["members"]}
model = {r["code"]: r for r in scores["forecast"]["candidates"]}
feature_rows = {r["code"]: r for r in features["relay"]}
assert set(original_members) == set(model) == set(feature_rows) and len(model) == 15
compat = []
for p, expected in old15["source_bindings"].items():
    actual = sha(p)
    if actual != expected:
        preimage = OUT/"source_preimages"/Path(p).name
        assert Path(p).name == "relay_cohort_output.py" and sha(preimage) == expected
        compat.append({"original_path": p, "original_sha256": expected,
                       "exact_preserved_preimage": str(preimage), "current_sha256": actual})
rows = []
for code, original in original_members.items():
    saved = model[code]
    assert all(saved[k] == original[k] for k in ("name", "boards", "raw_score", "actual_one_word_daily"))
    assert saved["predicted_positive"] == original["original_model_positive"]
    rows.append({**original, "model_positive": original["original_model_positive"],
                 "low_turnover_proxy": feature_rows[code]["features"]["signal_one_word"]})
rows.sort(key=lambda r: (-r["raw_score"], r["code"]))
assert [r["code"] for r in rows] == old15["original_ranking"] == scores["forecast"]["ranking"]

model_path = Path(scores["model_binding"]["model_path"])
assert sha(model_path) == scores["model_binding"]["model_sha256"]
estimator = joblib.load(model_path)
x = pd.DataFrame([feature_rows[r["code"]]["features"] for r in rows])
_, _, _, replayed, _, _, _ = components(estimator, x, RELAY_FIELDS)
assert max(abs(float(v)-r["raw_score"]) for v,r in zip(replayed, rows)) < 1e-10
groups = summarize(rows)
report = {k: v for k,v in old15.items() if k not in {"groups", "source_bindings", "formed_at"}}
report.update(formed_at=datetime.now(timezone.utc).isoformat(), groups=groups,
    non_one_word_relative_first=next((r["code"] for r in rows if r["actual_one_word_daily"] == 0), None),
    no_positive_threshold_for_relative_observation=True,
    original_source_compatibility=compat, frozen_model_full15_scores_replayed=True,
    source_bindings=binding(old15path, scorepath, featurepath, model_path,
        ROOT/"relay_cohort_output.py", __file__))
save("full15_relative_review.json", report)
lines = ["# 完整15只相对观察补全", "",
    "信号日2026-10-14，目标日2026-10-15。当前才生成的诊断附页；不是新事前预测。",
    "原评分、方向、排名和作者方向全部保留，正式来源门仍未通过。非一字观察不以0.5筛除，也不自动转为晋级、主动卡位或交易主选。",
    "旧低换手代理单列；未触发不等于模型不依赖该项。分差为模型显示分，不是概率差。", "",
    "| 股票 | 板位 | 原分数 | 原方向 | 独立方向 | 信号日真实一字 | 全池非一字名次 | 同位非一字名次 | 距同位首位分差 | 旧代理 |",
    "|---|---:|---:|---|---|---|---:|---:|---:|---|"]
members = {r["code"]: r for g in groups for r in g["members"]}
for r in rows:
    m=members[r["code"]]
    lines.append(f"| {r['name']} {r['code']} | {r['boards']} | {r['display_score']:.1f} | "
        f"{'偏晋级' if r['model_positive'] else '偏断板'} | {r['independent_call']} | "
        f"{r['actual_one_word_daily']} | {m['full_pool_nonword_rank'] or '—'} | "
        f"{m['same_height_nonword_rank'] or '—'} | {m['display_score_gap_to_group_first']:.1f} | {r['low_turnover_proxy']} |")
lines += ["", "逐组相对首位（同分全部保留）："]
for g in groups:
    names = [members[c]["name"] for c in g["nonword_relative_tied_first_codes"]]
    gap = g["nonword_top_two_original_display_gap"]
    lines.append(f"- {g['boards']}板组：{'、'.join(names) or '无可核非一字'}；"
        f"同位非一字前二分差{gap if gap is not None else '无第二只/未知'}；"
        f"原非一字正向首选{g['nonword_positive_first'] or '空'}。")
lines += ["", "正向假设须独立写清叙事或父锚的支持与替代解释、完整竞争者的事实与缺口、竞价/开盘取消条件。原分数不认证主动领先；缺证据不继承BASE负向，一字/唯一活口也不自动转正。",
          "全部候选保留，路畅科技/星帅尔的复牌资格缺口仍在。当前没有同刻成交、排队或退出证据，不报告买到或收益。"]
with (OUT/"完整15只相对观察.md").open("x",encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
with (OUT/"full15_capture_candidates.csv").open("x",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=["signal_date","code","name","boards","signal_close","signal_amount","eligibility_status"])
    writer.writeheader()
    for r in rows:
        fs=feature_rows[r["code"]]["features"]
        writer.writerow(dict(signal_date="2026-10-14",code=r["code"],name=r["name"],boards=r["boards"],
            signal_close=fs.get("price"),signal_amount=fs.get("signal_amount"),
            eligibility_status="RESUMPTION_PREVIOUS_SESSION_UNRESOLVED" if r["code"] in {"002813","002860"}
                else "ALTERNATIVE_SOURCE_NOT_FORMAL_REGISTERED"))

# No new requests: reconstruct the one completed-session retry.
manifest,pools,daily,unavailable=outcome_sources.reconstruct(OUT/"production_source14")
recovery=dict(formed_at=datetime.now(timezone.utc).isoformat(), signal_date="2026-10-13",
    target_date="2026-10-14", bounded_one_network_attempt=True,
    pool_requests=[{k:e.get(k) for k in ("date","kind","status","http_status","error")}
        for e in manifest["requests"] if e["kind"]!="daily"],
    reconstructed_pools=len(pools), daily_codes=len(daily),
    all_original_candidate_codes=manifest["requested_codes"], ledger_written=False, new_formal_mature_samples=0)
try:
    result=outcome_bridge.build(ROOT/"data/v271/expectation_registry/2026-10-13_47f5c10d3a45999a.json",
        OUT/"production_source14",None,OUT/"source14_prevalidation_outcome.json")
    recovery["bridge_status"]="SOURCE_RECONSTRUCTION_ONLY"
    recovery["verified_truths"]=sum(r["status"]=="verified" for r in result["candidate_closing_truth"])
except Exception as exc:
    recovery.update(bridge_status="REJECTED",bridge_error=str(exc))
save("source14_recovery_receipt.json",recovery)
save("manual_probe_failures.json",{
    "initial_delivery_failure":"Old cohort members inherit boards from their group rather than repeat it per member; initial missing-key failure and exact initial script retained in delivery.log and source_preimages. Corrected the adapter only, no source or fit repeated.",
    "source_response_probe":"Forecast is a direct candidates/ranking block, not nested under relay; KeyError preserved in tool output, no mutations.",
    "reconstruction_probe":"Original reconstruct returns a four-item tuple. Initial dictionary access failed before receipt creation; corrected by documented API unpacking, no network repeat or ledger writes."})

# Independent source/grid audit. Promotion outcomes are deliberately not read.
pathfile=ROOT/"research/model_upgrade_validation/leader_role_peer_paths_20261012T161008Z/actual_peer_history/source_rebuilt_peer_price_facts.json"
archive=read(pathfile); parent=read(archive["parent_path"])
fullpool={r["candidate_id"]:r for r in parent["rows"]}
assert len(fullpool)==9245 and set(archive["groups"]).issubset(fullpool)
source_checks=[]
for p,wanted in archive["source_bindings"].items():
    actual=sha(p) if Path(p).is_file() else None
    source_checks.append(dict(path=p,sha256=wanted,current_sha256=actual,matches=actual==wanted,
        source_kind="CODE" if Path(p).suffix==".py" else "SAVED_DATA_OR_DERIVATION"))
save("peer_source_hash_checks.json",source_checks)
times=[f"{h:02d}:{m:02d}:00" for h in (9,10,11,13,14,15) for m in range(0,60,5)
       if "09:35:00"<=f"{h:02d}:{m:02d}:00"<="11:30:00" or "13:05:00"<=f"{h:02d}:{m:02d}:00"<="15:00:00"]
assert len(times)==48
path_checks={}
for key,p in archive["paths"].items():
    bars=p["bars"]; actual_minutes=[b["minute"] for b in bars]
    identity=all(b["code"]==p["code"] and b["minute"][:10]==p["signal_date"] for b in bars)
    prices=all(0<float(b["low"])<=float(b["open"])<=float(b["high"])
        and float(b["low"])<=float(b["close"])<=float(b["high"]) for b in bars)
    expected=[p["signal_date"]+" "+t for t in times]
    path_checks[key]=dict(code=p["code"],signal_date=p["signal_date"],bar_count=len(bars),
        identity_matches=identity,ohlc_valid=prices,
        exact_complete_grid=actual_minutes==expected,
        previous_close_known=isinstance(p.get("previous_close"),(int,float)) and p["previous_close"]>0,
        price_reconciled=p["price_reconciled"],volume_reconciled=p["volume_reconciled"],
        amount_volume_unit_consistent=p["amount_volume_unit_consistent"],
        first_bar=actual_minutes[0] if bars else None,last_bar=actual_minutes[-1] if bars else None)
save("peer_complete_path_checks.json",path_checks)
audit_rows=[]
for key,item in archive["groups"].items():
    own=path_checks[key]; context=item["sampled_peers"]
    record=dict(candidate_id=key,signal_date=own["signal_date"],code=own["code"],
        declared_current_scope=context["current_scope"],declared_prior_scope=context["prior_scope"],
        raw_source_original_time_qualified=False,
        time_reason="Source-derived package and peer reconstruction formed in October, after September signal sessions; not original forward evidence.")
    for side in ("current","prior"):
        codes=context[side]["peer_codes"]
        ids=[own["signal_date"]+":"+c for c in [own["code"]]+codes]
        available=[path_checks.get(k) for k in ids]
        complete=bool(codes) and len(codes)==len(set(codes)) and all(
            a is not None and a["identity_matches"] and a["ohlc_valid"]
            and a["exact_complete_grid"] and a["previous_close_known"] and a["price_reconciled"] for a in available)
        record.update({side+"_declared_peer_count":len(codes),
            side+"_exact_price_grid_and_reference_complete":complete,
            side+"_saved_complete_price_comparison":context[side]["complete_price_comparison"],
            side+"_all_volume_reconciled":bool(complete and all(a["volume_reconciled"] for a in available)),
            side+"_missing_path_codes":[c for c in codes if own["signal_date"]+":"+c not in path_checks]})
    audit_rows.append(record)
save("full182_peer_qualification.json",audit_rows)
summary=dict(formed_at=datetime.now(timezone.utc).isoformat(),original_full_pool=9245,
    candidates_with_archived_peer_context=len(audit_rows),other_original_candidates_without_this_path_context=9245-len(audit_rows),
    unique_stock_day_paths=len(path_checks),exact_complete_grids=sum(p["exact_complete_grid"] for p in path_checks.values()),
    source_bindings_checked=len(source_checks),source_hash_mismatches=[p for p in source_checks if not p["matches"]],
    current_closing_survivor_comparisons_saved=sum(r["current_saved_complete_price_comparison"] for r in audit_rows),
    previous_declared_cohort_comparisons_saved=sum(r["prior_saved_complete_price_comparison"] for r in audit_rows),
    independently_complete_current_price_grids=sum(r["current_exact_price_grid_and_reference_complete"] for r in audit_rows),
    independently_complete_prior_price_grids=sum(r["prior_exact_price_grid_and_reference_complete"] for r in audit_rows),
    price_grid_complete_current_with_reconciled_volume=sum(r["current_all_volume_reconciled"] for r in audit_rows),
    price_grid_complete_prior_with_reconciled_volume=sum(r["prior_all_volume_reconciled"] for r in audit_rows),
    original_training_cutoff_path_context_counts={d:sum(r["signal_date"]<=d for r in audit_rows)
        for d in ("2025-06-30","2025-12-31","2026-08-31")},
    original_time_strict_qualified=0,read_target_outcomes=False,model_fits=0,
    novelty_supports_new_fixed_numerical_challenge=False,
    limits=["Current peer roster contains closing survivors; its relative rank cannot prove intraday active leadership against failed competitors.",
            "Five-minute price ranges are not tick/reseal/order queue evidence. Volume reconciliation is a separate gate.",
            "This requalifies existing evidence; the prior 122/1 coverage is not new information or predictive improvement.",
            "No repeated market-cap/VWAP/late-bar rule test; original source-time, role labels and prior failures retained."],
    source_bindings=binding(pathfile,archive["parent_path"],__file__))
save("peer_qualification_summary.json",summary)

old10=read(ROOT/"results/deep_review_20261014T115540Z/cohort_choice/report.json")
new10=read(OUT/"full10_choice/report.json")
for oldrow,newrow in zip(old10["rows"],new10["rows"]):
    assert oldrow==newrow
assert old10["ranking"]==new10["ranking"] and new10["all_model_terms_replayed"]
for oldgroup,newgroup in zip(old15["groups"],groups):
    for oldrow,newrow in zip(oldgroup["members"],newgroup["members"]):
        assert all(newrow[k]==v for k,v in oldrow.items())
start=read(OUT/"execution_start.json")
assert all(sha(p)==h for p,h in start["protected_hashes"].items())
save("review_verification.json",dict(formed_at=datetime.now(timezone.utc).isoformat(),
    original10_rows_all_fields_identical=True,original10_ranking_identical=True,
    original15_original_member_fields_identical=True,full15_frozen_scores_replayed=True,
    full15_complete=15,all_original_protected_hashes_unchanged=True,
    relative_first_no_positive_threshold=True,independent_calls_changed=False,
    model_fits=0,new_formal_forecasts=0,new_formal_mature_samples=0,new_orders_or_fills=0,
    predictive_improvement_proved=False,modified_source_bindings=binding(
        ROOT/"relay_cohort_output.py",ROOT/"relay_choice_reasoning.py",
        ROOT/"tests/test_relay_cohort_output.py",ROOT/"tests/test_relay_choice_reasoning.py")))
print(json.dumps({"full10":10,"full15":15,"relative15_first":report["non_one_word_relative_first"],
    "peer_audit":{k:v for k,v in summary.items() if k not in {"source_bindings","limits","source_hash_mismatches"}},
    "source14":{k:v for k,v in recovery.items() if k!="pool_requests"}},ensure_ascii=False))

