"""Source-bound field meanings and training context for immutable relay choices.

Descriptive context only: no fitting, alternative scores, thresholds or calls.
Target outcomes are never used for the current candidates.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import joblib
import pandas as pd

import model_upgrade_history as history
import stock_judgement_review as original
from relay_error_analysis import RELAY_FIELDS, components


SEMANTICS = {
    "boards": "本股信号日连续涨停板数，不是题材梯队层数。",
    "height_gap": "信号日原封板池最高板数减本股板数；0表示并列最高，不能认证唯一龙头。",
    "same_height_rivals": "原完整连板池同板数候选数减1；不限定同题材，也不预设只能活一只。",
    "unique_space": "本股处在原池最高板且该高度仅一只的指示值；不等于独立资金或题材核心。",
    "height_share": "本股板数除以原封板池最高板数；不是同题材占比、成交占比或市场份额。",
    "is_space_leader": "本股板数等于原池最高板数的指示值；允许多只同时为1。",
    "emotion_score": "原emotion_dimensions派生的市场情绪尺度，不是校准晋级概率。",
    "promotion_2to3": "上一原封板池二板候选在本日恰好升为三板的比例；分母是昨日二板组。",
    "promotion_3to4": "上一原封板池三板候选在本日恰好升为四板的比例；分母是昨日三板组。",
    "promotion_mid": "上一原封板池二至四板候选本日恰好加一板的比例；不是本日中位组的未来胜率。",
    "previous_limit_tail_loss": "完整昨日封板组本日反馈收益的10%分位，收益用小数；缺覆盖保持未知。",
    "failed_rate": "原炸板数除以原封板数加炸板数；供应商范围，不是全A认证。",
    "down_limit_rate": "原跌停家数除以10后截到[0,1]；不是跌停股票占全市场的比例。",
    "breadth_health": "原emotion_dimensions派生的涨停广度尺度；不认证主动主线。",
    "ladder_health": "原emotion_dimensions派生的梯队尺度；不等于已核同题材完整梯队。",
    "loss_effect_health": "原emotion_dimensions派生的亏钱效应尺度；不是实际可交易收益。",
    "crowding_risk": "原emotion_dimensions派生的拥挤尺度；不是实测讨论量或盘口拥挤概率。",
    "regime_fragility": "原emotion_dimensions派生的环境脆弱尺度；不是已校准风险概率。",
    "turnover": "当日供应商换手率除以100；0.03表示3%，不等于已消化抛压。",
    "signal_amount": "当日成交额，单位元；总量不表示买方主动性或承接质量。",
    "amplitude": "日线高低价差相对前收的百分数；5表示5%，不表示开板时长。",
    "turnover_vs_prev": "本日换手与上次进入连板池时换手的比值，参考日未必是前一交易日；原当前卡未导出参考日则保持未知。",
    "price": "信号日原不复权价格，单位元；不是买入成本。",
    "price_same_height_percentile": "完整当日同板数组内价格的平均秩百分位；并列允许平均秩。",
    "signal_amount_same_height_percentile": "完整当日同板数组内成交额的平均秩百分位；不限定同题材。",
    "turnover_same_height_percentile": "完整当日同板数组内换手的平均秩百分位；不是换手质量分。",
    "turnover_vs_prev_same_height_percentile": "完整当日同板数组内上次入池换手比的平均秩百分位；缺比值不自动当最低。",
    "signal_one_word": "旧代理：早期资料依赖封板时间、开板次数和低换手，且原逻辑允许部分字段缺失；9月及当前资料为日线OHLC一价乘低换手。两种来源不能自动认证等价，也都不等于真实一字身份。",
    "log_signal_amount": "log1p(以元计的成交额)；模型变换值，不是实际金额或新增资金。",
}


def describe(values):
    values = pd.to_numeric(values, errors="coerce")
    known = values.dropna()
    if any(not math.isfinite(float(x)) for x in known):
        raise ValueError("nonfinite original input")
    return {"rows": len(values), "known": len(known), "missing": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()) if len(values) else None,
            **{key: float(known.quantile(q)) if len(known) else None
               for key, q in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1))}}


def relative(value, distribution):
    if value is None:
        return "UNKNOWN_INPUT_MEDIAN_IMPUTATION_IS_NOT_WEAKNESS"
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite original value or explicit unknown required")
    if distribution["known"] == 0:
        return "TRAINING_FIELD_ALL_UNKNOWN"
    if value < distribution["min"] or value > distribution["max"]:
        return "OUTSIDE_OBSERVED_TRAINING_RANGE_DESCRIPTIVE_ONLY"
    if value < distribution["p05"] or value > distribution["p95"]:
        return "OUTSIDE_TRAINING_P05_P95_DESCRIPTIVE_ONLY"
    return "WITHIN_TRAINING_P05_P95"


def build(base_path, choice_path, receipt_path, output):
    if set(SEMANTICS) != set(RELAY_FIELDS):
        raise ValueError("all original ordered29 fields require explicit meanings")
    base, choice, receipt = map(original.read, (base_path, choice_path, receipt_path))
    if choice.get("status") != "READ_ONLY_FULL_POOL_CHOICE_REASONING_NOT_NEW_FORECAST":
        raise ValueError("actual original choice explanation required")
    original.check_sources(choice["source_bindings"])
    if choice["source_bindings"].get(str(Path(base_path).resolve())) != original.sha(base_path):
        raise ValueError("choice not bound to this original BASE")
    original.check_sources(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    feature_dir = Path(base["feature_archive"])
    manifest, features = map(original.read, (feature_dir/"manifest.json", feature_dir/"features.json"))
    original.check_sources(manifest["source_hashes"] | {str(feature_dir/"features.json"): manifest["features_sha256"]})
    if (base["signal_date"], base["target_date"], base["knowledge_cutoff"]) != (
            choice["signal_date"], choice["target_date"], choice["original_model_knowledge_cutoff"]):
        raise ValueError("original choice dates or knowledge cutoff differ")
    if (features["signal_date"], features["target_date"]) != (base["signal_date"], base["target_date"]):
        raise ValueError("original feature dates differ")
    calls = original.unique_rows(base["relay"]["candidates"])
    rows = original.unique_rows(choice["rows"])
    inputs = original.unique_rows(features["relay"])
    ranking = base["relay"]["ranking"]
    if set(calls) != set(rows) or set(calls) != set(inputs) or choice["ranking"] != ranking:
        raise ValueError("complete original pool/order required")
    binding = base["model_bindings"]["relay"]
    model_manifest = original.read(binding["path"])
    original.check_sources({binding["path"]: binding["sha256"], binding["model_path"]: binding["model_sha256"]})
    if model_manifest["fields"] != RELAY_FIELDS:
        raise ValueError("original field order differs")
    prepared = joblib.load(receipt["prepared_path"])
    train = history.training_rows(prepared["relay"], model_manifest["cutoff"], "relay")
    if (history._frame_digest(train, RELAY_FIELDS) != model_manifest["training_digest"]
            or len(train) != model_manifest["training_rows"]):
        raise ValueError("actual frozen model training members/fields differ")
    x = pd.DataFrame([inputs[c]["features"] for c in ranking])
    names, terms, logits, raw, imputed, means, intercept = components(joblib.load(binding["model_path"]), x, RELAY_FIELDS)
    distributions = {field: describe(train[field]) for field in RELAY_FIELDS}
    context_rows = []
    for i, code in enumerate(ranking):
        call, saved = calls[code], rows[code]
        if (abs(raw[i]-call["raw_score"]) > 1e-10 or saved["raw_score"] != call["raw_score"]
                or saved["model_positive"] != call["predicted_positive"]
                or bool(raw[i] > .5) != call["predicted_positive"]):
            raise ValueError("source model score/call altered")
        presented_terms = saved["positive_model_terms"] + saved["negative_model_terms"]
        saved_terms = {t["field"]: t for t in presented_terms}
        nonzero_names = {name for j, name in enumerate(names) if terms[i,j] != 0}
        if len(saved_terms) != len(presented_terms) or set(saved_terms) != nonzero_names:
            raise ValueError("complete original terms required")
        field_rows = []
        for j, field in enumerate(names):
            actual = None if field not in RELAY_FIELDS or pd.isna(x.iloc[i][field]) else float(x.iloc[i][field])
            if field in saved_terms and (saved_terms[field]["signal_value"] != actual or abs(saved_terms[field]["logit_contribution"]-terms[i,j]) > 1e-10):
                raise ValueError("self-consistent fictional contribution or observed value rejected")
            if field not in RELAY_FIELDS:
                continue
            field_rows.append({"field": field, "meaning": SEMANTICS[field], "original_value": actual,
                "original_imputed_value": float(imputed[i,j]), "original_logit_contribution": float(terms[i,j]),
                "training_distribution": distributions[field], "descriptive_position": relative(actual, distributions[field]),
                "turnover_reference_date": inputs[code].get("turnover_reference_date") if field == "turnover_vs_prev" else None})
        context_rows.append({"code": code, "name": call["name"], "rank": i+1, "raw_score": call["raw_score"],
            "model_positive": call["predicted_positive"], "independent_call": saved["independent_call"],
            "actual_one_word_daily": call["actual_one_word_daily"], "fields": field_rows})
    source_paths = [base_path, choice_path, receipt_path, feature_dir/"manifest.json", feature_dir/"features.json", __file__]
    sources = choice["source_bindings"] | receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]}
    sources.update({str(Path(p).resolve()): original.sha(p) for p in source_paths})
    report = {"status": "SOURCE_BOUND_DESCRIPTIVE_FIELD_CONTEXT_NOT_FORECAST", "formed_at": datetime.now(timezone.utc).isoformat(),
        "signal_date": base["signal_date"], "target_date": base["target_date"],
        "original_knowledge_cutoff": base["knowledge_cutoff"], "candidate_count": len(ranking), "ranking": ranking,
        "training_rows": len(train), "training_signal_dates": int(train.date.nunique()),
        "training_digest": model_manifest["training_digest"], "training_true_one_word_known": int(train.actual_one_word_daily.notna().sum()),
        "training_proxy_source": "LEGACY_SEAL_TIME_BOMB_COUNT_TURNOVER_PROXY_REAL_OHLC_IDENTITY_UNKNOWN",
        "current_proxy_source": "RAW_DAILY_OHLC_ONE_PRICE_TIMES_TURNOVER_LT3_PERCENT",
        "proxy_source_equivalence_verified": False, "current_market_input_status": features["market"]["status"],
        "rows": context_rows, "source_bindings": sources, "numeric_fits": 0, "score_call_order_changed": False,
        "new_mature_forward_samples": 0, "accuracy_improvement_verified": False,
        "limits": ["Training quantiles describe scope, not new decision thresholds or causal importance.",
                   "Missingness and proxy provenance do not alone authorize a positive/negative call.",
                   "Original current card has no turnover reference date; do not call the ratio previous-day.",
                   "This context was formed later and must not backfill the original forecast knowledge cutoff."]}
    original.check_sources(sources)
    destination = Path(output)
    destination.mkdir(exist_ok=False, parents=True)
    (destination/"report.json").write_text(original_json(report), encoding="utf-8")
    text = ["# 原字段含义与训练范围", "", f"信号日{base['signal_date']}，目标日{base['target_date']}；本附页实际形成{report['formed_at']}。", "",
            f"原模型训练{len(train)}只、{train.date.nunique()}个标签日期，真实OHLC一字身份已知{report['training_true_one_word_known']}只。",
            "历史封板时间代理与当前OHLC代理来源有差异，尚未认证等价；保留原分数，不能把差异单独当看空或看多理由。",
            "训练分位只描述原输入范围，不是晋级门槛。当前未知沿原模型插补，不等于弱势。", ""]
    for row in context_rows:
        text += [f"## {row['name']} {row['code']}：原分数{row['raw_score']*100:.1f}", "",
                 "| 字段 | 原值 | 原训练中位数 | 原训练P05—P95 | 含义 |", "| --- | ---: | ---: | --- | --- |"]
        for field in row["fields"]:
            d = field["training_distribution"]
            text.append(f"| {field['field']} | {field['original_value'] if field['original_value'] is not None else '未知'} | {d['median']} | {d['p05']}—{d['p95']} | {field['meaning']} |")
        text += ["", "各项原模型贡献、支持与反证见原选择理由；本页不生成新方向、分数或交易主选。", ""]
    (destination/"原字段含义与训练范围.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return report


def original_json(value):
    import json
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "choice", "prepared-receipt", "output-dir"):
        parser.add_argument("--"+field, required=True)
    args = parser.parse_args()
    report = build(args.base, args.choice, args.prepared_receipt, args.output_dir)
    print(original_json({k:report[k] for k in ("status","candidate_count","training_rows","training_true_one_word_known","numeric_fits")}))


if __name__ == "__main__":
    main()
