"""Read-only score semantics and full-pool research ranking view.

This module never fits, calibrates, predicts, or changes a supplied direction.
The live workbench must validate its existing complete author review gate before
calling this view.  The CLI consumes already exported scores as an archive
diagnostic; it is not a new-date scoring or prediction registration entrance.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "research-scorecard-semantics-v1"
PAIRWISE_ARM = "PAIRWISE_DAY_RANK"
LEGEND = {
    "view_scope": "只读研究展示；旧日期诊断。新日期仍须先由原工作台完成完整独立审阅及原留档门，再展示模型分数。",
    "absolute_event_score": "输入模型绝对分类分，未经概率校准；不可解释为收盘涨停概率。方向直接保留原输出，不按排名改写。",
    "research_ranking_score": "同一日期、同一方案内的相对排序值；PAIRWISE_DAY_RANK仅为成对偏好，不具有绝对事件方向，不跨日期或方案比较数值大小。",
    "score_units": "原始数值原样保留；不转为0—100概率，不补0，不用结果字段构造排名。",
    "pool": "完整当日候选池，包括无分数及资料未知者。无分数保留行，但不伪造名次。",
    "same_height": "当日同方案同板位候选；板位未知单列UNKNOWN组，不与已知板位混排。",
    "non_one_word": "原daily_one_word=0的完整非一字评价池。当前价型资料资格单列，不因资料未知从原评价分母删除。",
    "ties": "position按代码稳定拆分展示；rank为同分最小名次，tie_layer为并列层级，tie_count为该分数人数；同分首位均标记相对第一。",
    "percentile": "仅在至少2只有效分数时计算组内已评分成员的相对名次分位；单只/无分数为未知。它不是事件概率。",
    "gaps": "分差使用同一方案原始排序值：距首位及前后相邻不同分值；单只、末位或未知分数不填0冒充可比较差值。",
    "small_pool": "组内候选不足5只标记小池，只有1只标记单股；没有独立证据时相对第一不自动正向。",
    "call": "PROMOTE为原分类正向；FAIL为原分类负向；UNKNOWN/缺记录为弃判或未知。研究正向不认证可交易或收益。",
    "qualification": "current_price_shape仅表示当前价型资格：0非一字、1单一价格、缺失未知，不认证成交或完整原时点证据。",
    "baseline": "BASE原始分数、方向、输入排名和其完整池/同位/非一字排名另列；无BASE记录保留未知，不继承其他方案。",
    "outcomes": "target/label_valid等结果列仅作为评测附列，不参与分数、方向、名次、分差或选择。",
}


def _number(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype(float)
    return result.where(np.isfinite(result))


def _missing(value):
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_code(value):
    if _missing(value):
        raise ValueError("candidate code is required")
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value.zfill(6) if value.isdigit() else value


def _rank_group(frame: pd.DataFrame, indices, prefix: str):
    """Rank only finite supplied values, retaining the complete group size."""
    indices = list(indices)
    size = len(indices)
    values = frame.loc[indices, "research_ranking_score"]
    valid = values.dropna()
    known = len(valid)
    for field, value in [("pool_size", size), ("scored_count", known),
                         ("unknown_score_count", size - known),
                         ("small_pool", size < 5), ("singleton", size == 1)]:
        frame.loc[indices, prefix + "_" + field] = value
    if known == 0:
        return
    ordered = frame.loc[valid.index].sort_values(
        ["research_ranking_score", "code"], ascending=[False, True], kind="mergesort"
    )
    positions = pd.Series(range(1, known + 1), index=ordered.index)
    frame.loc[valid.index, prefix + "_position"] = positions.reindex(valid.index)
    frame.loc[valid.index, prefix + "_rank"] = valid.rank(method="min", ascending=False)
    frame.loc[valid.index, prefix + "_tie_layer"] = valid.rank(method="dense", ascending=False)
    counts = valid.value_counts()
    frame.loc[valid.index, prefix + "_tie_count"] = valid.map(counts)
    top = valid.max()
    frame.loc[valid.index, prefix + "_relative_first"] = valid.eq(top)
    frame.loc[valid.index, prefix + "_gap_to_first"] = top - valid
    if known > 1:
        frame.loc[valid.index, prefix + "_rank_percentile"] = (
            known - valid.rank(method="average", ascending=False)
        ) / (known - 1)
    distinct = sorted(valid.unique(), reverse=True)
    previous = {v: (distinct[i - 1] - v if i else np.nan) for i, v in enumerate(distinct)}
    following = {v: (v - distinct[i + 1] if i + 1 < len(distinct) else np.nan)
                 for i, v in enumerate(distinct)}
    frame.loc[valid.index, prefix + "_gap_previous_distinct"] = valid.map(previous)
    frame.loc[valid.index, prefix + "_gap_next_distinct"] = valid.map(following)


RANK_FIELDS = ["pool_size", "scored_count", "unknown_score_count", "small_pool", "singleton",
               "position", "rank", "tie_layer", "tie_count", "relative_first", "gap_to_first",
               "rank_percentile", "gap_previous_distinct", "gap_next_distinct"]
RANK_PREFIXES = ["pool", "same_height", "non_one_word_pool", "non_one_word_same_height"]


def build_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input candidate/arm, with no prediction or outcome use.

    Required columns: date/code. arm defaults to UNSPECIFIED. All original
    columns and order are retained; derived columns use explicit semantics.
    PAIRWISE_DAY_RANK suppresses event score/call even if supplied accidentally,
    retaining those inputs separately for audit. Missing scores remain missing.
    """
    if not {"date", "code"}.issubset(frame.columns):
        raise ValueError("date and code are required")
    if not frame.columns.is_unique:
        raise ValueError("duplicate column names are not allowed")
    output = frame.copy(deep=True).reset_index(drop=True)
    output["date"] = output["date"].astype(str)
    if output.date.isin(["", "None", "nan", "NaT"]).any():
        raise ValueError("candidate date is required")
    output["code"] = output.code.map(_as_code)
    if "arm" not in output:
        output["arm"] = "UNSPECIFIED"
    output["arm"] = output.arm.fillna("UNSPECIFIED").astype(str)
    if output.duplicated(["date", "code", "arm"]).any():
        raise ValueError("duplicate date/code/arm candidate; complete pool must be explicit")
    for field in ["name", "boards", "score", "ranking_score", "call", "current_price_shape", "daily_one_word"]:
        if field not in output:
            output[field] = None
    output["score_input_original"] = output.score
    output["ranking_score_input_original"] = output.ranking_score
    output["call_input_original"] = output.call
    output["input_rank_original"] = output["rank"] if "rank" in output else None
    pairwise = output.arm.eq(PAIRWISE_ARM)
    supplied_score, supplied_rank = _number(output.score), _number(output.ranking_score)
    output["absolute_event_score"] = supplied_score.where(~pairwise)
    output["research_ranking_score"] = supplied_rank.fillna(supplied_score)
    output["ranking_score_origin"] = np.where(supplied_rank.notna(), "SUPPLIED_RANKING_SCORE",
                                              np.where(supplied_score.notna(), "SUPPLIED_SCORE_FOR_RELATIVE_ORDER", "UNKNOWN_NO_SCORE"))
    calls = output.call.map(lambda v: "UNKNOWN" if _missing(v) else str(v).upper())
    output["absolute_event_call"] = calls.where(calls.isin(["PROMOTE", "FAIL", "UNKNOWN"]), "UNKNOWN").where(~pairwise)
    output["absolute_score_semantics"] = np.where(pairwise, "NOT_AVAILABLE_PAIRWISE_PREFERENCE_ONLY", "UNCALIBRATED_CLASSIFICATION_SCORE")
    output["ranking_score_semantics"] = np.where(pairwise, "PAIRWISE_PREFERENCE_WITHIN_DAY_ONLY", "RELATIVE_RESEARCH_ORDER_WITHIN_DAY_AND_ARM")
    output["event_probability_validated"] = False
    output["actionable"] = False
    output["production_eligible"] = False
    output["view_scope"] = "OFFLINE_POST_REVIEW_RESEARCH_DIAGNOSTIC_ONLY"
    output["attention_state"] = output.absolute_event_call.map({
        "PROMOTE": "POSITIVE_CLASSIFICATION_RESEARCH_ONLY", "FAIL": "OBSERVATION_NEGATIVE_CLASSIFICATION",
        "UNKNOWN": "UNKNOWN_ABSTAIN_OR_NO_RECORD"})
    output.loc[pairwise, "attention_state"] = "OBSERVATION_RELATIVE_PREFERENCE_ONLY"
    shape = _number(output.current_price_shape)
    output["price_qualification"] = np.select([shape.eq(0), shape.eq(1)],
        ["OBSERVED_NON_ONE_WORD_PRICE_SHAPE", "OBSERVED_ONE_PRICE_SHAPE_A_CHANNEL"], default="UNKNOWN_PRICE_SHAPE_QUALIFICATION")
    original_shape = _number(output.daily_one_word)
    output["original_non_one_word_pool_member"] = original_shape.eq(0)
    output["original_shape_known"] = original_shape.isin([0, 1])
    output["price_shapes_disagree"] = shape.isin([0, 1]) & original_shape.isin([0, 1]) & shape.ne(original_shape)
    height = _number(output.boards)
    output["same_height_group"] = height.map(lambda v: "UNKNOWN" if pd.isna(v) else str(v))
    output["same_height_known"] = height.notna()
    for prefix in RANK_PREFIXES:
        for field in RANK_FIELDS:
            output[prefix + "_" + field] = np.nan
        # Unknown members are not implicitly relative-first/last.
        output[prefix + "_relative_first"] = pd.Series([None] * len(output), dtype=object)
        output[prefix + "_small_pool"] = pd.Series([None] * len(output), dtype=object)
        output[prefix + "_singleton"] = pd.Series([None] * len(output), dtype=object)
    for _, group in output.groupby(["date", "arm"], sort=False, dropna=False):
        _rank_group(output, group.index, "pool")
        for _, subgroup in group.groupby("same_height_group", sort=False):
            _rank_group(output, subgroup.index, "same_height")
        b_group = group[group.original_non_one_word_pool_member]
        if not b_group.empty:
            _rank_group(output, b_group.index, "non_one_word_pool")
            for _, subgroup in b_group.groupby("same_height_group", sort=False):
                _rank_group(output, subgroup.index, "non_one_word_same_height")
    # A BASE record is never substituted with a research arm. Its input values
    # and ranks remain visible alongside all schemes on exactly the same code.
    base_fields = ["score_input_original", "call_input_original", "input_rank_original"] + [
        prefix + "_" + field for prefix in RANK_PREFIXES for field in ["position", "rank", "tie_layer", "tie_count"]]
    base = output[output.arm.eq("BASE")][["date", "code"] + base_fields]
    base = base.rename(columns={field: "base_" + field for field in base_fields})
    output = output.merge(base, on=["date", "code"], how="left", sort=False, validate="many_to_one")
    return output


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if _missing(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_html(frame: pd.DataFrame) -> str:
    """Self-contained full table with date/arm filters and expandable raw rows."""
    # A columnar payload avoids repeating 100+ column names for every candidate
    # in a multi-year full-pool export; no rows or fields are dropped.
    packed = {"columns": frame.columns.tolist(), "rows": _clean(frame.to_numpy().tolist())}
    payload = json.dumps(packed, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    legend = "".join("<dt>" + html.escape(k) + "</dt><dd>" + html.escape(v) + "</dd>" for k, v in LEGEND.items())
    template = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整研究评分对照</title>
<style>body{font:14px system-ui,sans-serif;margin:24px;color:#172735}h1{font-size:22px}.notice{background:#fff4dc;padding:12px}table{border-collapse:collapse;min-width:1350px;width:100%}th,td{padding:7px;border:1px solid #dbe0e5;vertical-align:top}th{position:sticky;top:0;background:#edf3f7}.scroll{overflow:auto;max-height:75vh}dt{font-weight:bold;margin-top:8px}dd{margin-left:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:900px}select{margin:8px 20px 8px 4px}.positive{background:#eff7ee}.unknown{background:#f7f3e9}summary{cursor:pointer}small{color:#526474}</style>
<h1>完整研究评分对照</h1><p class="notice">只读离线诊断。所有分数未经概率校准，PAIRWISE仅为相对偏好。未知候选、弃判和并列完整保留。新日期必须由原工作台独立审阅留档后调用此视图；本页不生成预测、不回写正式BASE。</p>
<details><summary>评分语义、完整分母与限制</summary><dl>__LEGEND__</dl></details>
<label>日期<select id="date"></select></label><label>方案<select id="arm"></select></label><span id="count"></span>
<div class="scroll"><table><thead><tr><th>日期 / 方案</th><th>股票 / 板位</th><th>BASE原分 / 原方向 / 完整池名次</th><th>绝对分类分 / 原方向</th><th>研究排序值 / 语义</th><th>完整池 / 同位名次</th><th>非一字 / 非一字同位名次</th><th>完整池距首位 / 下一不同分差</th><th>资格 / 关注状态</th><th>完整输入与派生字段</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>const packed=__DATA__;const data=packed.rows.map(v=>Object.fromEntries(packed.columns.map((c,i)=>[c,v[i]])));
const esc=v=>String(v??'未知').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v===null||v===undefined?'未知':typeof v==='number'?Number(v.toPrecision(8)).toString():esc(v);
function rank(r,p){if(r[p+'_pool_size']==null)return '不适用';let text=fmt(r[p+'_rank'])+'/'+fmt(r[p+'_pool_size'])+'（有效分'+fmt(r[p+'_scored_count'])+'）';if(r[p+'_tie_count']>1)text+=' 并列'+r[p+'_tie_count']+'只/层级'+r[p+'_tie_layer']+'/代码序'+r[p+'_position'];if(r[p+'_singleton'])text+=' 单股';else if(r[p+'_small_pool'])text+=' 小池';if(r[p+'_relative_first'])text+=' 相对第一';return text;}
for(const k of ['date','arm']){const el=document.getElementById(k);let values=[...new Set(data.map(x=>x[k]))].sort();el.innerHTML='<option value="">全部</option>'+values.map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('');if(k==='date'&&values.length)el.value=values[values.length-1];el.onchange=render;}
function render(){let d=document.getElementById('date').value,a=document.getElementById('arm').value;let shown=data.filter(r=>(!d||r.date===d)&&(!a||r.arm===a));document.getElementById('count').textContent='展示 '+shown.length+' / 完整 '+data.length+' 行';document.getElementById('rows').innerHTML=shown.map(r=>'<tr class="'+(r.absolute_event_call==='PROMOTE'?'positive':r.absolute_event_call==null||r.absolute_event_call==='UNKNOWN'?'unknown':'')+'"><td>'+esc(r.date)+'<br>'+esc(r.arm)+'</td><td>'+esc(r.code)+' '+esc(r.name)+'<br>'+fmt(r.boards)+'板</td><td>'+fmt(r.base_score_input_original)+' / '+esc(r.base_call_input_original)+' / '+fmt(r.base_pool_rank)+'</td><td>'+fmt(r.absolute_event_score)+' / '+esc(r.absolute_event_call)+'</td><td>'+fmt(r.research_ranking_score)+'<br><small>'+esc(r.ranking_score_semantics)+'</small></td><td>'+rank(r,'pool')+'<br>'+rank(r,'same_height')+'</td><td>'+rank(r,'non_one_word_pool')+'<br>'+rank(r,'non_one_word_same_height')+'</td><td>'+fmt(r.pool_gap_to_first)+' / '+fmt(r.pool_gap_next_distinct)+'</td><td>'+esc(r.price_qualification)+'<br>'+esc(r.attention_state)+'</td><td><details><summary>展开全部字段</summary><pre>'+esc(JSON.stringify(r,null,2))+'</pre></details></td></tr>').join('');}render();</script></html>"""
    return template.replace("__LEGEND__", legend).replace("__DATA__", payload)


def write_scorecard(frame: pd.DataFrame, output_dir, input_path=None):
    """Write new artifacts only; an existing output directory is rejected."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("scorecard output already exists; frozen artifacts are never overwritten")
    table = build_scorecard(frame)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "complete_scorecard.csv", index=False, encoding="utf-8-sig")
    content = {"version": VERSION, "legend": LEGEND, "rows": _clean(table.to_dict("records"))}
    (output / "complete_scorecard.json").write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "complete_scorecard.html").write_text(render_html(table), encoding="utf-8")
    bindings = {str(Path(__file__).resolve()): _digest(__file__)}
    if input_path is not None:
        bindings[str(Path(input_path).resolve())] = _digest(input_path)
    manifest = {"version": VERSION, "view_scope": "ARCHIVE_DIAGNOSTIC_ONLY", "complete_input_rows": len(frame),
        "complete_output_rows": len(table), "dates": sorted(table.date.unique().tolist()),
        "arms": sorted(table.arm.unique().tolist()), "original_models_changed": False, "new_fits": 0,
        "production_directions_changed": False, "existing_author_review_gate_required_for_new_dates": True,
        "event_probability_validated": False, "source_bindings": bindings,
        "artifact_hashes": {p.name: _digest(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="already exported CSV or JSON rows; archive diagnostic only")
    parser.add_argument("--output-dir", required=True, help="new output directory; never overwrite")
    args = parser.parse_args()
    source = Path(args.predictions)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    else:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        frame = pd.DataFrame(raw["rows"] if isinstance(raw, dict) else raw)
    manifest = write_scorecard(frame, args.output_dir, source)
    print(json.dumps({k: manifest[k] for k in ["view_scope", "complete_input_rows", "complete_output_rows", "arms", "new_fits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
