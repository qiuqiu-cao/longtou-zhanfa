"""Apply V274's predeclared monthly procedure to reported September pools.

All model choice/refit labels precede 2026-09-01. Pools, unadjusted bars and
missing sessions are source-audited; September outcomes never select a model.
This is retrospective falsification, not pristine blind or forward evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from next_session_forecast_lab_v271 import ROOT, DATA, prepare_market, prepare_relay, prediction, top1, metrics, rate, sha
from market_expectation_extensions_v272 import INDEX_PATH, direction
from rolling_forecast_challenger_v274 import KINDS, WINDOWS, VALIDATION_DATES, known_before, trailing, fit, selection_metric
from reported_seal_archive_v275 import sessions, CALENDAR_SOURCES
from leader_system_v3 import promotion_rates, rolling_emotion_context, emotion_dimensions, classify_emotion
from historical_ohlc_archive_v278 import verify_archive

VERSION = "v278-september-fixed-monthly-falsification-2"
ARCHIVE = ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"
DAILY = ROOT / "data/v278/september_unadjusted_20261002_v3/unadjusted_daily.csv"
CUTOFF = "2026-09-01"


def attach_ohlc(pools, daily):
    bars = daily.sort_values(["symbol", "date"]).copy()
    bars["pre_close"] = bars.groupby("symbol").close.shift(1)
    f = pools.copy()
    f["symbol"] = f.code.map(lambda c: ("sh" if c.startswith("6") else "sz")+c)
    f = f.merge(bars, on=["date", "symbol"], how="left", validate="many_to_one")
    agreed = (f.close-f.price).abs().le(.011)
    f["ohlc_status"] = np.where(agreed, "CLOSE_AGREES_UNADJUSTED_PROVIDER_BAR", "UNAVAILABLE_OR_DISAGREES")
    f["actual_one_word_daily"] = np.where(agreed, np.isclose(f.high, f.low, atol=.001, rtol=0).astype(float), np.nan)
    # Match the original V3 proxy: all-day one-price plus turnover under 3%.
    # Also retain the pure OHLC definition, which can differ at higher turnover.
    f["one_word_proxy"] = f.actual_one_word_daily * f.turnover.lt(3).astype(float)
    f["amplitude"] = ((f.high-f.low)/f.pre_close*100).where(agreed)
    return f


def auxiliary(source_path, day, kind):
    folder = (ROOT / source_path).parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if pd.Timestamp(str(manifest["date"])).strftime("%Y-%m-%d") != day:
        return None, {"status": "SOURCE_DATE_MISMATCH", "kind": kind, "date": day}
    path = folder / (kind+".csv")
    block = manifest.get("datasets", {}).get(kind, {})
    if not path.exists() or block.get("rows") is None:
        return None, {"status": "UNAVAILABLE", "kind": kind, "date": day}
    if block.get("sha256") and sha(path) != block["sha256"]:
        raise ValueError("auxiliary source hash mismatch")
    try:
        f = pd.read_csv(path, dtype={"代码": str})
    except pd.errors.EmptyDataError:
        if block["rows"] != 0:
            raise
        f = pd.DataFrame({"代码":pd.Series(dtype=str),"名称":pd.Series(dtype=str),"涨跌幅":pd.Series(dtype=float)})
    if len(f) != block["rows"]:
        raise ValueError("auxiliary source row-count mismatch")
    f = f[f["代码"].str.match(r"^(60|00)\d{4}$") & ~f["名称"].str.contains("ST|退", regex=True)].copy()
    if f["代码"].duplicated().any():
        raise ValueError("duplicate auxiliary stock code")
    return f, {"status": "MANIFEST_BOUND_PROVIDER_OBSERVATION", "date": day, "kind": kind,
               "path": str(path.relative_to(ROOT)), "sha256": sha(path),
               "captured_at": manifest.get("collected_at"), "mainboard_rows": len(f)}


def feedback_cohort(previous, feedback):
    if previous is None or feedback is None:
        return feedback, {"status":"DENOMINATOR_UNAVAILABLE","coverage":None}
    expected = set(previous.code)
    supplied = set(feedback["代码"])
    matched = feedback[feedback["代码"].isin(expected)].copy()
    missing, extra = sorted(expected-supplied),sorted(supplied-expected)
    return matched,{"status":"EXACT_PREVIOUS_CLOSING_POOL_COVERED" if not missing else "PARTIAL_PREVIOUS_CLOSING_POOL",
                    "expected":len(expected),"matched":len(matched),
                    "coverage":len(matched)/len(expected) if expected else None,
                    "missing_codes":missing,"excluded_extra_codes":extra}


def rebuild_market(pools, base):
    cal = sessions(); predecessor = dict(zip(cal[1:], cal[:-1]))
    by_day = {d: g for d,g in pools.groupby("date")}
    history = base[base.date < cal[0]].to_dict("records")
    rows, audit = [], []
    for day in cal:
        if day > "2026-09-30":
            break
        pool = by_day.get(day)
        if pool is None:
            row = {"date": day, "data_status": "unavailable", "emotion_state": None}
            rows.append(row); history.append(row)
            continue
        previous = by_day.get(predecessor.get(day))
        previous_market = history[-1] if history else {}
        ladder = pool[pool.boards.ge(2)]
        height = int(pool.boards.max())
        counts = {}
        for kind in ["failed_board", "limit_down", "previous_limit"]:
            frame, item = auxiliary(pool.source_path.iloc[0], day, kind)
            audit.append(item); counts[kind] = frame
        failed = float(len(counts["failed_board"])) if counts["failed_board"] is not None else np.nan
        down = float(len(counts["limit_down"])) if counts["limit_down"] is not None else np.nan
        feedback, cohort = feedback_cohort(previous,counts["previous_limit"])
        audit.append({"date":day,"kind":"previous_limit_cohort",**cohort})
        returns = pd.to_numeric(feedback["涨跌幅"], errors="coerce")/100 if feedback is not None else pd.Series(dtype=float)
        row = {"date": day, "limit_count": float(len(pool)), "consecutive_count": len(ladder),
               "max_board": height, "failed_count": failed, "down_limit_count": down,
               "seal_rate": len(pool)/(len(pool)+failed), "failed_rate": failed/(len(pool)+failed),
               "height_change": height-previous_market.get("max_board", np.nan),
               "limit_change": len(pool)-previous_market.get("limit_count", np.nan),
               "middle_density": ladder.boards.between(2,4).sum()/max(1,len(ladder)),
               "ladder_gap_count": sum(b not in set(ladder.boards) for b in range(2,height+1)),
               "unique_space": float(ladder.boards.eq(height).sum() == 1),
               "one_word_share": float(ladder.one_word_proxy.mean()) if ladder.one_word_proxy.notna().all() else np.nan,
               "yesterday_premium": float(returns.median()) if len(returns) else np.nan,
               "previous_limit_tail_loss": float(returns.quantile(.1)) if len(returns) else np.nan,
               "source_available_at": pool.captured_at.iloc[0],
               "feedback_cohort_status":cohort["status"],
               "previous_limit_feedback_coverage":cohort["coverage"],
               **promotion_rates(previous, pool)}
        values = [r.get("space_feedback", np.nan) for r in history[-2:]]+[row["space_feedback"]]
        row["space_feedback_3d"] = float(pd.Series(values).mean())
        row["breadth_height_divergence"] = float(row["height_change"] > 0 and row["limit_change"] < -20)
        row.update(rolling_emotion_context(history, row))
        row.update(emotion_dimensions(row))
        row["emotion_score"] = row["emotion_quality_score"]
        row["emotion_state"] = classify_emotion(row, previous_market.get("emotion_state"))
        row["data_status"] = "verified" if (row["emotion_data_completeness"] >= .9 and previous is not None
            and pd.notna(row["height_change"]) and pd.notna(row["limit_change"])
            and cohort["status"] == "EXACT_PREVIOUS_CLOSING_POOL_COVERED") else "partial"
        row["scope_status"] = "PROVIDER_REPORTED_NOT_EXCHANGE_CERTIFIED"
        rows.append(row); history.append(row)
    return pd.DataFrame(rows), audit


def frozen_model(frame, fields, channel):
    known = known_before(frame, CUTOFF)
    dates = sorted(known.date.unique()); val_start = dates[-VALIDATION_DATES]
    train = known[(known.date < val_start) & (known.target_date < val_start)]
    validation = known[known.date >= val_start]
    options = []
    for kind in KINDS:
        for window in WINDOWS:
            subset = trailing(train, window)
            mdl = fit(subset, fields, kind, channel)
            score, n = selection_metric(prediction(mdl, validation, fields, channel), channel)
            options.append({"kind": kind, "window": window, "validation_accuracy": score, "validation_dates": n})
    chosen = sorted(options, key=lambda r:(-r["validation_accuracy"], KINDS.index(r["kind"]), -(r["window"] or 100000)))[0]
    refit = trailing(known, chosen["window"])
    if refit.target_date.max() >= CUTOFF or validation.target_date.max() >= CUTOFF:
        raise ValueError("September label leaked into model selection/refit")
    log = {"chosen": chosen, "validation_candidates": options,
           "max_refit_label_date": refit.target_date.max(), "max_validation_label_date": validation.target_date.max(),
           "refit_dates": int(refit.date.nunique()), "validation_start": val_start,
           "september_labels_used": False, "confidence_status": "UNCALIBRATED_AFTER_REFIT"}
    return fit(refit, fields, chosen["kind"], channel), log


def relay_features(pools, market, history):
    by_market = market.set_index("date")
    # V3's dictionary persists across sessions: this feature compares with the
    # last OBSERVED relay-pool appearance, not necessarily the previous day.
    # Initialize before the first reported pool to avoid feeding a later row
    # back into an earlier overlap reconstruction.
    previous = {}
    first_date = pools.date.min()
    for r in history[history.date < first_date].sort_values(["date","code"]).itertuples():
        if pd.notna(r.turnover):
            previous[r.code] = (r.date,r.turnover*100)
    rows = []
    for day,g in pools.groupby("date",sort=True):
        ladder = g[g.boards.ge(2)]; height = g.boards.max(); count = ladder.groupby("boards").size()
        for r in ladder.itertuples():
            prior_date, prior = previous.get(r.code,(None,np.nan))
            m = by_market.loc[day]
            row = {"date": day,"target_date":r.target_date,"code":r.code,"name":r.name,"boards":r.boards,
                   "target":r.promoted_next_day,"label_valid":pd.notna(r.promoted_next_day),
                   "height_gap":height-r.boards,"same_height_rivals":int(count[r.boards])-1,
                   "unique_space":float(r.boards==height and count[r.boards]==1),"height_share":r.boards/height,
                   "is_space_leader":float(r.boards==height),"turnover":r.turnover/100,
                   "turnover_vs_prev":r.turnover/prior if prior>0 else np.nan,"price":r.price,
                   "turnover_reference_date":prior_date,
                   "turnover_ratio_definition":"LAST_OBSERVED_RELAY_POOL_TURNOVER_NOT_PREVIOUS_SESSION",
                   "signal_amount":r.amount,"amplitude":r.amplitude,"one_word_proxy":r.one_word_proxy,
                   "actual_one_word_daily":r.actual_one_word_daily,"ohlc_status":r.ohlc_status,
                   "source_available_at":r.captured_at,"market_data_status":m.data_status}
            for k in ["emotion_score","promotion_2to3","promotion_3to4","promotion_mid","previous_limit_tail_loss",
                      "failed_rate","breadth_health","ladder_health","loss_effect_health","crowding_risk","regime_fragility"]:
                row[k] = m.get(k,np.nan)
            row["down_limit_rate"] = np.clip(m.get("down_limit_count",np.nan)/10,0,1)
            rows.append(row)
            if pd.notna(r.turnover):
                previous[r.code] = (day,r.turnover)
    frame = pd.DataFrame(rows)
    for k in ["price","signal_amount","turnover","turnover_vs_prev"]:
        frame[k+"_same_height_percentile"] = frame.groupby(["date","boards"])[k].rank(pct=True)
    frame["signal_one_word"] = frame.one_word_proxy
    frame["log_signal_amount"] = np.log1p(frame.signal_amount.clip(lower=0))
    return frame


def relay_summary(pred):
    complete = pred.groupby("date").label_valid.all()
    evaluated = pred[pred.date.isin(complete[complete].index)].copy()
    output = {"dates_total":int(pred.date.nunique()), "dates_with_complete_next_pool":int(evaluated.date.nunique())}
    for key, mask in [("identity",pd.Series(True,index=evaluated.index)),
                      ("legacy_non_one_word_proxy",evaluated.one_word_proxy.eq(0)),
                      ("ohlc_non_one_word",evaluated.actual_one_word_daily.eq(0))]:
        selected = top1(evaluated[mask]).copy(); selected["correct"] = selected.target.astype(int)
        output[key] = rate(selected)
    return output


def run(output):
    target = Path(output); target.mkdir(parents=True,exist_ok=False)
    daily = verify_archive(DAILY.parent)
    pools = attach_ohlc(pd.read_csv(ARCHIVE,dtype={"code":str}),daily)
    base = pd.read_csv(DATA / "market_emotion.csv")
    rebuilt, source_audit = rebuild_market(pools,base)
    rebuilt.to_csv(target / "reported_market_rows.csv",index=False,encoding="utf-8-sig")
    overlap = base[base.date.isin(["2026-08-28","2026-08-31"])].merge(rebuilt,on="date",suffixes=("_old","_reported"))
    overlap.to_csv(target / "overlap_source_drift.csv",index=False,encoding="utf-8-sig")
    combined = pd.concat([base[base.date<CUTOFF],rebuilt[rebuilt.date>=CUTOFF]],ignore_index=True)
    market_path = target / "combined_market_input.csv"; combined.to_csv(market_path,index=False)
    market, fields = prepare_market(market_path)
    index = pd.read_csv(INDEX_PATH)[["date","sh_a_return_pct","sz_a_return_pct"]]
    bars = pd.read_csv(DAILY).sort_values(["symbol","date"])
    bars["return_pct"] = bars.groupby("symbol").close.pct_change()*100
    new_index = bars[bars.symbol.isin(["sh000002","sz399107"])].pivot(index="date",columns="symbol",values="return_pct").reset_index()
    new_index = new_index.rename(columns={"sh000002":"sh_a_return_pct","sz399107":"sz_a_return_pct"})
    index = pd.concat([index[index.date<CUTOFF],new_index[new_index.date>=CUTOFF]],ignore_index=True)
    market = market.merge(index,on="date",how="left",validate="one_to_one").sort_values("date").reset_index(drop=True)
    for k in ["sh_a_return_pct","sz_a_return_pct"]:
        fields.append(k)
        for lag in [1,3]:
            market[k+f"_lag_{lag}"] = market[k].shift(lag); fields.append(k+f"_lag_{lag}")
    market["target"] = direction(market.emotion_quality_score.shift(-1)-market.emotion_quality_score,.05)
    successor = dict(zip(sessions()[:-1],sessions()[1:]))
    actual_next = market.date.shift(-1)
    september = market.date.ge(CUTOFF)
    market.loc[september,"target_date"] = market.loc[september,"date"].map(successor)
    market.loc[september,"label_valid"] &= actual_next[september].eq(market.loc[september,"target_date"])
    mdl, mlog = frozen_model(market,fields,"market")
    test = market[september & market.limit_count.notna()].copy()
    mp = prediction(mdl,test,fields,"market"); valid = mp[mp.label_valid & mp.target.notna()]
    mreport = {"dates_total":len(mp),"complete_next_session_dates":len(valid),
               "complete_source_forecasts":metrics(valid.target,valid.predicted)}
    mreport["all_three_class_support"] = {k:int(valid.target.eq(k).sum()) for k in ["IMPROVE","CONTINUE","WORSEN"]}
    mreport["balanced_accuracy_scope"] = "OBSERVED_CLASSES_ONLY; absent classes are untested"
    historical, relay_fields = prepare_relay(DATA / "leader_candidates.csv",base.date.tolist())
    rf = relay_features(pools,rebuilt,historical)
    rt = rf[rf.date.ge(CUTOFF)].copy()
    mdl, rlog = frozen_model(historical,relay_fields,"relay")
    rp = prediction(mdl,rt,relay_fields,"relay")
    for p in [mp,rp]:
        p["evidence_class"] = "HISTORICAL_REPLAY_NOT_FORWARD"
        p["model_freeze"] = CUTOFF
    mp.to_csv(target / "market_predictions_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    rp.to_csv(target / "relay_predictions_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    report = {"version":VERSION,"status":"RETROSPECTIVE_FALSIFICATION_ONLY","goal_verified":False,
              "market":mreport,"relay":relay_summary(rp),"selection_logs":{"market":mlog,"relay":rlog},
              "source_audit":source_audit,"calendar":{"sessions":sessions(),"sources":CALENDAR_SOURCES},
              "ohlc_agreement":{"relay_rows":int(pools.boards.ge(2).sum()),
                  "agreed_rows":int((pools.boards.ge(2)&pools.ohlc_status.eq('CLOSE_AGREES_UNADJUSTED_PROVIDER_BAR')).sum())},
              "missing_feature_rates":{k:float(rt[k].isna().mean()) for k in relay_fields},
              "coverage":{"calendar_signal_sessions":int(sum(CUTOFF<=d<="2026-09-30" for d in sessions())),
                  "observed_market_sessions":len(mp),"market_evaluable_sessions":len(valid),
                  "observed_relay_sessions":int(rp.date.nunique()),
                  "missing_pool_signal_dates":[d for d in sessions() if CUTOFF<=d<="2026-09-30" and d not in set(pools.date)]},
              "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [ARCHIVE,DAILY,DAILY.parent/'report.json',Path(__file__),
                  ROOT/'historical_ohlc_archive_v278.py',ROOT/'rolling_forecast_challenger_v274.py',
                  ROOT/'leader_system_v3.py',DATA/'market_emotion.csv',DATA/'leader_candidates.csv',INDEX_PATH]},
              "limitations":["September history previously inspected; no pristine blind or forward registration",
                  "Provider pools not exchange-certified, historical security universe still incomplete",
                  "Reconstructed OHLC loss fields vs reported pools can shift definitions; overlap drift retained",
                  "Missing sessions remain unknown; no shift across missing pools",
                  "Previous-limit feedback is filtered to exact prior closing-pool codes; incomplete coverage marks partial",
                  "Unadjusted Tencent daily bars are retrospectively received; no minute, queue or fill evidence",
                  "Legacy one-word proxy also requires turnover under 3%; pure OHLC definition reported separately",
                  "Legacy turnover_vs_prev compares with last observed relay-pool appearance, not necessarily previous session",
                  "Small available September denominator cannot prove stable 60%"],"trade_authorized":False}
    (target / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {k:report[k] for k in ["market","relay","ohlc_agreement"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    output_was_present = Path(args.output).exists()
    try:
        print(json.dumps(run(args.output),ensure_ascii=True))
    except Exception as error:
        failure = Path(args.output) / "FAILED.json"
        if not output_was_present and failure.parent.exists() and not failure.exists():
            failure.write_text(json.dumps({"status":"FAILED_NOT_A_RESULT","error":str(error)},ensure_ascii=False,indent=2),encoding="utf-8")
        raise
