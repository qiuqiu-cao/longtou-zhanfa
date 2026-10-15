"""One fixed two-arm causal-feature experiment; never refit protected BASE.

Monthly expanding clock / trailing252 eligible signal dates. This is a pointwise
ranking surrogate with explicit nonword channel training and daily TopK tests.
All old experiments stay immutable; no parameter selection on their outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import model_upgrade_history as history
from current_model_diagnostics import RELAY_FIELDS
import short_term_thinking_dataset_builder as b

ARMS = ("ST_B_ONLY", "ST_DUAL")
BASE_ROOT = b.ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z"
FEATURES = [k for k in RELAY_FIELDS if k != "signal_one_word"] + b.DYNAMIC_FIELDS
PARAMS = dict(learning_rate=.06, max_iter=120, max_leaf_nodes=7, max_depth=3,
    min_samples_leaf=25, l2_regularization=3., early_stopping=False, random_state=20261015)
SCREEN = dict(september_nonword_recall_gain_min=.10, nonword_f1_not_lower_each_fold=True,
    top1_hit_gain_over_height_and_base_september=True, top2_hit_not_lower_september=True,
    precision_decline_limit=.05, paired_recall_interval_lower_positive=True,
    b1_large_loss_rate_not_higher=True, b2_mean_net_return_higher=True,
    execution_proxy_min_matured=10, execution_proxy_min_coverage=.5,
    old_all_pool_accuracy_plus_3pp_veto=False, independent_role_accuracy_required_for_identity_claim=True,
    research_screen_is_not_live_production_promotion=True)


def data_dir(out):
    return Path(out) / "historical_final" if (Path(out) / "historical_final/dataset.joblib").exists() else Path(out)


def merged(bundle):
    original, features = bundle["prepared"]["relay"], bundle["features"]
    extras = features.drop(columns=["name", "boards"])
    overlap = (set(extras) & set(original)) - {"date", "code"}
    result = original.drop(columns=list(overlap)).merge(extras, on=["date", "code"], validate="one_to_one")
    if len(original) != len(result):
        raise ValueError("original cohort changed")
    for key in ("target", "target_date", "label_valid") + tuple(RELAY_FIELDS):
        pd.testing.assert_series_equal(original.sort_values(["date", "code"])[key].reset_index(drop=True), result.sort_values(["date", "code"])[key].reset_index(drop=True))
    return result


def freeze(output_dir=b.DEFAULT_RUN):
    out = Path(output_dir)
    data = data_dir(out)
    report = b.read(data / "coverage_and_role_distribution.json")
    b.verify(report["source_bindings"])
    if b.sha(data / "dataset.joblib") != report["dataset_sha256"]:
        raise ValueError("feature dataset hash changed")
    bundle = joblib.load(data / "dataset.joblib")
    full = merged(bundle)
    sources = dict(report["source_bindings"])
    for path in [data / "dataset.joblib", data / "coverage_and_role_distribution.json", Path(__file__),
                 BASE_ROOT / "relay_history/predictions.csv", BASE_ROOT / "models/relay/BASE/model.joblib",
                 BASE_ROOT / "models/relay/BASE/manifest.json", b.RECENCY / "full_candidate_predictions.csv"]:
        sources[str(path.resolve())] = b.sha(path)
    if (out / "current_extension_receipt.json").exists():
        current = b.read(out / "current_extension_receipt.json")
        b.verify(current["source_bindings"])
        sources.update(current["source_bindings"])
        for path in [out / "current_extension.joblib", out / "current_extension_receipt.json"]:
            sources[str(path.resolve())] = b.sha(path)
    training = []
    for period in pd.period_range("2025-07", "2026-10", freq="M"):
        cutoff = str(period.start_time.date())
        train = history.training_rows(full, cutoff, "relay")
        if train.date.nunique() != 252 or train.target_date.max() >= cutoff:
            raise ValueError("same252 past-only training dates required")
        if period.month == 10 and period.year == 2026:
            test_n = 0  # October runs separately from archived source feature adapters.
        else:
            test_n = int(full.date.str[:7].eq(str(period)).sum())
        training.append(dict(month=str(period), cutoff=cutoff, complete_train_rows=len(train), train_days=252,
            max_target=train.target_date.max(), b_train_rows=int(train.daily_one_word.eq(0).sum()), test_rows=test_n,
            input_target_digest=history._frame_digest(train, FEATURES)))
    p = dict(version=b.VERSION, formed_at=b.now(), authorization_path=str((out / "user_authorization.txt").resolve()),
        authorization_sha256=b.sha(out / "user_authorization.txt"), hypothesis="Chronological parent/PK/turnover/cycle relations may improve real nonword prediction and relative ranking.",
        challengers=list(ARMS), features=FEATURES, estimator=PARAMS, training_checks=training, maximum_new_fits=32,
        fitting_clock="Month start; signals and exact official target strictly before cutoff; last252 eligible full-pool dates before channel filtering",
        channel_b_members="ST_B_ONLY: daily OHLC nonword==0, never old low-turnover proxy. ST_DUAL: full same252 pool.",
        class_weight_policy="Each included signal day starts with mass1, then training-only positive/negative class mass each0.5; normalize to included-day count.",
        channel_a="Exact daily one-price boards keep original immutable BASE score; monitoring only.",
        missing_identity="Retain in full pool and route UNRESOLVED_IDENTITY; not part of real-nonword denominator. ST_DUAL may train with explicit missing indicator.",
        numeric_scores="Uncalibrated internal classifier/ranking scores; strictly >0.5 decisions, no threshold tuning.",
        fixed_screen=SCREEN, folds=history.FOLDS, block_interval=dict(block_sessions=5, draws=2000, seed=20261015, confidence=.95),
        execution_proxy="B=T+1 non-upper-limit OPEN with 0<=gap<=7%; fixed adverse entry/exit slippage0.1% each plus round-trip cost0.3%; exit B+1/B+2 at close, exact one-price lower-limit sell blocked -> unresolved mark separately. Daily open proxy cannot prove auction/queue/intraday entry.",
        source_time_policy="Retrospective event-date causal research authorized now; no claim archives were collected before historical cutoff.",
        role_labels="Derived state machine features, not independent targets. Train only original promotion labels. Independent six-role recognition accuracy remains unmeasurable.",
        stop="One fit per month per arm, fitting phase <=30min. No parameter/threshold retuning after results; failed fit never repeated.",
        production_base_changed=False, old_ledgers_changed=False, new_forward_predictions=0,
        historical_data_previously_inspected=True, sources=sources,
        runtime_versions={k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scikit-learn", "joblib")})
    b.write_new(out / "protocol_BEFORE_ANY_FIT.json", p)
    return p


def training_weights(train):
    if train.empty or train.target.nunique() != 2:
        raise ValueError("both training target classes required")
    w = 1 / train.groupby("date").date.transform("size")
    mass = w.groupby(train.target).sum()
    w = w * train.target.map({k: .5 / v for k, v in mass.items()})
    return w * train.date.nunique()


def training_subset(full, cutoff, arm):
    if arm not in ARMS:
        raise ValueError("unregistered challenger")
    train = history.training_rows(full, cutoff, "relay")
    dates = sorted(train.date.unique())
    if len(dates) != 252 or train.date.ge(cutoff).any() or train.target_date.ge(cutoff).any():
        raise ValueError("future training target or missing252 full-pool dates")
    return train[train.daily_one_word.eq(0)].copy() if arm == "ST_B_ONLY" else train


def fit_month(full, expected, arm, out):
    month, cutoff = expected["month"], expected["cutoff"]
    original_train = history.training_rows(full, cutoff, "relay")
    if history._frame_digest(original_train, FEATURES) != expected["input_target_digest"]:
        raise ValueError("frozen pre-month input or labels changed")
    train = training_subset(full, cutoff, arm)
    prefix = out / (month + "_" + arm)
    b.write_new(str(prefix) + "_fit_STARTED.json", dict(formed_at=b.now(), arm=arm, month=month,
        train_rows=len(train), original_full_pool_training_dates=252, train_max_target=train.target_date.max(),
        training_digest=history._frame_digest(train, FEATURES), attempt=1))
    model = HistGradientBoostingClassifier(**PARAMS)
    with threadpool_limits(limits=2):
        model.fit(train[FEATURES], train.target.astype(int), sample_weight=training_weights(train))
    joblib.dump(dict(model=model, features=FEATURES, cutoff=cutoff, arm=arm,
        estimator=PARAMS, training_rows=len(train), trained_at=b.now(), role_version=b.VERSION), str(prefix) + "_model.joblib")
    return model


def metrics(frame):
    scored = frame[frame.target.notna() & frame.label_valid.fillna(False).astype(bool) & frame.score.notna()]
    if scored.empty:
        return dict(candidates=len(frame), scored=0, Accuracy=None, Recall=None, Precision=None, F1=None)
    y, p = scored.target.eq(1), scored.score.gt(.5)
    tp, fp, fn, tn = [int(v.sum()) for v in (y & p, ~y & p, y & ~p, ~y & ~p)]
    return dict(candidates=len(frame), scored=len(scored), positives=int(y.sum()), positive_calls=int(p.sum()),
        tp=tp, fp=fp, fn=fn, tn=tn, Accuracy=(tp+tn)/len(scored), Recall=tp/(tp+fn) if tp+fn else None,
        Precision=tp/(tp+fp) if tp+fp else None, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        label_coverage=len(scored)/len(frame) if len(frame) else None)


def top_selection(frame, k=1, cohort=False, height=False):
    group = ["date", "boards"] if cohort else ["date"]
    f = frame[frame.daily_one_word.eq(0)].copy()
    # A tie is resolved deterministically using code; no realized-label tie pick.
    order = group + (["boards"] if height and not cohort else []) + ([] if height else ["score"]) + ["code"]
    asc = [True] * len(group) + ([False] if height and not cohort else []) + ([] if height else [False]) + [True]
    f = f.sort_values(order, ascending=asc).groupby(group, sort=True).head(k)
    return f


def top_metrics(frame, k=1, cohort=False, height=False):
    selected = top_selection(frame, k, cohort, height)
    valid = selected[selected.target.notna() & selected.label_valid.fillna(False).astype(bool)]
    group = ["date", "boards"] if cohort else ["date"]
    hits = valid.groupby(group).target.max()
    return dict(selected=len(selected), known=len(valid), groups=int(selected.groupby(group).ngroups),
        candidate_hit_rate=float(valid.target.mean()) if len(valid) else None,
        group_any_hit_rate=float(hits.mean()) if len(hits) else None,
        groups_with_unresolved_truth=int(selected.groupby(group).target.apply(lambda s: s.isna().any()).sum()))


def paired_block(left, right, field="recall", draws=2000):
    key = ["date", "code"]
    f = left.merge(right[key + ["score"]], on=key, suffixes=("", "_other"), validate="one_to_one")
    f = f[f.daily_one_word.eq(0) & f.target.notna() & f.label_valid.fillna(False).astype(bool)]
    if field == "recall":
        f = f[f.target.eq(1)]
        f["base"] = f.score.gt(.5).astype(int)
        f["challenger"] = f.score_other.gt(.5).astype(int)
    else:
        a = top_selection(left, 1)[key]
        z = top_selection(right, 1)[key]
        aa = left.merge(a, on=key).set_index("date").target
        zz = right.merge(z, on=key).set_index("date").target
        f = pd.DataFrame({"base": aa, "challenger": zz}).dropna().reset_index()
    if not len(f):
        return dict(lower=None, upper=None, pairs=0)
    values = f.groupby("date").agg(n=("base", "size"), base=("base", "sum"), challenger=("challenger", "sum")).to_numpy(float)
    n = len(values); block = min(5, n)
    rng = np.random.default_rng(20261015)
    starts = rng.integers(0, n, (draws, (n+block-1)//block))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    s = values[index].sum(axis=1)
    delta = (s[:, 2] - s[:, 1]) / s[:, 0]
    return dict(lower=float(np.quantile(delta, .0125)), upper=float(np.quantile(delta, .9875)),
        pairs=len(f), draws=draws, unit="five observed signal-date circular blocks", family_multiple_testing_adjusted=True,
        adjustment="two fixed arms Bonferroni tails; exploratory auxiliary metrics not confirmatory")


def proxy_returns(frame, bundle):
    """Daily OPEN scenario, never an actual fill and never a target-day rank."""
    sessions = bundle["sessions"]; index = {d:i for i,d in enumerate(sessions)}
    codes = set(frame.code)
    bars = bundle["bars"]
    bars = bars[bars.code.isin(codes)]
    bi = {(r["date"], r["code"]):r for r in bars.to_dict("records")}
    logs = []
    for r in frame.to_dict("records"):
        i = index[r["date"]]
        dates = sessions[i+1:i+4]
        record = dict(date=r["date"], code=r["code"], status="MISSING_B_DAY", b1_net=None, b2_net=None, b1_mark=None, b2_mark=None)
        if len(dates) != 3:
            logs.append(record); continue
        signal, entry = bi.get((r["date"],r["code"])), bi.get((dates[0],r["code"]))
        if not signal or not entry:
            logs.append(record); continue
        try:
            if cents_equal_limit_open(entry, signal):
                record["status"] = "NO_FILL_UPPER_LIMIT_OPEN_PROXY"
                logs.append(record); continue
            gap = entry["open"] / signal["close"] - 1
            if b.finite(entry.get("volume")) is not None and entry["volume"] <= 0:
                record["status"] = "NO_VOLUME_NO_FILL_PROXY"
                logs.append(record); continue
            if not 0 <= gap <= .07:
                record["status"] = "CANCEL_GAP_OUTSIDE_0_TO_7PCT"
                logs.append(record); continue
            entry_price = entry["open"] * 1.001
            record.update(status="INFERRED_OPEN_PROXY_NOT_ACTUAL_FILL", B=dates[0], entry_open=entry["open"], fee_roundtrip=.003)
            prev = entry
            for h in (1, 2):
                exitbar = bi.get((dates[h],r["code"]))
                if not exitbar:
                    record["status_b"+str(h)] = "MISSING_EXIT_BAR"
                    prev = None
                    continue
                record["b"+str(h)+"_mark"] = exitbar["close"] / entry_price - 1
                down_cap = int((Decimal(str(prev["close"])) * 90).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) if prev else None
                shape = b.price_state(exitbar)
                if down_cap is not None and shape["one_word"] == 1 and b.cents(exitbar["close"]) == down_cap:
                    record["status_b"+str(h)] = "SELL_BLOCKED_ONE_PRICE_DOWN_LIMIT"
                elif prev:
                    record["b"+str(h)+"_net"] = exitbar["close"] * .999 / entry_price - 1 - .003
                    record["status_b"+str(h)] = "INFERRED_CLOSE_EXIT_PROXY"
                prev = exitbar
        except (ValueError, TypeError, KeyError):
            record["status"] = "INVALID_DAILY_PRICE"
        logs.append(record)
    return pd.DataFrame(logs)


from decimal import Decimal, ROUND_HALF_UP


def cents_equal_limit_open(entry, signal):
    return b.cents(entry["open"]) >= b.cap_cents(signal["close"])


def execution_metrics(log):
    out = dict(selected=len(log), entry_status_counts=log.status.value_counts().to_dict() if len(log) else {}, actual_fills=0)
    for h in (1, 2):
        s = pd.to_numeric(log.get("b"+str(h)+"_net", pd.Series(dtype=float)), errors="coerce").dropna()
        mark = pd.to_numeric(log.get("b"+str(h)+"_mark", pd.Series(dtype=float)), errors="coerce").dropna() - .003
        status = log.get("status_b"+str(h), pd.Series(dtype=str))
        out["b"+str(h)] = dict(matured=len(s), selected_coverage=len(s)/len(log) if len(log) else None,
            mean_net=float(s.mean()) if len(s) else None, loss_ge7pct=float(s.le(-.07).mean()) if len(s) else None,
            marked_candidates=len(mark), risk_loss_ge7pct_including_blocked=float(mark.le(-.07).mean()) if len(mark) else None,
            sell_blocked=int(status.eq("SELL_BLOCKED_ONE_PRICE_DOWN_LIMIT").sum()),
            median_net=float(s.median()) if len(s) else None, worst=float(s.min()) if len(s) else None,
            cumulative_return=None, portfolio_drawdown=None, note="Equal selected-candidate scenario distribution, not an executable portfolio backtest")
    return out


def evaluate(predictions, bundle, out):
    comparisons = {}
    folds = history.FOLDS + [dict(id="2026Oct", start="2026-10-01", end="2026-10-31")]
    for fold in folds:
        f = predictions[predictions.date.between(fold["start"], fold["end"])]
        if f.empty:
            continue
        arms = {arm: f[f.arm.eq(arm)].copy() for arm in ("BASE", "ROLL252") + ARMS if f.arm.eq(arm).any()}
        report = {}
        execution_logs = []
        for arm, part in arms.items():
            nonword = part[part.daily_one_word.eq(0)]
            raw_nonword = nonword[nonword.actual_one_word_daily.eq(0)] if "actual_one_word_daily" in nonword else nonword.iloc[:0]
            selected = top_selection(part)
            log = proxy_returns(selected, bundle)
            log["arm"] = arm
            execution_logs.append(log)
            report[arm] = dict(full=metrics(part), observed_nonword=metrics(nonword), original_raw_qualified_nonword=metrics(raw_nonword),
                oneword=metrics(part[part.daily_one_word.eq(1)]), unknown_identity=metrics(part[part.daily_one_word.isna()]),
                top1=top_metrics(part), top2=top_metrics(part, 2), same_height_top1=top_metrics(part, 1, True),
                same_height_top2=top_metrics(part, 2, True), height_baseline_top1=top_metrics(part, height=True),
                height_baseline_top2=top_metrics(part, 2, height=True), execution_proxy=execution_metrics(log))
        intervals = {}
        for arm in ARMS:
            intervals[arm] = dict(nonword_recall=paired_block(arms["BASE"], arms[arm]), top1=paired_block(arms["BASE"], arms[arm], "top1"))
        comparisons[fold["id"]] = dict(metrics=report, intervals=intervals)
        pd.concat(execution_logs, ignore_index=True).to_csv(out / (fold["id"] + "_all_top1_execution_proxy.csv"), index=False, encoding="utf-8-sig")
    screens = {}
    for arm in ARMS:
        guards = {}
        for fold_id, value in comparisons.items():
            if fold_id == "2026Oct":
                continue  # Small, already seen October is disclosed separately.
            bm, cm = value["metrics"]["BASE"], value["metrics"][arm]
            a, z = bm["observed_nonword"], cm["observed_nonword"]
            guards[fold_id + "_f1"] = a["F1"] is not None and z["F1"] is not None and z["F1"] >= a["F1"]
            guards[fold_id + "_precision"] = a["Precision"] is not None and z["Precision"] is not None and z["Precision"] >= a["Precision"] - .05
        sep = comparisons["2026Sep"]; bm, cm = sep["metrics"]["BASE"], sep["metrics"][arm]
        a, z = bm["observed_nonword"], cm["observed_nonword"]
        guards["september_recall_gain"] = z["Recall"] is not None and a["Recall"] is not None and z["Recall"] - a["Recall"] >= .10
        guards["september_top1_over_base_and_height"] = cm["top1"]["candidate_hit_rate"] > bm["top1"]["candidate_hit_rate"] and cm["top1"]["candidate_hit_rate"] > cm["height_baseline_top1"]["candidate_hit_rate"]
        guards["september_top2"] = cm["top2"]["candidate_hit_rate"] >= bm["top2"]["candidate_hit_rate"]
        guards["paired_recall_interval_lower_positive"] = sep["intervals"][arm]["nonword_recall"]["lower"] is not None and sep["intervals"][arm]["nonword_recall"]["lower"] > 0
        ex = cm["execution_proxy"]; be = bm["execution_proxy"]
        guards["execution_evidence_sufficient"] = all(ex[k]["matured"] >= SCREEN["execution_proxy_min_matured"] and ex[k]["selected_coverage"] >= SCREEN["execution_proxy_min_coverage"] and be[k]["matured"] >= SCREEN["execution_proxy_min_matured"] and be[k]["selected_coverage"] >= SCREEN["execution_proxy_min_coverage"] for k in ("b1", "b2"))
        guards["b1_loss_not_higher"] = ex["b1"]["risk_loss_ge7pct_including_blocked"] is not None and be["b1"]["risk_loss_ge7pct_including_blocked"] is not None and ex["b1"]["risk_loss_ge7pct_including_blocked"] <= be["b1"]["risk_loss_ge7pct_including_blocked"]
        guards["b2_expected_return_higher"] = ex["b2"]["mean_net"] is not None and be["b2"]["mean_net"] is not None and ex["b2"]["mean_net"] > be["b2"]["mean_net"]
        screens[arm] = dict(guards=guards, retrospective_research_screen_pass=all(guards.values()), production_deployment_authorized=False,
            six_role_independent_accuracy=None, role_gold_rows=0)
    return dict(version=b.VERSION, formed_at=b.now(), comparisons=comparisons, fixed_screen=SCREEN, arm_verdicts=screens,
        actual_new_fits=len(list(out.glob("*_fit_STARTED.json"))), base_refits=0, new_formal_forecasts=0, new_forward_samples=0,
        no_calibrated_probability_claim=True, retrospective_not_blind=True)


def run(output_dir=b.DEFAULT_RUN):
    out = Path(output_dir); p = b.read(out / "protocol_BEFORE_ANY_FIT.json")
    if p["estimator"] != PARAMS or p["features"] != FEATURES or p["fixed_screen"] != SCREEN:
        raise ValueError("frozen experiment definition changed")
    b.verify(p["sources"])
    b.write_new(out / "experiment_STARTED.json", dict(formed_at=b.now(), protocol_sha256=b.sha(out / "protocol_BEFORE_ANY_FIT.json"), no_repeated_fits=True))
    started = datetime.now(timezone.utc)
    bundle = joblib.load(data_dir(out) / "dataset.joblib"); full = merged(bundle)
    base = pd.read_csv(BASE_ROOT / "relay_history/predictions.csv", dtype={"code":str})
    base = base[base.arm.eq("BASE")][["date", "code", "raw_score"]]
    # Saved monthly ROLL scores are a fair refresh baseline, not another refit.
    roll = pd.read_csv(b.RECENCY / "full_candidate_predictions.csv", dtype={"code":str})
    roll = roll[roll.arm.eq("ROLL252")][["signal_date", "code", "raw_score"]].rename(columns={"signal_date":"date"})
    all_parts = []
    original_test = full[full.date.between("2025-07-01", "2026-09-30")].copy()
    for name, saved in (("BASE", base), ("ROLL252", roll)):
        part = original_test.merge(saved.rename(columns={"raw_score":"score"}), on=["date", "code"], validate="one_to_one")
        if len(part) != len(original_test):
            raise ValueError("complete original baseline cohort required")
        part["arm"] = name; all_parts.append(part)
    for expected in p["training_checks"]:
        month = expected["month"]
        if month == "2026-10":
            test = joblib.load(out / "current_extension.joblib")["full"] if (out / "current_extension.joblib").exists() else full.iloc[:0]
            if not test.empty:
                extension = joblib.load(out / "current_extension.joblib")
                test = test[test.date.ge("2026-10-01")].copy()
                bundle = {**bundle, "bars": extension["bars"]}
                for name in ("BASE", "ROLL252"):
                    part = test.copy(); part["score"] = part["base_score"]; part["arm"] = name
                    all_parts.append(part)
        else:
            test = full[full.date.str[:7].eq(month)].copy()
        for arm in ARMS:
            if (datetime.now(timezone.utc) - started).total_seconds() > 1800:
                b.write_new(out / "FIT_BUDGET_STOP.json", dict(formed_at=b.now(), month=month, arm=arm)); raise TimeoutError("30min fitting budget")
            try:
                model = fit_month(full, expected, arm, out)
                if test.empty:
                    continue
                with threadpool_limits(limits=2):
                    raw_b = model.predict_proba(test[FEATURES])[:, list(model.classes_).index(1)]
                part = test.copy(); part["b_score"] = raw_b
                bm = base.set_index(["date","code"]).raw_score
                base_scores = test.base_score.to_numpy() if "base_score" in test else np.asarray([bm.get((r.date,r.code), np.nan) for r in test.itertuples()])
                part["score"] = np.where(part.daily_one_word.eq(1), base_scores, raw_b)
                part["channel"] = np.where(part.daily_one_word.eq(1), "A_EMOTION_ANCHOR", np.where(part.daily_one_word.eq(0), "B_NONWORD", "UNRESOLVED_IDENTITY"))
                part["arm"] = arm; part["model_cutoff"] = expected["cutoff"]
                keep = ["date","code","name","boards","target_date","target","label_valid","daily_one_word","actual_one_word_daily","primary_role","cycle_regime","score","b_score","channel","arm","model_cutoff"]
                part[keep].to_csv(out / (month+"_"+arm+"_predictions_BEFORE_EVALUATION.csv"), index=False, encoding="utf-8-sig")
                # Outputs are durable before comparison or interpretation.
                all_parts.append(part)
                print(month, arm, "fit saved; predictions", len(part), flush=True)
            except Exception as exc:
                b.write_new(out / "FIT_FAILURE_STOP.json", dict(formed_at=b.now(), month=month, arm=arm, error=repr(exc), repeated_fit_allowed=False))
                raise
    predictions = pd.concat(all_parts, ignore_index=True)
    keep = [k for k in ["date","code","name","boards","target_date","target","label_valid","daily_one_word","actual_one_word_daily","price_identity_source_tier","primary_role","roles","cycle_regime","parent_anchor_code","parent_anchor_state","score","b_score","channel","arm","model_cutoff"] if k in predictions]
    predictions[keep].to_csv(out / "complete_paired_predictions.csv", index=False, encoding="utf-8-sig")
    result = evaluate(predictions, bundle, out)
    b.write_new(out / "model_validation_results.json", result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("freeze","run"));p.add_argument("--output-dir",default=str(b.DEFAULT_RUN))
    a=p.parse_args(argv);r=globals()[a.command](a.output_dir)
    print(b.clean({k:r.get(k) for k in ("formed_at","actual_new_fits","arm_verdicts")}))


if __name__ == "__main__":
    main()
