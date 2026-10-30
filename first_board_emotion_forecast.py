"""Strictly forward, day-level promotion-rate forecast; not an emotion gold label."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import first_board_joint_seal_model as prior
import first_board_ladder_emotion_features as features

base, stage = prior.base, prior.stage
FIELD = "emotion_next_first_promotion_estimate"
TARGET = "first_pool_promotion_rate_TARGET_ONLY"
MIN_DAYS = 63
WINDOW = 252


def macro_table(rows, context):
    """Keep every complete original signal-day cohort and its unknown denominator."""
    if rows.duplicated(prior.KEYS).any():
        raise ValueError("Duplicate stock/day")
    if context.signal_date.duplicated().any():
        raise ValueError("Duplicate context date")
    g = rows.groupby("signal_date", sort=True)
    if g.label_end.nunique().gt(1).any() or g.p.nunique().gt(1).any():
        raise ValueError("Conflicting target clocks")
    if g.previous_first_promotion.nunique().gt(1).any():
        raise ValueError("Naive reference is not day-level")
    daily = g.agg(p=("p", "first"), label_end=("label_end", "first"),
                  members=("code", "size"), known=(stage.TARGET, "count"),
                  target=(stage.TARGET, "mean"),
                  naive_last_mature_promotion=("previous_first_promotion", "first")).reset_index()
    daily["outcome_coverage"] = daily.known / daily.members
    daily[TARGET] = daily.target.where(daily.outcome_coverage.ge(.95))
    daily = daily.drop(columns="target").merge(context[["signal_date"] + features.FIELDS],
                         on="signal_date", how="left", sort=False, validate="one_to_one")
    assert len(daily) == rows.signal_date.nunique()
    return daily


def mature_members(daily, cut, sessions):
    pos = int(np.searchsorted(sessions, cut))
    return daily[daily.p.ge(pos-WINDOW) & daily.p.lt(pos) &
                 daily.signal_date.lt(cut) & daily.label_end.lt(cut) & daily[TARGET].notna()].copy()


def member_digest(train):
    cols = ["signal_date", "p", "label_end", "members", "known", TARGET] + features.FIELDS
    return hashlib.sha256(train[cols].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def cut_description(daily, cut, sessions):
    train = mature_members(daily, cut, sessions)
    return dict(cut=cut, rows=len(train), fit_required=len(train) >= MIN_DAYS,
                min_signal=None if train.empty else train.signal_date.min(),
                max_signal=None if train.empty else train.signal_date.max(),
                max_label_end=None if train.empty else train.label_end.max(),
                member_sha256=member_digest(train))


class ForwardEmotion:
    def __init__(self, daily, sessions, output, allowed_cuts):
        self.daily = daily
        self.sessions = sessions
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)
        self.allowed = {x["cut"]: x for x in allowed_cuts}
        self.memory = {}

    def model_at(self, cut):
        if cut not in self.allowed:
            raise ValueError("Unregistered macro cut")
        desc = cut_description(self.daily, cut, self.sessions)
        assert desc == self.allowed[cut], "Macro members changed after freeze"
        if cut in self.memory:
            return self.memory[cut]
        out = self.output / cut
        if out.exists():
            complete = out / "COMPLETED.json"
            cold = out / "COLD_START_NO_FIT.json"
            if complete.exists():
                receipt = json.loads(complete.read_text("utf-8"))
                assert all(receipt[k] == v for k, v in desc.items())
                assert base.sha(out / "ridge.joblib") == receipt["model_sha256"]
                model = joblib.load(out / "ridge.joblib")
            elif cold.exists():
                receipt = json.loads(cold.read_text("utf-8"))
                assert all(receipt[k] == v for k, v in desc.items()) and not desc["fit_required"]
                model = None
            else:
                raise FileExistsError("Unfinished macro cut; preserve attempt, do not restart")
            self.memory[cut] = model
            return model
        out.mkdir()
        train = mature_members(self.daily, cut, self.sessions)
        base.write_json(out / "BEFORE_FIT_OR_COLD_START.json", dict(at=base.now(), **desc,
                          fields=features.FIELDS, target=TARGET, alpha=1., imputer="median_keep_empty",
                          scale="StandardScaler", output_clip=[0., 1.], equal_day_weights=True))
        train.to_csv(out / "TRAINING_DAYS_WITH_TARGET_CLOCKS.csv", index=False, encoding="utf-8-sig")
        if len(train) < MIN_DAYS:
            base.write_json(out / "COLD_START_NO_FIT.json", dict(at=base.now(), **desc))
            self.memory[cut] = None
            return None
        assert train.label_end.lt(cut).all() and train.signal_date.lt(cut).all()
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=1.))
        model.fit(train[features.FIELDS], train[TARGET])
        joblib.dump(model, out / "ridge.joblib")
        base.write_json(out / "COMPLETED.json", dict(at=base.now(), **desc,
                         model_sha256=base.sha(out / "ridge.joblib")))
        self.memory[cut] = model
        return model

    def predict_at(self, cut, dates):
        requested = list(dict.fromkeys(dates))
        if any(day < cut for day in requested):
            raise ValueError("Macro model must precede every prediction date")
        context = self.daily.set_index("signal_date").loc[requested]
        model = self.model_at(cut)
        estimate = np.full(len(context), np.nan) if model is None else np.clip(
            model.predict(context[features.FIELDS]), 0., 1.)
        return pd.DataFrame(dict(signal_date=requested, **{FIELD: estimate}, macro_model_cut=cut))

    def oof(self, rows):
        parts = []
        for month, part in rows.groupby(rows.signal_date.str[:7], sort=True):
            parts.append(self.predict_at(month + "-01", sorted(part.signal_date.unique())))
        return pd.concat(parts, ignore_index=True)


def attach(rows, forecasts):
    if forecasts.signal_date.duplicated().any():
        raise ValueError("Duplicate date forecast")
    result = rows.copy()
    result[FIELD] = result.signal_date.map(forecasts.set_index("signal_date")[FIELD])
    assert result[prior.KEYS].equals(rows[prior.KEYS])
    return result


def calibration_parts(train):
    dates = sorted(train.signal_date.unique())
    if len(dates) < 100:
        raise ValueError("Insufficient stock training dates")
    cut = dates[-42]
    return cut, train[train.label_end.lt(cut)].copy(), train[train.signal_date.ge(cut)].copy()


def fit_stock_head(train, score, fields, out, name, emotion):
    """Nested 42-day calibration: no calibration outcomes train its macro predictor."""
    out = Path(out)
    directory = out / "models"
    directory.mkdir(exist_ok=True)
    cut, inner, cal = calibration_parts(train)
    oof = emotion.oof(train)
    full_train = attach(train, oof)
    inner = attach(inner, oof)
    cal_predictions = emotion.predict_at(cut, sorted(cal.signal_date.unique()))
    cal = attach(cal, cal_predictions)
    boundary = score.signal_date.min()[:7] + "-01"
    deployed_predictions = emotion.predict_at(boundary, sorted(score.signal_date.unique()))
    score = attach(score, deployed_predictions)
    assert train.label_end.lt(boundary).all()
    assert inner.label_end.lt(cut).all()
    oof.to_csv(directory / f"{name}_FORWARD_OOF_INPUTS.csv", index=False)
    cal_predictions.to_csv(directory / f"{name}_NESTED_CALIBRATION_EMOTION_INPUTS.csv", index=False)
    deployed_predictions.to_csv(directory / f"{name}_FROZEN_SCORING_EMOTION_INPUTS.csv", index=False)
    inner_model = stage.fit_core(inner, fields, out, name + "_calibration_prefix")
    raw = inner_model.predict_proba(cal[fields])[:, 1]
    logit = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1-1e-6))
    calibration_start = base.now()
    base.write_json(directory / f"{name}_calibrator_BEFORE_FIT.json", dict(started_at=calibration_start,
        rows=len(cal), cutoff=cut, macro_calibration_cut=cut, C=1., max_iter=1000, random_state=42))
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(
        logit.reshape(-1, 1), cal[stage.TARGET].astype(int))
    positive = bool(calibrator.coef_[0, 0] > 0)
    if not positive:
        calibrator = None
    cal[prior.KEYS + ["stage", "label_end", stage.TARGET, FIELD]].assign(uncalibrated_oof=raw).to_csv(
        directory / f"{name}_calibration_rows.csv", index=False)
    base.write_json(directory / f"{name}_calibration_receipt.json", dict(started_at=calibration_start,
        completed_at=base.now(), name=name, cutoff=cut, inner_max_label_end=inner.label_end.max(),
        calibration_rows=len(cal), positive_slope=positive, macro_calibration_cut=cut,
        macro_calibration_max_label_end=emotion.allowed[cut]["max_label_end"],
        scoring_macro_cut=boundary, nested_label_clock_verified=True))
    model = stage.fit_core(full_train, fields, out, name + "_full")
    bundle = dict(version="JOINT_EMOTION_NESTED_V1", stage=1, model=model, calibrator=calibrator,
        features=fields, variant=name, train_prior=float(train[stage.TARGET].mean()),
        train_max_label_end=train.label_end.max(), max_signal=train.signal_date.max(),
        calibration_status="HISTORICAL_QUALIFICATION_PENDING", research_only=True)
    joblib.dump(bundle, directory / f"{name}_bundle.joblib")
    return bundle, score


def score_forecast(predictions, daily):
    estimates = predictions[["signal_date", FIELD]].drop_duplicates()
    assert not estimates.signal_date.duplicated().any()
    paired = estimates.merge(daily[["signal_date", "label_end", "members", "known", "outcome_coverage",
                                   TARGET, "naive_last_mature_promotion"]], on="signal_date", validate="one_to_one")
    used = paired[[FIELD, TARGET, "naive_last_mature_promotion"]].notna().all(axis=1)
    x = paired[used]
    if len(x):
        error = x[FIELD] - x[TARGET]
        naive = x.naive_last_mature_promotion - x[TARGET]
        mae, rmse = float(error.abs().mean()), float(np.sqrt((error**2).mean()))
        nmae, nrmse = float(naive.abs().mean()), float(np.sqrt((naive**2).mean()))
    else:
        mae = rmse = nmae = nrmse = None
    coverage = len(x) / len(paired) if len(paired) else 0.
    return paired, dict(target=TARGET, dates=len(paired), paired_days=len(x), unknown_or_unpaired=len(paired)-len(x),
        paired_coverage=coverage, mae=mae, rmse=rmse, naive_mae=nmae, naive_rmse=nrmse,
        independent_emotion_phase_gold_label=False, calibrated_stock_probability=False,
        pass_emotion_validation=bool(len(x) and coverage >= .95 and mae < nmae and rmse < nrmse))
