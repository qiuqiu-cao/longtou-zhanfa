"""Read saved choices and outcomes only. Never train or alter predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
JOINT = OUT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize(frame, dates):
    rows = []
    for day in dates:
        g = frame[frame.date.eq(day)].sort_values(['raw_score', 'code'], ascending=[False, True])
        pos = g.target.eq(1)
        top = g.iloc[0] if len(g) else None
        rows.append(dict(date=day, candidates=len(g), known=int(g.target.notna().sum()),
            unknown=int(g.target.isna().sum()), positives=int(pos.sum()),
            top1_code=None if top is None else top.code,
            top1_boards=None if top is None else int(top.boards),
            top1_target=None if top is None or pd.isna(top.target) else int(top.target),
            first_positive_rank=int(np.flatnonzero(pos)[0]+1) if pos.any() else None,
            top1_hit=bool(pos.iloc[0]) if len(g) else False,
            top2_any_hit=bool(pos.iloc[:2].any()), top3_any_hit=bool(pos.iloc[:3].any())))
    daily = pd.DataFrame(rows)
    return dict(dates=len(daily), candidates=len(frame),
        positive_exists_days=int(daily.positives.gt(0).sum()),
        no_positive_all_known_days=int((daily.positives.eq(0) & daily.unknown.eq(0)).sum()),
        no_observed_positive_with_unknown_days=int((daily.positives.eq(0) & daily.unknown.gt(0)).sum()),
        top1_hits=int(daily.top1_hit.sum()),
        top1_unknown=int(daily.top1_target.isna().sum()),
        top1_known_failure_but_positive_elsewhere_days=int((daily.top1_target.eq(0)&daily.positives.gt(0)).sum()),
        top1_unknown_but_positive_elsewhere_days=int((daily.top1_target.isna()&daily.positives.gt(0)).sum()),
        top2_any_hit_days=int(daily.top2_any_hit.sum()),
        top3_any_hit_days=int(daily.top3_any_hit.sum()),
        first_positive_rank_counts=daily.first_positive_rank.fillna('NONE_OBSERVED').value_counts().to_dict(),
        note='Outcome-conditioned diagnostic; presence of a winner is not an achievable model ceiling or a prospective selection rule.'), daily


def main():
    source=ROOT/'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/historical_final/dataset.joblib'
    pred_path=ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/relay_history/predictions.csv'
    data=joblib.load(source)
    feats=data['features']
    pred=pd.read_csv(pred_path, dtype={'code':str}, low_memory=False)
    pred=pred[pred.arm.eq('BASE')]
    frame=pred[['date','code','raw_score','target']].merge(
        data['prepared']['relay'][['date','code','target','boards']],
        on=['date','code'],validate='one_to_one',suffixes=('_saved',''))
    assert len(frame)==3707 and frame.date.nunique()==302
    assert frame.target_saved.fillna(-1).equals(frame.target.fillna(-1))
    frame=frame.merge(feats[['date','code','daily_one_word']],on=['date','code'],validate='one_to_one')
    dates=sorted(frame.date.unique())
    results={}
    for name,g in [('full',frame),('nonword',frame[frame.daily_one_word.eq(0)])]:
        results[name],daily=summarize(g,dates)
        daily.to_csv(OUT/(name+'_all302_day_dispositions.csv'),index=False,encoding='utf-8-sig')
    v2=JOINT/'p1_correction_v2'
    module_path=v2/'run_p1_correction_v2.py'
    spec=importlib.util.spec_from_file_location('gemini_v2_readonly',module_path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    saved=pd.read_csv(v2/'p1_features_9245_v2.csv',dtype={'code':str,'anchor_break_age_sessions':str})
    rebuilt=mod.build_p1_features(data['sessions'],data['attempts'],data['bars'],feats)
    rebuilt=rebuilt.sort_values(['date','code']).reset_index(drop=True)
    original=saved.sort_values(['date','code']).reset_index(drop=True)
    mismatches={}
    for col in rebuilt.columns:
        a,b=rebuilt[col],original[col]
        if col=='anchor_break_today':
            a,b=a.astype(str),b.astype(str)
        unequal=~(a.eq(b)|(a.isna()&b.isna()))
        if unequal.any(): mismatches[col]=int(unequal.sum())
    assert not mismatches, mismatches
    joined=rebuilt.merge(data['prepared']['relay'][['date','code','target']],on=['date','code'],validate='one_to_one')
    joined=joined.merge(feats[['date','code','daily_one_word']],on=['date','code'],validate='one_to_one')
    tables=json.loads((v2/'P1_CONDITION_TABLES_V2.json').read_text('utf-8'))
    independent_gap=mod.day_gap(joined,nonword=True)
    assert independent_gap==tables['amount_rank_delta_within_day_paired']['non_one_word_board_2']
    for gkeys,filename in [(['anchor_break_age_sessions','boards','daily_one_word'],'age_board_price_shape_complete_cells.csv')]:
        counts=[]
        for keys,g in joined.groupby(gkeys,dropna=False):
            counts.append(dict(zip(gkeys,keys))|dict(N=len(g),known=int(g.target.notna().sum()),unknown=int(g.target.isna().sum()),hits=int(g.target.eq(1).sum())))
        pd.DataFrame(counts).to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=0,http_requests=0,
        sources={str(p.relative_to(ROOT)):sha(p) for p in [source,pred_path,module_path,v2/'p1_features_9245_v2.csv',v2/'P1_CONDITION_TABLES_V2.json']},
        saved_baseline_choice_diagnostic=results,
        v2_reproduction=dict(rows=len(rebuilt),mismatches=mismatches,nonword_board2_day_gap=independent_gap),
        limitations=['Saved historical results already seen; not fresh blind evidence.',
            'Current-universe survivor bias and original source limitations persist.',
            'No trade execution or realized profit win rate measured.',
            'V2 middle nonpromotion is not severe damage; severe-damage path and complete bounds still need peer clarification.',
            'Prior pairwise ranking models failed; no authorization to rerun them under a new name.'])
    (OUT/'DIAGNOSTIC_AND_V2_REPRODUCTION.json').write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(plain(dict(choices=results,v2_rows=len(rebuilt),v2_mismatches=mismatches)),ensure_ascii=False))


if __name__=='__main__':
    main()
