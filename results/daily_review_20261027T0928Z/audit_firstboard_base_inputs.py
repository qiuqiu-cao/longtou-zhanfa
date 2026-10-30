"""Read-only saved-model/input qualification; no model load or inference."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
day='2026-10-27';target='2026-10-28'
facts=read(OUT/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json');daily=read(OUT/'facts/normalized_daily.json')
first=[r for r in facts['rows'] if r['kind']=='limit_up' and r['boards']==1]
modelroot=ROOT/'research/first_second_stage_20261020_run1'
receiptpath=modelroot/'models/latest_S1_full_COMPLETED.json';receipt=read(receiptpath)
modelpath=modelroot/'models/latest_S1_full.joblib';bundlepath=modelroot/'first_second_stage_bundle.joblib'
assert modelpath.exists() and bundlepath.exists() and sha(modelpath)==receipt['sha256']
cal=read(ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json')['sessions'];position={d:i for i,d in enumerate(cal)}
coverage=[]
for r in first:
    days=daily.get(r['code'],{}).get('days',{})
    available=sorted(d for d in days if d<=day)
    consecutive=0
    for d in reversed([d for d in cal if d<=day]):
        if d not in days:break
        consecutive+=1
    coverage.append(dict(code=r['code'],name=r['name'],exact_current_day_available=day in days,daily_rows=len(available),first_available=available[0] if available else None,last_available=available[-1] if available else None,adjacent_official_history_rows=consecutive))
gaps=[
    dict(id='SOURCE_UNIVERSE_MISMATCH',fields=['market_positive_fraction','market_median_return'],reason='Training formulas use all eligible bars in the declared historical security universe. Today collection is the limit/failed/down event union plus selected prior cohorts; it is not a whole-universe daily cross-section. Event-only counts cannot recreate whole-market return breadth.'),
    dict(id='HISTORY_LENGTH_AND_SCOPE',fields=['return_60','close_ma60','position_60','gene_max_streak_60','gene_limits_60','gene_relay_rate_60','listing_age_sessions'],reason='Current bound daily responses begin 2026-09-18 and do not supply 60 adjacent official-session history or the original historical security status universe. Older bars exist only through the saved rebuild cutoff; missing intervening sessions cannot be silently bridged.'),
    dict(id='HISTORICAL_FIELD_MISSING',fields=['turnover_ratio_5','two_session_turnover_sum','amount_ratio_5','cycle_height_break_20'],reason='Public OHLCV day rows do not provide dated per-session turnover/amount/security-state and the same training-universe 20-session full ladder. Current quote turnover/cap cannot be propagated backwards.'),
    dict(id='EXACT_PRICE_MISSING',codes=[r['code'] for r in coverage if not r['exact_current_day_available']],reason='Entire current first-board pool remains retained; missing exact-day raw OHLC prevents verified first-board scope for affected rows.'),
    dict(id='NO_CURRENT_QUALIFIED_FEATURE_BINDING',reason='No complete current-date 62-field CSV with verified first/eligible/scope_known and the original fixed cap/risk screen is bound to this saved head. Allowing NaN at inference does not qualify an incompatible data universe.'),
]
result=dict(formed_at=datetime.now(timezone.utc).isoformat(),signal_date=day,target_date=target,status='NOT_OUTPUT_CURRENT_FIRSTBOARD_BASE_INPUT_NOT_QUALIFIED',complete_firstboard_rows=len(first),candidate_codes=[r['code'] for r in first],model_identity='Saved research 62-field STAGE_CYCLE_GENE latest_S1; separate first-to-second target',target='Exactly next official-session closing promotion to board 2',model_path=str(modelpath.resolve()),model_sha256=sha(modelpath),bundle_path=str(bundlepath.resolve()),bundle_sha256=sha(bundlepath),training_receipt_path=str(receiptpath.resolve()),training_receipt_sha256=sha(receiptpath),training_label_cutoff=receipt['max_label_end'],features=receipt['features'],feature_count=len(receipt['features']),current_input_coverage=coverage,gaps=gaps,known_scoring_entry='first_second_stage_model.py recheck --bundle <saved bundle> --features <qualified full current feature CSV> --output <new CSV>',source_files={str(p.resolve()):sha(p) for p in [ROOT/'first_board_leader_model.py',ROOT/'first_second_stage_model.py',ROOT/'first_board_capital_risk_model.py',OUT/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',OUT/'facts/normalized_daily.json']},raw_score_is_not_qualified_probability=True,new_model_loads=0,new_inferences=0,new_fits=0,production_changes=False,old_predictions_or_ledger_modified=False)
write_new(OUT/'features/FIRSTBOARD_BASE_QUALIFICATION.json',result)
print(json.dumps({k:result[k] for k in ['status','complete_firstboard_rows','feature_count','new_inferences','new_fits']},ensure_ascii=False))


