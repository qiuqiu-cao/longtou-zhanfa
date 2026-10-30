"""Audit longer price history without reading promotion outcomes or fitting models."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_long_high_features as features
import first_board_joint_seal_model as prior

out=Path(__file__).resolve().parent
protocol=dict(at=prior.base.now(),previous_goal_turn="PROGRESS:twojointreopenarmsactuallytrainedandindependentlysettled,bothrejected,original60targetactive",
    budget_started="2026-10-21T02:24:49+00:00",deadline="2026-10-21T02:54:49+00:00",fits=0,outcomes_read=False,
    hypothesis="Longer previously unused120-session prior-high position andlatest-high age addinformation beyond original20/60-day context",
    sources={str(p):prior.base.sha(p) for p in prior.base.INPUTS.values()},
    source_start_observation="Rawdatesstart2023-11-28;265devsignaldatepool;250hasnoexpectedinnerprefixsupportbefore2024-11-01. Auditcoveragewithouttraining250.",
    features=features.FIELDS,quarantined="250-session distance diagnostic only;not anarm. No rawunadjusteddistance,reopen/sealfeatures,labels,futureprices,currentthemes or actoridentity in eithernewhead.",
    formula="a(t)=log(reportedpreclose(t))-log(close(t-1))onlyadjacentvalidbars;L=cumsum(a);zHigh=log(high)-L;distance=log(closeS)-L(S)-max(zHigh[S-120:S]);latestargmaxageusing1..120officialpositions,currentS-high excluded",
    source_admission="EntireS-120..S validtradedcentOHLC,positiveamount/volume,10%regularbounds andexacth consecutiveofficialpositions;missinginvalidsuspendedwindowNaN. Missingneverzero. ExactsourceS-centclosematch withoriginalfull28503pool.",
    before_outcome_qualification_gate="120fieldnonmissing>=1000rows and>=60distinctsignaldates inoriginalbaselineinnerprefix;>=300rows and>=30dates incalibration42;validation>=70%fieldcoverage and>=95%dateshaveatleastoneknownfield;nonconstantinnerfield;bothprefixchecksidentical. Theseareinputqualification,notperformanceacceptance.",
    source_interpretation="Archivedreportedreferencechainproxy,notindependentlycertifiedofficialadjustmentfactor,corporateactiontruth,shareholdercostbasisorfundcause. Rawpricejumpalonenotmarketreturn. Oldsurvivorshipandlateacquisitionremain.",
    proposed_arms="Gemini original62+distance120(63);Codex original62+distance120+latesthighage120(64). Actualpeerreviewandcommonfreezearemandatorybeforefits.")
with (out/'PROTOCOL_BEFORE_FEATURE_AUDIT.json').open('x',encoding='utf-8') as file:json.dump(protocol,file,ensure_ascii=False,indent=2)
facts=pd.read_csv(prior.FACTS,usecols=features.KEYS+['stage','close'],dtype={'code':str},float_precision='round_trip')
facts=facts[facts.stage.eq(1)].drop(columns='stage').reset_index(drop=True)
assert len(facts)==28503
bars=features.load_source()
print('source_rows='+str(len(bars)),flush=True)
all_features=features.from_bars(bars)
joined=features.attach_full_pool(facts,all_features)
assert 'promotion_next_LABEL_ONLY' not in joined and 'label_end' not in joined
checks=[]
for cutoff in ('2025-06-30','2025-12-31'):
    prefix=features.from_bars(bars[bars.date.le(cutoff)])
    expected=all_features[all_features.signal_date.le(cutoff)]
    assert prefix[features.KEYS].equals(expected[features.KEYS].reset_index(drop=True))
    np.testing.assert_allclose(prefix[features.FIELDS+[features.DIAGNOSTIC250]],expected[features.FIELDS+[features.DIAGNOSTIC250]],rtol=0,atol=0,equal_nan=True)
    checks.append(dict(cutoff=cutoff,rows=len(prefix),allfields_equal=True))
    print(cutoff+' prefix_pass',flush=True)
cutoff=json.loads((prior.REFERENCE/'models/validation_S1_STAGE_CYCLE_GENE_calibration_receipt.json').read_text('utf-8'))['cutoff']
sessions=json.loads(prior.base.INPUTS['calendar'].read_text('utf-8'))['sessions']
successor={day:sessions[i+1] for i,day in enumerate(sessions[:-1])}
next_day=joined.signal_date.map(successor)
masks={'development':joined.signal_date.lt('2025-01-01'),
       'inner_prefix':joined.signal_date.lt(cutoff)&next_day.lt(cutoff),
       'calibration42':joined.signal_date.ge(cutoff)&next_day.lt('2025-01-01'),
       'validation2025H1':joined.signal_date.between('2025-01-01','2025-06-30'),
       'evaluation':joined.signal_date.ge('2025-07-01')}
periods={}
for name,mask in masks.items():
    part=joined[mask]
    info=dict(rows=len(part),dates=part.signal_date.nunique(),fields={})
    for field in features.FIELDS+[features.DIAGNOSTIC250]:
        known=part[field].notna()
        info['fields'][field]=dict(known=int(known.sum()),coverage=float(known.mean()),observed_dates=part.loc[known,'signal_date'].nunique(),unique=int(part[field].nunique()))
    periods[name]=info
qualified={}
for field in features.FIELDS:
    inner=periods['inner_prefix']['fields'][field];cal=periods['calibration42']['fields'][field];val=periods['validation2025H1']['fields'][field]
    qualified[field]=bool(inner['known']>=1000 and inner['observed_dates']>=60 and inner['unique']>1 and cal['known']>=300 and cal['observed_dates']>=30 and val['coverage']>=.7 and val['observed_dates']/117>=.95)
chain_known=joined[features.DISTANCE].notna()
reference_changes=joined.reported_reference_changes120.gt(0)&chain_known
raw=joined.raw_unadjusted_distance120_DIAGNOSTIC_ONLY
changed_distance=(joined[features.DISTANCE]-raw).abs().gt(1e-10)&chain_known
comparison=dict(known120=int(chain_known.sum()),knownwithreportedreferencechanges=int(reference_changes.sum()),pricebasis_distance_changes=int(changed_distance.sum()),signchanges=int(((joined[features.DISTANCE].ge(0)!=raw.ge(0))&chain_known).sum()))
joined.to_csv(out/'COMPLETE28503_LONG_HIGH_FEATURES_NO_TARGETS.csv',index=False,encoding='utf-8-sig')
report=dict(at=prior.base.now(),qualified_fields=qualified,periods=periods,prefix=checks,price_basis_comparison=comparison,
    diagnostic250_quarantined_for_inner_support=True,full_rows=len(joined),source_rows=len(bars),fits=0,outcomes_read=False,source_acquired_later=True,survivorship_unchanged=True,
    feature_source_sha256=prior.base.sha(features.__file__))
prior.base.write_json(out/'qualification.json',report)
print(json.dumps(prior.base.clean(report),ensure_ascii=True),flush=True)
