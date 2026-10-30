"""Verify three saved estimators reproduce all pre-evaluation predictions; no fit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'validation_final'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(OUT/'FROZEN_PROTOCOL.json'); ack=read(OUT/'GEMINI_ACK_BEFORE_FIT.json')
assert ack['accepted'] is True and ack['actual_executor']=='Antigravity Gemini'
assert ack['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.json')
for path,digest in p['sources'].items(): assert sha(Path(path))==digest,path
spec=importlib.util.spec_from_file_location('qualified_fixed_runner_audit',HERE/'fixed_event_validation.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
full=r.prepare(Path(p['feature_file']))
models=[]
for i,check in enumerate(p['checks'],1):
    fold=check['fold']; tag=fold['id']
    started=read(OUT/(tag+'_FIT_STARTED.json'))
    completed=read(OUT/(tag+'_FIT_COMPLETED.json'))
    assert pd.Timestamp(started['at_utc'])>=pd.Timestamp(ack['at_utc'])
    assert started['attempt']==1 and completed['fits_attempted']==i
    path=OUT/(tag+'_MODEL.joblib'); saved=joblib.load(path)
    assert sha(path)==completed['model_sha256']
    assert saved['features']==p['fields'] and len(saved['features'])==89
    assert saved['cutoff']==fold['start'] and saved['params']==p['estimator']
    assert saved['training_digest']==check['training_digest']
    estimator=saved['model'].named_steps['hgb']
    assert estimator.n_iter_==120
    for k,v in p['estimator'].items(): assert estimator.get_params()[k]==v
    test=full.loc[full.date.between(fold['start'],fold['end'])]
    with threadpool_limits(limits=2): score=saved['model'].predict_proba(test[p['fields']])[:,1]
    predicted=test[['date','code']].copy();predicted['score_loaded']=score
    disk=pd.read_csv(OUT/(tag+'_PREDICTIONS_BEFORE_EVALUATION.csv'),dtype={'code':str})
    paired=disk.merge(predicted,on=['date','code'],validate='one_to_one')
    assert len(paired)==check['test_rows']
    error=float(np.abs(paired.score-paired.score_loaded).max())
    assert error<1e-12,(tag,error)
    models.append(dict(fold=tag,model_sha256=sha(path),rows=len(paired),max_saved_load_score_error=error,
                       fit_started=started['at_utc'],fit_completed=completed['at_utc']))
result=read(OUT/'RESULT.json')
assert result['actual_fits_attempted']==3 and result['rows']==3707 and result['dates']==302 and result['unknown']==67
receipt=dict(at_utc=datetime.now(timezone.utc).isoformat(),auditor_fits=0,actual_fits=3,
             protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'),models=models,all_bound_sources_unchanged=True)
with (OUT/'CODEX_SAVED_MODEL_AUDIT.json').open('x',encoding='utf-8') as handle:
    json.dump(receipt,handle,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=True))
