from pathlib import Path
import json,hashlib,subprocess
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
peer=[OUT/'gemini/INDEPENDENT_ROUND1.md',OUT/'gemini/INDEPENDENT_ROUND1.json']
assert all(p.exists() for p in peer)
(OUT/'BEFORE_BASE_BOTH_ROUND1_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in peer+[OUT/'codex/INDEPENDENT_ROUND1_BEFORE_PEER_AND_MODEL.json']},indent=2,ensure_ascii=False),encoding='utf-8')
t=(ROOT/'results/daily_review_20261020T101630Z/score_original_BASE.py').read_text(encoding='utf-8')
t=t.replace("own=OUT/'independent_full46_BEFORE_MODEL.json';receipt=b.read(OUT/'independent_freeze_receipt_BEFORE_MODEL.json')", "own=OUT/'codex/INDEPENDENT_ROUND1_BEFORE_PEER_AND_MODEL.json';receipt=b.read(OUT/'codex/INDEPENDENT_FREEZE_RECEIPT.json')")
t=t.replace("receipt['full_rows']==46", "receipt['full_rows']==67").replace("len(pack['relay'])==13", "len(pack['relay'])==16")
t=t.replace("signal_date='2026-10-20',target_date='2026-10-21'", "signal_date='2026-10-21',target_date='2026-10-22'")
t=t.replace('complete13=True,independent_full46_unchanged=True','complete16=True,independent_full67_unchanged=True')
(OUT/'score_original_BASE.py').write_text(t,encoding='utf-8')
subprocess.run([r'D:\python\python.exe',str(OUT/'score_original_BASE.py')],check=True,cwd=ROOT)
