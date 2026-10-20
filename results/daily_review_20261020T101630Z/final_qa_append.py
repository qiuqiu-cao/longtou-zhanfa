from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
P=Path(__file__).resolve().parent
report=P/'10月20日复盘与21日预期预测.md'
own=json.loads((P/'independent_full46_BEFORE_MODEL.json').read_text(encoding='utf-8'))
base=json.loads((P/'BASE_scores_AFTER_INDEPENDENT.json').read_text(encoding='utf-8'))
assert datetime.fromisoformat(own['formed_at']) < datetime.fromisoformat(base['formed_at'])
before=hashlib.sha256(report.read_bytes()).hexdigest()
with report.open('a',encoding='utf-8') as f:
    f.write('\n\n资料范围补正（追加，不改预测）：原独立稿及采集说明中的“限定四作者”表述过宽。本轮实际定向检索了延边刺客、毛竹下的橙子、林曜01，未单独检索凯域竟知；四名是项目优先作者清单，不是四人都已检索或阅读。实际读取的相关19日正文仍为另两位作者的两篇，20日合格正文仍为0。\n')
with (P/'FINAL_QA_APPEND_ONLY.json').open('x',encoding='utf-8') as f:
    json.dump(dict(formed_at=datetime.now(timezone.utc).isoformat(),report_before_append_sha256=before,
        report_final_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        correction='Three preferred authors searched, fourth not searched. Actual background bodies2; current20 qualified bodies0. No prediction, model or source file changed.',
        prediction_clock_before_model_verified=True,initial_shell_QA_failed_before_any_mutation=True),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(report=str(report),full_candidates=len(own['rows']),top3=[r['name'] for r in sorted([r for r in own['rows'] if r['priority_rank']],key=lambda r:r['priority_rank'])[:3]],clocks_verified=True,old_models_changed=False),ensure_ascii=False))
