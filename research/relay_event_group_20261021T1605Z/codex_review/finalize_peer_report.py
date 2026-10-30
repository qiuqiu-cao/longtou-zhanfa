from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'validation_final'
report=OUT/'EVENT_COHORT_VALIDATION_REPORT.md'
original=OUT/'EVENT_COHORT_VALIDATION_REPORT_GEMINI_ORIGINAL.md'
assert report.exists() and not original.exists()
shutil.copyfile(report,original)
verified=(OUT/'CODEX_VERIFIED_RESULT.md').read_text(encoding='utf-8')
text=verified.replace('# 连板事件固定群体：Codex核验结果','# 连板事件固定群体与个股地位验证报告（Codex复核定稿）',1)
text+='\nGemini原稿完整保留在EVENT_COHORT_VALIDATION_REPORT_GEMINI_ORIGINAL.md。原稿分段计数、旧实验引用、覆盖率及未经验证的归因已纠正，以本定稿和原RESULT.json为准。\n'
report.write_text(text,encoding='utf-8')
corrections=dict(at_utc=datetime.now(timezone.utc).isoformat(),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    corrected_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),new_fits=0,
    corrections=[
        '2026JanAug control/new is67/69, not68/70; Sep8/8, not7/7.',
        'F1 H2 control/new .31847/.31310; JanAug .33640/.34993; Sep .48780/.45570 andRecall .50/.45.',
        'Prior five-path top2slots246 top3slots351 top2any194 top3any221, CI[-.03642,.01656]; original report inserted different values.',
        'Final matched1118/9245=12.09%, not range1118-1997 or21.6%; source-unknown3613 is not all candidate identity unknown.',
        'Only one prior fold hadTop1decline. New+3does not prove localgroup causality,10-day optimum,theme identity,orautomaticfallback tooldmodel.',
        'Cross-board pooled Top1gain is notsum ofwithin-board separate-top1changes.'
    ])
with (OUT/'REPORT_CORRECTIONS.json').open('x',encoding='utf-8') as f:json.dump(corrections,f,ensure_ascii=False,indent=2)
print(json.dumps({'original_preserved':True,'report_corrected':True,'new_fits':0}))
