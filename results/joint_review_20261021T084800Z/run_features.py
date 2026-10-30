from pathlib import Path
import subprocess
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
t=(ROOT/'results/daily_review_20261020T101630Z/prepare_features.py').read_text(encoding='utf-8')
t=t.replace("day='2026-10-20'", "day='2026-10-21'")
t=t.replace("oldfacts=b.read(OLD/'facts/qualified_current_facts_SCORES_MASKED.json')", "oldfacts=b.read(ROOT/'results/daily_review_20261020T101630Z/facts/qualified_full_facts_SCORES_MASKED.json')")
t=t.replace("a=item.get('2026-10-19')", "a=item.get('2026-10-20')")
t=t.replace("signal_date='2026-10-19'", "signal_date='2026-10-20'")
t=t.replace("alt=OUT/'features/alternative_2026-10-20'", "alt=OUT/'features/alternative_2026-10-21'")
insertion="""
    p20=ROOT/'results/daily_review_20261020T101630Z/features/alternative_2026-10-20'
    m20=b.read(p20/'manifest.json');pools20={tuple(k.split('/')):v for k,v in b.read(p20/'normalized.json')['pools'].items()}
    d20=b.read(ROOT/'results/daily_review_20261020T101630Z/facts/normalized_daily.json')
    extra[str(p20.resolve())]=(m20,pools20,d20,[])
"""
t=t.replace("    adapter=latest.prior_adapter()",insertion+"    adapter=latest.prior_adapter()")
t=t.replace("    def reader(path):", "    history += [str(p20.resolve())]\n    def reader(path):")
t=t.replace("('2026-10-20','2026-10-21',13)", "('2026-10-21','2026-10-22',16)")
t=t.replace('full_relay=13','full_relay=16').replace("'full_relay':13","'full_relay':16")
(OUT/'prepare_features.py').write_text(t,encoding='utf-8')
subprocess.run([r'D:\python\python.exe',str(OUT/'prepare_features.py')],check=True,cwd=ROOT)
