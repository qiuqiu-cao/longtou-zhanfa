from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
old=ROOT/'results/daily_review_20261020T101630Z/collect_today.py'
text=old.read_text(encoding='utf-8').replace("DAY='2026-10-20'", "DAY='2026-10-21'")
text=text.replace("oldpage=json.loads((OLD/'facts/2026-10-19_complete_page_pools.json').read_text(encoding='utf-8'))", "oldpage=json.loads((ROOT/'results/daily_review_20261020T101630Z/facts/2026-10-20_complete_page_pools.json').read_text(encoding='utf-8'))")
text=text.replace("codes|={'600825','002696','002242'}", "codes|={'002086','600825','002696','002242','002724','600178','603278','002929'}")
target=OUT/'collect_today.py'
target.write_text(text,encoding='utf-8')
subprocess.run([r'D:\python\python.exe',str(target)],check=True,cwd=ROOT)
