from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
text=(ROOT/'results/daily_review_20261020T101630Z/qualify_today.py').read_text(encoding='utf-8')
text=text.replace('2026-10-20','2026-10-21')
text=text.split('    # Prior forecasts settle')[0]+'\n\nif __name__==\'__main__\':main()\n'
(OUT/'qualify_today.py').write_text(text,encoding='utf-8')
subprocess.run([r'D:\python\python.exe',str(OUT/'qualify_today.py')],check=True,cwd=ROOT)
