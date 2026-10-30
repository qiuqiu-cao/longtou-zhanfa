"""Record the actually read excerpt as a later author assertion, never a signal."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from evidence_contract import check_claims

BASE=Path(__file__).resolve().parent

def main():
    excerpt='“羊”“阳”字辈：九阳股份（12天7板）、百洋股份（8天6板）、海洋王（3板）'
    path=BASE/'forum_excerpt_20261020.txt'
    if path.exists() or (BASE/'FORUM_CLAIM_TIME_AUDIT.json').exists():
        raise FileExistsError('existing evidence cannot be overwritten')
    # The exact text was inspected from the linked page on this run; it is not a full post.
    now=datetime.now(timezone.utc).isoformat()
    path.write_text(excerpt,encoding='utf-8')
    source={'url':'https://m.tgb.cn/a/2w3lFFtjK6g?type=hot','source_description':'风一样的胖刺猬 10月20日主帖，仅读取正文相关位置；评论另计',
            'published_at':'2026-10-20T20:10:00+08:00','obtained_at':now,
            'saved_text_path':str(path),'saved_text_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    claims=[]
    for code,name in [('002242','九阳股份'),('002696','百洋股份'),('002724','海洋王')]:
        claims.append({'claim_id':code,'code':code,'name':name,'theme_id':'author_name_narrative_20261020',
          'theme_type':'name_narrative','role':'unknown','phase':'unknown','relation':'unknown',
          'evidence_text':'作者将三股列为字辈，但没有证明首板时已被市场识别或资金因果',
          'counterevidence':'同音不自动关联；存在其他题材、共同催化和竞争解释；缺失败成员与分钟路径',
          'reviewer':'Codex','reviewed_at':now,'source_id':'post','source_span':excerpt})
    outputs={}
    for cutoff in ('2026-10-09T15:00:00+08:00','2026-10-15T15:00:00+08:00','2026-10-16T15:00:00+08:00','2026-10-21T15:00:00+08:00'):
        result=check_claims({'cutoff':cutoff,'formed_at':now,'sources':{'post':source},'claims':claims})
        outputs[cutoff]=result
    (BASE/'FORUM_CLAIM_TIME_AUDIT.json').write_text(json.dumps({'source':source,'reviewed_at':now,'by_cutoff':outputs},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'claims':len(claims),'cutoffs':len(outputs),'actual_archived_eligible':False},ensure_ascii=False))

if __name__=='__main__': main()
