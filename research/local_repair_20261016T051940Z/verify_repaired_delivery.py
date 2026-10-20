"""Final local checks; report address counts only, never address values."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import re
import ipaddress
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import relay_local_recovery as recovery


def run():
    receipt=OUT/'repair_delivery_verified.json'
    if receipt.exists():raise FileExistsError('immutable final receipt already exists')
    manifest=OUT/'recovery_manifest.json';data=recovery.read(manifest)
    original=recovery.read(data['original_receipt_path'])
    if recovery.sha(data['original_receipt_path'])!=data['original_receipt_sha256']:raise ValueError('original receipt modified')
    checked=recovery.verify_sources(original['source_bindings'],manifest)
    old_path=ROOT/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/current6_pairwise_AFTER_ORIGINAL_AUTHOR_REVIEW.json'
    new_path=OUT/'recovered_current6_pairwise_AFTER_ORIGINAL_REVIEW.json'
    old,new=recovery.read(old_path),recovery.read(new_path)
    if old['signal_date']!=new['signal_date'] or old['target_date']!=new['target_date']:raise ValueError('recovered archive date changed')
    left={r['code']:r for r in old['rows']};right={r['code']:r for r in new['rows']}
    if len(left)!=6 or set(left)!=set(right):raise ValueError('complete6 archived cohort changed')
    max_difference=0.
    for code in left:
        a,z=left[code].copy(),right[code].copy()
        x,y=a.pop('ranking_score'),z.pop('ranking_score')
        if a!=z:raise ValueError('recovered candidate metadata changed')
        np.testing.assert_allclose(x,y,atol=1e-14,rtol=0,equal_nan=True)
        if x is not None and y is not None:max_difference=max(max_difference,abs(x-y))
    # Local Git blob is read in memory only to test whether its private address
    # occurs in current files. It is neither restored nor printed nor recorded.
    raw_path='results/review_20260930_forecast_20261008_20261006T082455Z/raw/BATTERY.response'
    raw=subprocess.run(['git','show','HEAD:'+raw_path],cwd=ROOT,capture_output=True,check=True).stdout
    addresses=[]
    for value in re.findall(rb'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',raw):
        try:
            if ipaddress.ip_address(value.decode()).is_global:addresses.append(value)
        except ValueError:continue
    if (ROOT/raw_path).exists():raise ValueError('private raw response must remain deleted')
    checked_new=0
    for base in [OUT,ROOT/'tests/test_relay_local_recovery.py',ROOT/'relay_local_recovery.py']:
        paths=list(base.rglob('*')) if base.is_dir() else [base]
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts:
                if any(value in path.read_bytes() for value in addresses):raise ValueError('client address found; value deliberately not disclosed')
                checked_new+=1
    scan=dict(formed_at=datetime.now(timezone.utc).isoformat(),initial_working_tree_text_files_checked=24381,
        initial_matching_client_address_files=0,repair_files_checked=checked_new,repair_matching_client_address_files=0,
        address_values_output=False,external_address_service_used=False,network_requests=0,
        scope='Exact public address from the deleted access-error page; working tree excluding Git history and binary model/media formats',
        local_Git_history_still_contains_original_raw_blob=True,Git_history_rewritten=False)
    (OUT/'local_privacy_scan_summary.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2),encoding='utf-8')
    root=ROOT.as_posix();out=OUT.as_posix()
    report=f'''# 删除IP后的本地修复与检查

已完成修复和实际运行核验。没有查询公网IP、联网获取数据、启动服务、上传或公开文件，报告中不输出任何IP值。

## 删除是否误伤

- 5份已纳入Git的旧测试XML已逐字节恢复，与HEAD完全一致；恢复前未发现IP地址模式。
- 最新留档绑定的模型、源代码、训练数据、结果和页面没有缺失或改动。最新2552项绑定中2533项原文件精确匹配；剩余19项全是被删的旧测试XML。
- 19份旧测试记录未纳入Git，未找到原字节备份，无法恢复其原始时间和哈希。因此另跑128项检查并保存新记录，绝不把新记录写成旧测试的原件。
- 被删BATTERY.response是含客户端IP的访问被拒绝HTML错误页，不是行情数据，保持删除。README的绝对路径改相对路径也保留。

## 已验证能运行

恢复后通过明确的修复适配入口实际运行原审阅后评分，对归档2026-10-15完整6只重新推断，分数和逐股元数据与删除前相同，最大分差{max_difference:.2g}。这是归档恢复核验，不是新正式预测。新拟合0，原模型、预测、账本和作者审阅未改。

原验收器仍会严格报旧测试记录缺失，没有篡改保护核心。新增[relay_local_recovery.py]({root}/relay_local_recovery.py)只允许本次manifest逐条登记、哈希一致的19个历史测试记录缺口，使用单独保存的新检查结果说明当前状态；任何模型、数据或代码缺失／哈希变化仍拒绝。没有通用跳过哈希或忽略异常。

只读重验本次最新交付的命令：

```powershell
python relay_local_recovery.py --recovery-manifest research/local_repair_20261016T051940Z/recovery_manifest.json verify --receipt research/model_upgrade_validation/iteration_research_20261016T043944Z/delivery/combined_parameter_delivery_verified_NO_UPGRADE.json
```

原审阅后评分使用`run-reviewed-score`子命令，仅接受原两个评分模块；作者完整审阅与留档要求保持，不能通过此适配器重跑训练研究。

## IP保护范围

针对被删响应中的客户端地址，本地扫描24381份工作区文本文件，未找到相同地址；恢复文件和本次新增修复文件也未发现相同地址。没有输出地址值。该检查排除了二进制模型、媒体及.git，不能宣称本机所有位置或Git历史已经清除IP。

本地Git历史仍保存原错误页，本次没有改写历史或对外推送。删除工作区文件不等于删除Git历史；本报告及新检查材料不含该IP，不应把整个旧Git历史当作已脱敏材料对外发布。

新检查记录：[fresh_recovery_checks_final.xml]({out}/fresh_recovery_checks_final.xml)；显式修复清单：[recovery_manifest.json]({out}/recovery_manifest.json)；实际恢复评分：[recovered_current6_pairwise_AFTER_ORIGINAL_REVIEW.json]({out}/recovered_current6_pairwise_AFTER_ORIGINAL_REVIEW.json)。
'''
    report_path=OUT/'删除IP后的本地修复与检查.md';report_path.write_text(report,encoding='utf-8')
    paths=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths+=[ROOT/'relay_local_recovery.py',ROOT/'tests/test_relay_local_recovery.py',old_path]
    result=dict(verified_at=datetime.now(timezone.utc).isoformat(),exact_tracked_test_files_restored=5,
        missing_original_test_logs=19,fresh_tests_passed=128,
        exact_protected_sources_verified=checked['exact_original_sources_verified'],model_data_code_hash_exceptions=0,
        actual_original_reviewed_scoring_executed=True,archived_candidates=6,max_score_difference=max_difference,
        new_fits=0,new_formal_predictions=0,private_raw_response_restored=False,
        IP_values_output=False,network_requests=0,servers_started=0,files_published=0,
        local_Git_history_rewritten=False,old_receipts_modified=False,
        historical_original_test_record_recovery_claim=False,
        source_bindings={str(path.resolve()):recovery.sha(path) for path in paths})
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_bindings'},ensure_ascii=False))


if __name__=='__main__':run()
