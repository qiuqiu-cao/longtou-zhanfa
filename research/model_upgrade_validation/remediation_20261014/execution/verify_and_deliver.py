"""Save the actual full-pool execution coverage and focused checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forum_execution_review as execution


def new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


facts = ROOT / "research/model_upgrade_validation/forum_reasoning_20261014/actual_full15/facts_with_forum_reasoning.json"
review = ROOT / "research/model_upgrade_validation/forum_reasoning_20261014/actual_full15/author_argument_completed.json"
predictions = ROOT / "results/daily_review_20261014T1118Z/15日完整15只条件判断.json"
calendar = ROOT / "data/v298/exchange_calendar_extension_20261003_v2/calendar.json"
old = execution.read(OUT / "actual_full15_pending/execution_review.json")
current = (ROOT / "forum_execution_review.py").read_text(encoding="utf-8")
before = current.replace('''            if (cover.get("execution_records_complete") is not True
                    or not stamp(facts["target_date"] + "T15:00:00+08:00") <= stamp(cover["observed_at"]) <= stamp(cover["available_at"]) <= cutoff):''',
'''            if cover.get("execution_records_complete") is not True or stamp(cover["available_at"]) > cutoff:''')
expected = old["source_bindings"][str(ROOT / "forum_execution_review.py")]
assert hashlib.sha256(before.encode("utf-8")).hexdigest() == expected
with (OUT / "forum_execution_review.actual_pending_v1_source.py").open("x", encoding="utf-8", newline="") as handle:
    handle.write(before)
new(OUT / "actual_pending_v1_source_compatibility.json", dict(
    old_report=str(OUT / "actual_full15_pending/execution_review.json"), old_report_sha256=execution.sha(OUT / "actual_full15_pending/execution_review.json"),
    bound_old_source_sha256=expected, exact_preimage=str(OUT / "forum_execution_review.actual_pending_v1_source.py"),
    exact_preimage_sha256=execution.sha(OUT / "forum_execution_review.actual_pending_v1_source.py"),
    change="Only complete cash-day declaration must be observed after target-session close. No current execution input was supplied; the full15 pending values are unchanged.",
    original_report_changed=False, fits=0))
asof = datetime.now(timezone.utc).isoformat()
report = execution.run(facts, review, predictions, calendar, OUT / "actual_full15_final", asof=asof)
tests = ["tests.test_forum_execution_review", "tests.test_execution_funnel_audit", "tests.test_trade_account_ledger", "tests.test_broker_execution_import_v16", "tests.test_broker_exit_import_v16"]
test_run = subprocess.run([sys.executable, "-m", "unittest", *tests, "-v"], cwd=ROOT, text=True, capture_output=True)
with (OUT / "focused_tests.log").open("x", encoding="utf-8") as handle:
    handle.write(test_run.stdout + test_run.stderr)
assert test_run.returncode == 0
assert report["pool_size"] == 15 and report["actual_execution_records"] == 0
assert all(row["opening_confirmation_status"] == "UNKNOWN" and row["actual_complete_trade_net_return"] is None for row in report["rows"])
assert {r["code"] for r in report["rows"]} == {r["code"] for r in execution.read(predictions)["rows"]}
assert report["funnel"]["cash_day_rate"] is None and report["supplied_account_projection"] is None
new(OUT / "execution_coverage.json", dict(formed_at=asof, signal_date=report["signal_date"], target_date=report["target_date"],
    full_pool=15, original_direction_records=15, original_textual_condition_records=15, current_forum_direction_records=0,
    confirmed_opening_records=0, actual_buy_fill_records=0, actual_exit_fill_records=0, actual_fee_records=0,
    known_actual_complete_net_returns=0, known_initial_cash_records=0, verified_cash_days=0,
    unknown_complete_trade_returns=15, no_fill_is_not_a_zero_return=True, numerical_fits=0,
    source_bindings={str(p):execution.sha(p) for p in (facts,review,predictions,calendar,ROOT/"forum_execution_review.py")}))
new(OUT / "completion_receipt.json", dict(status="PASS", formed_at=asof, focused_checks_exit_code=test_run.returncode,
    focused_test_count=41, actual_full_pool=15, actual_fills=0, new_fits=0, original_predictions_changed=False,
    actual_profits_verified=False, forecasting_accuracy_improved=False,
    source_bindings={str(p):execution.sha(p) for p in (OUT/"focused_tests.log",OUT/"execution_coverage.json",OUT/"actual_full15_final/execution_review.json",ROOT/"forum_execution_review.py",ROOT/"tests/test_forum_execution_review.py")}))
for src in (ROOT/"forum_execution_review.py",ROOT/"tests/test_forum_execution_review.py"):
    with (OUT/(src.stem + ".release_source.py")).open("xb") as handle:
        handle.write(src.read_bytes())
doc = """# 执行和收益闭环补强

本次解决了新股吧审阅入口与已有成交/退出/现金账本脱节的问题。新增 `forum_execution_review.py`；旧模块、预测、注册和账本未修改。

实际核对 10 月 14 日信号、15 日目标的完整 15 只，保留原独立方向和原条件计划。本次论坛分析方向仍为空，不能把背景分析填成新预测。15 只开盘确认、实际成交、退出、费用、实际完整收益全部保持 UNKNOWN；没有成交不等于赚 0%，未见成交记录也不能登记为空仓日。

## 可调用接口

`run(facts_path, review_path, original_predictions_path, calendar_path, output_dir, asof=..., evidence_path=None)` 先用原论坛入口核验精确事实/作者审阅，再绑定原完整预测及官方交易日。输入必须覆盖相同股票和日期；执行输入还必须绑定这四份文件的 SHA-256。旧来源失败保留，不重新采集行情。

执行证据 JSON 使用 `plans / confirmations / orders / fills / exit_confirmations / marks / benchmarks` 列表，字段沿用现有 entry_plan_audit、execution_funnel_audit、exit_plan_audit、trade_account_ledger。每条必须有保存原声明的 `evidence_ref={path,sha256,json_pointer}`，且与原 JSON 行完全相同。CSV 券商导出可先沿现有 v16 normalize_orders/normalize_fills 规范化，原 CSV 与声明银行一起保存；本接口不会把来源真伪认证成已核券商接口。

计划、实际确认、实际委托、成交顺序分别核验；确认必须在委托前实际可用，不能看完同一开盘价又声称按该开盘价成交。成交必须在计划和确认窗口内、官方交易日及交易时段内；买入目标日与原预测目标一致，卖出遵守 T+1。

已成交但未退出、停牌/跌停/无卖出流动性受阻、超出持有期和退出规则不符均保留。实际退出与费用完整时计算完整成交扣费收益，退出规则不符仍保留实际交易事实，但不给策略合格标签。没有费用则净收益未知；实际成交价格已包含实际价格影响，报告滑点另列且不二次扣除。

账户初始现金必须另有来源，且不存在未导入的期初持仓；现金和未平仓收盘盯市复用原 trade_account_ledger。缺持仓报价时权益/回撤未知，不把未卖仓位删除。这个账户是已供本次审阅成交的隔离投影，不能冒称完整券商账户。

日线开盘/收盘代理不能进入实际收益接口，原代理研究、原失败报告保留，需另列诊断页。本次没有重复历史代理拟合，也没有给当前 15 只合成收益。

## 验证和交付

新增 17 项针对核验，联同既有券商导入、执行漏斗及现金账本共 41 项通过。覆盖源哈希和原行篡改、跨池、同一开盘观察后成交、时间窗、非交易日/午间、T+1、扣费现金 PnL、缺费、未平仓盯市缺失、停牌受阻、纸面/实际分栏与日线代理拒绝。它们证明代码边界，不证明预测准确率或盈利能力提高。

实际文件 `actual_full15_final/execution_review.json` 和 `actual_full15_final/完整池成交与收益待结算.md`；覆盖收据 `execution_coverage.json`，完整核验日志 `focused_tests.log`，结束收据 `completion_receipt.json`。最初 pending 结果及其精确源码前像另存，当前源码不能冒称最初运行源码。
"""
with (OUT / "执行收益闭环交付.md").open("x", encoding="utf-8") as handle:
    handle.write(doc)
print(json.dumps({"status":"PASS","pool":15,"fills":0,"tests":41,"fits":0},ensure_ascii=True))
