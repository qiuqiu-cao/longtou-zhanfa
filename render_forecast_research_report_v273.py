"""Render the actual V271/V272 evidence without upgrading diagnostic status."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pct(value):
    return f"{value*100:.2f}%" if value is not None else "不可估计"


def main():
    relay = json.loads((ROOT / "results/v271/next_session_nested_20261002_v1/report.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / "results/v272/market_expectation_20261002_v1/report.json").read_text(encoding="utf-8"))
    lines = ["# 隔日预期优化检验报告 V273", "", "检验日期：2026-10-02。状态：DIAGNOSTIC_ONLY；目标尚未完成。",
        "", "历史按训练、验证、测试顺序切分，模型和置信度阈值只能用验证期选择。2025H2测试126日，2026YTD测试159日。历史已被项目查看，不是纯盲测或真实前向业绩。",
        "", "## 主要结果", "", "| 指标 | 2025H2 | 2026YTD | 验收结论 |", "|---|---:|---:|---|",
        f"| 短线生态改善/延续/恶化 | {pct(market['tasks']['short_ecology_direction']['2025H2']['all_predictions']['accuracy'])} | {pct(market['tasks']['short_ecology_direction']['2026YTD']['all_predictions']['accuracy'])} | 尚未稳定超过60% |",
        f"| 接力友好/修复/分歧退潮 | {pct(relay['market']['2025H2']['all_predictions']['accuracy'])} | {pct(relay['market']['2026YTD']['all_predictions']['accuracy'])} | 未达到60% |",
        f"| 连板每日首选，包含一字身份池 | {pct(relay['relay']['2025H2']['all_identity']['top1']['accuracy'])} | {pct(relay['relay']['2026YTD']['all_identity']['top1']['accuracy'])} | 历史点估计超过60%，不代表可交易胜率 |",
        f"| 连板每日首选，非一字池 | {pct(relay['relay']['2025H2']['non_one_word']['top1']['accuracy'])} | {pct(relay['relay']['2026YTD']['non_one_word']['top1']['accuracy'])} | 未达到60% |",
        f"| 沪市A股指数涨/平/跌 | {pct(market['tasks']['sh_a_return_pct']['2025H2']['all_predictions']['accuracy'])} | {pct(market['tasks']['sh_a_return_pct']['2026YTD']['all_predictions']['accuracy'])} | 未达到60% |",
        f"| 深市A股指数涨/平/跌 | {pct(market['tasks']['sz_a_return_pct']['2025H2']['all_predictions']['accuracy'])} | {pct(market['tasks']['sz_a_return_pct']['2026YTD']['all_predictions']['accuracy'])} | 未达到60% |",
        "", "三分类与二分类、身份池与非一字池、区间覆盖率与方向准确率均不能互相替代。短线方向标签是质量分变化超过±0.05，否则为延续；指数涨平跌阈值是±0.3%。这些口径是冻结研究定义，仍需与实际交易生态检验。",
        "", "## 同日期基线和区间", "", "| 测试段与池 | 成功/日期 | 首选晋级率95%区间 | 同日最高板基线 |", "|---|---:|---|---:|"]
    for fold, report in relay["relay"].items():
        for pool, name in [("all_identity", "身份池"), ("non_one_word", "非一字池")]:
            s, base = report[pool]["top1"], report[pool]["height_baseline_same_dates"]
            lines.append(f"| {fold} {name} | {s['successes']}/{s['n']} | {pct(s['wilson95'][0])}—{pct(s['wilson95'][1])} | {pct(base['accuracy'])} |")
    lines += ["", "身份池包含大量事前已经一字涨停的股票，2025H2、2026YTD被选首选约85%和79%属于当日一字代理；次日晋级不意味着能够买到。当前历史库题材全部不可用，盘中封板路径、竞价、VWAP、盘口与实际成交不完整。",
        "", "## 数值预期与超预期", "", "V272对昨日涨停中位溢价、中位晋级率、炸板率、跌停数、情绪质量分生成训练模型及验证期扩展的80%预期区间。每个指标的覆盖率、区间宽度和点预测误差全部保留。以下为中位溢价：",
        "", "| 测试期 | 区间覆盖率 | 中位预测绝对误差 | 沿用今日数值的误差基线 |", "|---|---:|---:|---:|"]
    for fold, bands in market["numeric_bands"].items():
        b = bands["yesterday_premium"]
        lines.append(f"| {fold} | {pct(b['coverage_80_band'])} | {b['median_prediction_mae']*100:.2f}个百分点 | {b['previous_value_baseline_mae']*100:.2f}个百分点 |")
    lines += ["", "区间覆盖较高说明多数实际值在区间内，不等于主情景预测正确。区间偏宽的指标仍需优化，不能宣称80%准确率。对不利方向指标（炸板率、跌停数），低于预期区间才算有利超预期。",
        "", "## 已落地的稳定性修复", "", "- 同一输入和版本执行同一判断；事实不完整保留未知，不补叙事分。",
        "- 盘后缺明日竞价为正常待确认；观察时点缺关键数据为只观察。",
        "- 事后注册为历史回放；一个日期只允许冻结一个前向主预期，防止结果后挑选版本。",
        "- 超预期、符合、不及、冲突、缺失五状态，硬否决后当日取消锁定。",
        "- 09:35判断不能按09:30成交；主候选不能自动替换。",
        "- 结算必须是独立日历的下一交易日；完整收盘池保留未晋级失败样本。",
        "- 龙头种子、换手核心、卡位、补涨和新周期升级要求不同的时点证据。",
        "- 股吧与玄学保留来源/反例，未验证时不直接进入主排序。",
        "", "测试：15项关键契约测试通过。测试证明时间、分母和状态转换约束，不证明预测准确率。",
        "", "## 尚未通过的目标", "", "市场预期稳定超过60%、非一字连板稳定超过60%、真实前向准确率、概率校准、实时数据覆盖和成交后的净收益仍未通过。下一步应优先解决市场方向在2026段的退化、非一字的身份与收益分离，并用新交易日的真实冻结预测验证。",
        "", "原始报告：`results/v271/next_session_nested_20261002_v1/report.json`、`results/v272/market_expectation_20261002_v1/report.json`；每条预测与真实标签独立保存在相邻CSV文件，便于复核。"]
    (ROOT / "隔日预期优化检验报告V273.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
