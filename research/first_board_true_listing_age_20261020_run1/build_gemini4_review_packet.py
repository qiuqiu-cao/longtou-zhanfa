"""Package this completed, rejected trial for the explicitly requested Gemini 4 audit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = RUN / "gemini4_review"
OUT.mkdir(exist_ok=True)

summary = """审核任务：用户指定 Codex 独立训练、Chrome 的 Gemini 4 只审查。本轮已完成且淘汰；不得让 Gemini 训练、调参或访问账户。用户目标仍为首板收盘时预测次一官方交易日收盘一进二，完整日 Top1 >=60%，已知选择>=200、结果覆盖>=95%、不按结果弃日，2025H2及2026各>=55%；同一股票真晋级后在二板收盘重新评估二进三。

本轮唯一假设：将原62字段估计器中的 listing_age_sessions（原归档日历左截断时钟）替换为 listing_calendar_age_days（信号日减真实IPO日的精确自然日数）；其他61字段、原训练成员、成熟标签、日期权重、HGB超参数、42日Platt校准、市值/薄换手筛选完全不变。原时钟字段在事实表中留存但不进入本估计器。此前删除时钟的61字段挑战已失败；该旧协议全文附入，审查应讨论本批独立IPO新信息与旧批禁止事后替换的边界，不能隐瞒或称原失败重跑成功。新字段无log/上限/分箱或年龄阈值优化。

真实IPO日来自既存官方当前及退市名单，逐代码与原security master唯一日期相符。来源本轮晚取得，不认证原时点HTTP取得。首次资格核查仅当前名单，第二版纳入历史退市；第三版修复将无关name/ST区间source_verified误用于稳定IPO日期的错误；两次修复均在首个fit前，旧源码和产物全留存。最终28503/28503年龄可知，开发10630、验证4724、后期13149，两个前缀检查不变，无冲突。本批没有消除原历史股票池幸存者偏差。

本轮真实结果：完整2025H1验证4724行/117日，4714已知10未知，863正例。固定市值/薄换手筛选后，挑战Top1=48/117(41.03%)，原基准52/117(44.44%)；非一字挑战43/117(36.75%)，原基准48/117(41.03%)。未筛选挑战52/117只作诊断，不可改变主口径。预设门要求主口径严格>52且非一字>=48，故直接淘汰，不进入14个月样本外、不新评二进三。全4724行分数和全部117日双范围配对选择另附CSV，unknown保留、未截获胜例。主配对共胜39/挑战独胜9/基准独胜13/共败56；非一字35/8/13/61。

真实拟合仅1逻辑头：前缀及全窗HGB各1、Platt1。独立verifier拟合0，加载保存模型复算全部4724分数、62字段、旧训练前缀/全窗成员、全部28503原字段/标签及输入/保护哈希一致；原基准和S2重训0。5项针对测试通过（最后运行已完成），正式模型、原预测、定时任务、账本和交易均未改。目标60%未达，历史已被研究过，不称新盲测；分数不能凭本试验当准确的事件概率。

审核范围限制：你收到的是附件内源码/协议/资格收据/完整验证分数及配对CSV。没有向你提供350MB原始价格、joblib模型或私有GitHub权限。你可静态审代码、在实际具备工具时独立计算CSV；请明确哪些由你直接核对、哪些是Codex本地收据声明，不得声称跑过本地训练、读取D盘或验证未提供的原始行情。原始raw report.reviewer=PENDING保留，正式审查另存附页，不回写冻结结果。

请审查：1)有无泄漏或口径/代码实质错误；2)IPO资格的退市补齐和字段级修复是否合理，以及剩余源与幸存者限制；3)完整CSV能否复核48/117、52/117和43/117、48/117；4)是否同意按协议淘汰，不能宣称哪些提升；5)如有真实bug给精确位置和必要修复，不提出重训年龄变换或新参数。没有发现不等于证明无任何泄漏。请用约1000中文字符给明确裁决和审查边界。
"""

files = [
    ROOT / "first_board_true_listing_age.py",
    ROOT / "tests/test_first_board_true_listing_age.py",
    RUN / "protocol_BEFORE_ANY_FIT.json",
    RUN / "report.json",
    RUN / "selection_BEFORE_EVALUATION.json",
    RUN / "verify_saved_result.py",
    RUN / "INDEPENDENT_SAVED_MODEL_VERIFICATION.json",
    ROOT / "research/first_board_listing_date_qualification_20261020_final/protocol_BEFORE_DATA_NO_OUTCOMES.json",
    ROOT / "research/first_board_listing_date_qualification_20261020_final/qualification.json",
    ROOT / "research/first_board_listing_date_qualification_20261020/QUALIFICATION_SCOPE_ADJUSTMENT_BEFORE_ANY_FIT.json",
    ROOT / "research/first_board_listing_date_qualification_20261020_complete/IPO_FIELD_SCOPE_CORRECTION_BEFORE_ANY_FIT.json",
    ROOT / "research/first_board_drop_clock_20261020_run1/protocol_BEFORE_ANY_FIT.json",
    ROOT / "research/first_board_drop_clock_20261020_run1/report.json",
    ROOT / "first_second_stage_model.py",
    ROOT / "first_board_capital_risk_model.py",
    ROOT / "first_board_drop_clock.py",
]
attachments = [RUN / "AUDIT_COMPLETE4724_SCORE_COMPARISON_WITH_UNKNOWN.csv", RUN / "INDEPENDENT_ALL117_DAILY_PAIRED_SELECTIONS.csv"]
manifest = dict(at=datetime.now(timezone.utc).isoformat(), destination="Gemini4 Chrome ec17e48589786054", trainer="Codex", reviewer="Gemini4 only", new_files_not_committed=True,
    preserved_remote_main="02fb21632b0d75d133426624a28bbc98a604b35b", files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files + attachments])
text = summary + "\n\nMANIFEST\n" + json.dumps(manifest, ensure_ascii=False, indent=2)
for path in files:
    text += "\n\n===== FILE: " + path.relative_to(ROOT).as_posix() + " =====\n" + path.read_text("utf-8-sig")
packet = OUT / "IPO_TENURE_GEMINI4_REVIEW_PACKET.txt"
packet.write_text(text, encoding="utf-8")
(OUT / "UPLOAD_MANIFEST.json").write_text(json.dumps({**manifest, "packet":dict(path=packet.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(packet.read_bytes()).hexdigest(), bytes=packet.stat().st_size)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(packet=str(packet), bytes=packet.stat().st_size, attachments=[str(p) for p in attachments]), ensure_ascii=True))
