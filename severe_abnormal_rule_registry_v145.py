"""Versioned official-rule registry for main-board severe abnormal moves.

This module is deliberately small and conservative.  It verifies which rule
text and benchmark apply on a date; it does not infer exchange publications,
reset dates, corporate-action treatment, or the probability of a future hit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


SSE_RULE_2023 = (
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/"
    "c/c_20250519_10779397.shtml"
)
SSE_RULE_2026 = (
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/"
    "c/c_20260424_10816482.shtml"
)
SSE_DISCLOSURE = "https://www.sse.com.cn/disclosure/diclosure/ycjyxx/main/"
SZSE_QA_2023 = "https://investor.szse.cn/knowledge/t20230406_599705.html"
SZSE_RULE_2026 = (
    "https://docs.static.szse.cn/www/lawrules/rule/trade/current/"
    "W020260424690713155663.pdf"
)


@dataclass(frozen=True)
class SevereAbnormalRule:
    rule_id: str
    exchange: str
    effective_from: date
    effective_to: date | None
    benchmark_code: str
    benchmark_name: str
    positive_10d_threshold_pct: float
    positive_30d_threshold_pct: float
    same_direction_abnormal_count_10d: int
    formula: str
    reset_rule: str
    no_limit_session_rule: str
    source_urls: tuple[str, ...]
    verification_status: str = "verified"

    def public_record(self) -> dict:
        record = asdict(self)
        record["effective_from"] = self.effective_from.isoformat()
        record["effective_to"] = (
            self.effective_to.isoformat() if self.effective_to else None
        )
        record["source_urls"] = list(self.source_urls)
        return record


FORMULA = (
    "(stock_end_close/stock_pre_window_close-1)*100"
    "-(benchmark_end_close/benchmark_pre_window_close-1)*100"
)
RESET = "next_trading_session_or_resumption_after_exchange_publication"
NO_LIMIT = "exclude_sessions_without_price_limit"


RULES = (
    SevereAbnormalRule(
        rule_id="SSE_MAIN_20230217",
        exchange="SSE",
        effective_from=date(2023, 2, 17),
        effective_to=date(2026, 7, 5),
        benchmark_code="sh000002",
        benchmark_name="上证A股指数",
        positive_10d_threshold_pct=100.0,
        positive_30d_threshold_pct=200.0,
        same_direction_abnormal_count_10d=4,
        formula=FORMULA,
        reset_rule=RESET,
        no_limit_session_rule=NO_LIMIT,
        source_urls=(SSE_RULE_2023, SSE_DISCLOSURE),
    ),
    SevereAbnormalRule(
        rule_id="SZSE_MAIN_20230217",
        exchange="SZSE",
        effective_from=date(2023, 2, 17),
        effective_to=date(2026, 7, 5),
        benchmark_code="sz399107",
        benchmark_name="深证A股指数",
        positive_10d_threshold_pct=100.0,
        positive_30d_threshold_pct=200.0,
        same_direction_abnormal_count_10d=4,
        formula=FORMULA,
        reset_rule=RESET,
        no_limit_session_rule=NO_LIMIT,
        source_urls=(SZSE_QA_2023,),
    ),
    SevereAbnormalRule(
        rule_id="SSE_MAIN_20260706",
        exchange="SSE",
        effective_from=date(2026, 7, 6),
        effective_to=None,
        benchmark_code="sh000002",
        benchmark_name="上证A股指数",
        positive_10d_threshold_pct=100.0,
        positive_30d_threshold_pct=200.0,
        same_direction_abnormal_count_10d=4,
        formula=FORMULA,
        reset_rule=RESET,
        no_limit_session_rule=NO_LIMIT,
        source_urls=(SSE_RULE_2026, SSE_DISCLOSURE),
    ),
    SevereAbnormalRule(
        rule_id="SZSE_MAIN_20260706",
        exchange="SZSE",
        effective_from=date(2026, 7, 6),
        effective_to=None,
        benchmark_code="sz399107",
        benchmark_name="深证A股指数",
        positive_10d_threshold_pct=100.0,
        positive_30d_threshold_pct=200.0,
        same_direction_abnormal_count_10d=4,
        formula=FORMULA,
        reset_rule=RESET,
        no_limit_session_rule=NO_LIMIT,
        source_urls=(SZSE_RULE_2026,),
    ),
)


def _date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def exchange_for_mainboard_code(code: str) -> str:
    """Return exchange only for known Shanghai/Shenzhen main-board A prefixes."""
    normalized = str(code).zfill(6)
    if not normalized.isdigit() or len(normalized) != 6:
        raise ValueError("six-digit security code required")
    if normalized.startswith(("600", "601", "603", "605")):
        return "SSE"
    if normalized.startswith(("000", "001", "002", "003")):
        return "SZSE"
    raise ValueError("code is outside the supported Shanghai/Shenzhen main-board scope")


def rule_for(exchange: str, session_date) -> SevereAbnormalRule:
    venue, session = str(exchange).upper(), _date(session_date)
    matches = [
        rule for rule in RULES
        if rule.exchange == venue
        and rule.effective_from <= session
        and (rule.effective_to is None or session <= rule.effective_to)
    ]
    if len(matches) != 1:
        raise ValueError("no unique verified rule version for exchange/date")
    return matches[0]


def verified_rule_for_code(code: str, session_date) -> dict:
    return rule_for(exchange_for_mainboard_code(code), session_date).public_record()

