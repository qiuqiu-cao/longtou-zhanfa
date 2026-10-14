"""Display model tendencies for newly registered plans; raw calls remain intact."""
from datetime import datetime
import re

EFFECTIVE_AT = "2026-10-11T16:00:49+00:00"
LABELS = {"PROMOTE": "模型偏晋级", "FAIL": "模型偏断板", "UNKNOWN": "未知"}


def model_call_label(call):
    return LABELS.get(call, "未知")


def render_tendencies(rendered, registered):
    recorded = registered.get("recorded_at")
    if not recorded or datetime.fromisoformat(recorded.replace("Z", "+00:00")) < datetime.fromisoformat(EFFECTIVE_AT):
        return rendered
    lines = rendered.splitlines()
    active = False
    for index, line in enumerate(lines):
        if line == "【六、连板晋级预测】":
            active = True
            lines[index] += "\n模型判断表示倾向，保留原PROMOTE/FAIL、分数和排序。支持、反证、缺口及开盘待确认条件分别核对；缺证据不新增硬否决。"
        elif active and line.startswith("【"):
            active = False
        if active and re.match(r"^\| \d{6} ", line):
            cells = line.split("|")
            if len(cells) >= 8 and cells[4].strip() in ("晋级", "断板"):
                cells[4] = " 模型偏" + cells[4].strip() + " "
                lines[index] = "|".join(cells)
    return "\n".join(lines) + "\n"
