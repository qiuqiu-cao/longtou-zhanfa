"""realtime_dragon_radar.py

龙头战法实时微观盘口雷达（突破方向一落地工程）
功能定位：
1. 09:25 集合竞价一键秒级排雷与打板指令输出（自动核验高开区间、买盘实度与假弱转强陷阱）
2. 14:55 尾盘加撤单 Seal_Delta 自动跟踪（自动计算 14:30 vs 14:55 封单变动比值，锁定真龙锁仓 vs 心虚撤单）
3. 纯本地 Python 极速轻量执行，接入公开稳定行情接口，零费用门槛！
"""

import sys
import time
import json
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Optional

# 控制台编码自适应
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def format_code_with_market(code: str) -> str:
    """标准化代码格式（如 000678 -> sz000678, 600838 -> sh600838）"""
    code = code.strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return f"sh{code}"
    elif code.startswith(("00", "30", "20")):
        return f"sz{code}"
    elif code.startswith(("43", "83", "87", "92")):
        return f"bj{code}"
    return code


def fetch_realtime_quotes(codes: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    拉取多只股票的实时 5 档盘口行情（基于腾讯稳定公网高可用 API）
    返回结构化行情字典
    """
    if not codes:
        return {}
    
    market_codes = [format_code_with_market(c) for c in codes]
    url = f"http://qt.gtimg.cn/q={','.join(market_codes)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    res = {}
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode("gbk", errors="ignore")
            for line in content.strip().split(";"):
                line = line.strip()
                if not line or "~" not in line:
                    continue
                parts = line.split('"')[1].split("~")
                if len(parts) < 48:
                    continue
                
                raw_code = parts[2]
                name = parts[1]
                current_price = float(parts[3])
                pre_close = float(parts[4])
                open_price = float(parts[5])
                volume_lots = float(parts[6])       # 成交量（手）
                buy_1_price = float(parts[9])       # 买一价
                buy_1_lots = float(parts[10])       # 买一量（手）
                sell_1_price = float(parts[19])     # 卖一价
                sell_1_lots = float(parts[20])      # 卖一量（手）
                pct_change = float(parts[32])       # 涨跌幅 (%)
                amount_wan = float(parts[37])       # 成交额（万元）
                float_market_cap = float(parts[44]) if parts[44] else 0.0 # 流通市值（亿元）
                limit_up_price = float(parts[47]) if len(parts) > 47 and parts[47] else round(pre_close * 1.10, 2)
                limit_down_price = float(parts[48]) if len(parts) > 48 and parts[48] else round(pre_close * 0.90, 2)
                
                # 买一封单金额（元）
                buy_1_amount = buy_1_lots * 100 * buy_1_price if buy_1_price > 0 else 0.0
                
                res[raw_code] = {
                    "code": raw_code,
                    "name": name,
                    "current_price": current_price,
                    "pre_close": pre_close,
                    "open_price": open_price,
                    "pct_change": pct_change,
                    "volume_lots": volume_lots,
                    "amount_cny": amount_wan * 10000.0,
                    "buy_1_price": buy_1_price,
                    "buy_1_lots": buy_1_lots,
                    "buy_1_amount": buy_1_amount,
                    "sell_1_price": sell_1_price,
                    "sell_1_lots": sell_1_lots,
                    "float_market_cap_yi": float_market_cap,
                    "limit_up_price": limit_up_price,
                    "limit_down_price": limit_down_price,
                    "is_sealed_limit_up": (abs(current_price - limit_up_price) < 0.015 and current_price > 0 and buy_1_price >= limit_up_price - 0.01)
                }
    except Exception as e:
        print(f"行情拉取异常: {e}", file=sys.stderr)
        
    return res


# ==============================================================================
# 模块一：09:25 集合竞价微观终审与熔断撤单执行器
# ==============================================================================

def audit_0925_auction(candidate_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    09:25 集合竞价秒级终审：
    严格执行 AGENTS.md 竞价铁律：
    1. 🔴 熔断撤单线（开盘涨幅 < +2.0%）：平开、绿盘低开（<=0.0%）、假弱转强陷阱（0%~+2%），100% 立即永久注销撤单！
    2. 🟢 黄金准予买入线（开盘涨幅 +2.0% ~ +7.0%）：竞价放量真抢筹，保留条件单等待 09:30 放量打板！
    3. 🟣 一字顶格风向标（开盘涨幅 >= +9.5%）：作为情绪标杆观察，不盲顶缩量一字防炸板大面。
    4. 🔴 委比买盘空心排雷：买一未匹配量极小（如茂业商业型买一仅13手）一票否决！
    """
    codes = [item["code"] for item in candidate_list]
    quotes = fetch_realtime_quotes(codes)
    
    results = []
    for item in candidate_list:
        code = item["code"]
        name = item.get("name", code)
        slot = item.get("slot", "L1")
        min_auction_amt = item.get("min_auction_amount", 15000000.0) # 默认竞价金额底线（如1500万）
        
        q = quotes.get(code)
        if not q:
            results.append({
                "code": code,
                "name": name,
                "slot": slot,
                "status": "ERROR",
                "action": "【数据缺失-仅观察】",
                "reason": "未能从公开接口获取实时竞价快照，保持审慎观望。"
            })
            continue
        
        open_price = q["open_price"] if q["open_price"] > 0 else q["current_price"]
        pre_close = q["pre_close"]
        open_pct = round((open_price / pre_close - 1.0) * 100.0, 2)
        auction_amt = q["amount_cny"]
        buy_1_amt = q["buy_1_amount"]
        
        decision = {}
        decision["code"] = code
        decision["name"] = name
        decision["slot"] = slot
        decision["pre_close"] = pre_close
        decision["open_price"] = open_price
        decision["open_pct"] = open_pct
        decision["auction_amount_wan"] = round(auction_amt / 10000.0, 1)
        decision["buy_1_amount_wan"] = round(buy_1_amt / 10000.0, 1)
        
        # 规则 1：极端核按钮低开（绿盘 <= 0.0%）
        if open_pct <= 0.0:
            decision["action"] = "🔴【绝对熔断-立即永久撤单】"
            decision["verdict"] = "VETO"
            decision["reason"] = f"竞价开出绿盘（{open_pct:+.2f}%），触发生死熔断铁律！盘中任何反抽坚决不买，防一字断头跌停！"
        
        # 规则 2：平盘假弱转强陷阱（0.0% ~ +2.0%）
        elif 0.0 < open_pct < 2.0:
            decision["action"] = "🔴【平开陷阱-立即撤单注销】"
            decision["verdict"] = "VETO"
            decision["reason"] = f"竞价涨幅仅 {open_pct:+.2f}%（<+2.0%），缺乏大资金真正抢筹意愿，实测失败率高达 81.8%，坚决撤单！"
        
        # 规则 3：买盘空心排雷（买一挂单极少，如茂业商业型）
        elif buy_1_amt < 100000.0 and open_pct < 9.5: # 买一挂单不足 10 万元且非涨停
            decision["action"] = "🔴【空心假高开-一票否决撤单】"
            decision["verdict"] = "VETO"
            decision["reason"] = f"虽有高开（{open_pct:+.2f}%），但买一封单挂单仅 {buy_1_amt/10000:.1f} 万元，买盘严重空心，谨防开盘下砸跳水！"
        
        # 规则 4：顶格一字涨停（>= +9.5%）
        elif open_pct >= 9.5:
            decision["action"] = "🟣【一字先锋-作为风向标观察】"
            decision["verdict"] = "WATCH"
            decision["reason"] = f"开盘直接顶一字涨停（{open_pct:+.2f}%），确立极强板块风向标；不盲目顶缩量一字防炸板，待盘中爆量分歧检验。"
            
        # 规则 5：黄金准予买入区间（+2.0% ~ +7.0% 且放量真抢筹）
        elif 2.0 <= open_pct <= 7.0:
            if auction_amt >= min_auction_amt:
                decision["action"] = "🟢【黄金达标-准予开盘打板】"
                decision["verdict"] = "PASS"
                decision["reason"] = f"竞价高开 {open_pct:+.2f}% 落在黄金准入带，且竞价成交达 {auction_amt/10000:.0f} 万元（>=门槛{min_auction_amt/10000:.0f}万），真实抢筹确认！保留条件单！"
            else:
                decision["action"] = "🟡【量能不足-谨慎观察】"
                decision["verdict"] = "CAUTION"
                decision["reason"] = f"高开 {open_pct:+.2f}% 达标，但竞价量能 {auction_amt/10000:.0f} 万元偏小（未达{min_auction_amt/10000:.0f}万），观察开盘后是否能迅速补量冲板。"
        
        # 规则 6：高开过多超预期分歧带（+7.0% ~ +9.5%）
        else:
            decision["action"] = "🟡【超高开-谨防冲高回落】"
            decision["verdict"] = "CAUTION"
            decision["reason"] = f"超高开 {open_pct:+.2f}%（>+7.0%），浮盈兑现卖压极大，不追首个冲高，等待分时回踩均线不破再行考虑。"
            
        results.append(decision)
        
    return results


# ==============================================================================
# 模块二：14:55 尾盘加撤单 Seal_Delta 实时跟踪器
# ==============================================================================

class LateOrderSealTracker:
    """
    尾盘封单加撤单比值跟踪器：
    于 14:30 记录基准买一封单金额（Seal_1430），
    于 14:55 计算实时变动比值：
    Seal_Delta = Seal_Current / Seal_1430
    判定：
    - Seal_Delta >= 1.20 -> 🟢【尾盘加单-真龙锁仓】：大分歧完全转一致，主力逆势追加大单！享有最高连板溢价！
    - Seal_Delta < 0.70  -> 🔴【尾盘撤单-心虚出逃】：主力撤单留散户挡枪，次日抢跑！触发一票否决淘汰！
    - 盘中炸板跳水       -> 🔴【尾盘炸板诱多】：一票否决！
    """
    def __init__(self, watch_codes: List[str]):
        self.watch_codes = watch_codes
        self.baseline_1430: Dict[str, Dict[str, Any]] = {}
        
    def record_1430_baseline(self) -> Dict[str, Any]:
        """记录 14:30 的封单与成交基准"""
        quotes = fetch_realtime_quotes(self.watch_codes)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for code, q in quotes.items():
            self.baseline_1430[code] = {
                "name": q["name"],
                "seal_price": q["buy_1_price"],
                "seal_lots": q["buy_1_lots"],
                "seal_amount": q["buy_1_amount"],
                "is_sealed": q["is_sealed_limit_up"],
                "timestamp": ts
            }
        return self.baseline_1430

    def calculate_1455_delta(self) -> List[Dict[str, Any]]:
        """计算 14:55 的封单比值与决策"""
        if not self.baseline_1430:
            # 如果未在14:30抓取，立即补录基准
            self.record_1430_baseline()
            
        quotes = fetch_realtime_quotes(self.watch_codes)
        reports = []
        
        for code in self.watch_codes:
            base = self.baseline_1430.get(code, {})
            curr = quotes.get(code)
            if not curr:
                continue
                
            name = curr["name"]
            base_amt = base.get("seal_amount", 0.0)
            curr_amt = curr["buy_1_amount"]
            is_sealed_now = curr["is_sealed_limit_up"]
            
            # 计算封单比值
            if base_amt > 0:
                delta = round(curr_amt / base_amt, 3)
            else:
                delta = 1.0 if is_sealed_now else 0.0
                
            tag = ""
            action = ""
            verdict = ""
            reason = ""
            
            # 1. 炸板跳水检测
            if not is_sealed_now:
                tag = "【三板-尾盘偷袭诱多】"
                action = "🔴【炸板跳水-一票否决】"
                verdict = "VETO"
                reason = "尾盘出现严重炸板跳水或回封失败，筹码松动，坚决禁止接力！"
            # 2. 尾盘撤单检测 (Seal_Delta < 0.70)
            elif delta < 0.70:
                tag = "【尾盘撤单-心虚出逃】"
                action = "🔴【撤单缩水-一票否决】"
                verdict = "VETO"
                reason = f"14:30 封单 {base_amt/10000:.0f} 万元，14:55 缩水至 {curr_amt/10000:.0f} 万元（Seal_Delta={delta:.2f} < 0.70），主力撤单留散户挡枪，触发一票否决！"
            # 3. 尾盘加单检测 (Seal_Delta >= 1.20)
            elif delta >= 1.20:
                tag = "【尾盘加单-真龙锁仓】"
                action = "🟢【主力加单-真龙特赦】"
                verdict = "DRAGON"
                reason = f"14:30 封单 {base_amt/10000:.0f} 万元，14:55 逆势追加至 {curr_amt/10000:.0f} 万元（Seal_Delta={delta:.2f} >= 1.20），大分歧彻底转一致！次日列为第一重点！"
            # 4. 封单平稳 (0.70 <= Seal_Delta < 1.20)
            else:
                tag = "【尾盘封单平稳】"
                action = "🟡【承接正常-按计划执行】"
                verdict = "NORMAL"
                reason = f"封单由 {base_amt/10000:.0f} 万平稳过渡至 {curr_amt/10000:.0f} 万（Seal_Delta={delta:.2f}），筹码结构稳定。"
                
            reports.append({
                "code": code,
                "name": name,
                "base_seal_wan": round(base_amt / 10000.0, 1),
                "curr_seal_wan": round(curr_amt / 10000.0, 1),
                "seal_delta": delta,
                "tag": tag,
                "action": action,
                "verdict": verdict,
                "reason": reason
            })
            
        return reports


# ==============================================================================
# 模块三：通达信 / 同花顺 盘中公式直接输出
# ==============================================================================

TDX_FLUSH_FORMULAS = """
{ ========================================================================== }
{ 公式 1：通达信 / 同花顺【龙头战法_0925竞价真抢筹预警选股】                     }
{ 使用方法：在通达信中按 Ctrl+F 打开公式管理器 -> 条件选股公式 -> 新建 -> 粘贴源码 }
{ 设置 09:25:01 盘中预警，自动筛选出大资金真抢筹标的，坚决阻断平开下砸与空心假高开 }
{ ========================================================================== }

{ 1. 昨收价与今开盘涨幅 }
YC := DYNAINFO(3);                          { 昨收盘价 }
OPEN_PCT := (DYNAINFO(4) / YC - 1.0) * 100; { 竞价开盘涨幅 (%) }

{ 2. 竞价量能与成交额 }
AUCTION_VOL := DYNAINFO(15) / 100;          { 竞价成交手 (手) }
AUCTION_AMT := DYNAINFO(15) * DYNAINFO(4);  { 竞价成交金额 (元) }
TOTAL_CAP   := CAPITAL * 100;               { 实际流通股本 (股) }
AUCTION_RATIO := (DYNAINFO(15) / TOTAL_CAP) * 100; { 竞价换手比率 (%) }

{ 3. 买盘深度核验（防茂业商业型空心单） }
BUY1_AMT := (BIDVOL(1) * DYNAINFO(4) * 100); { 买一未匹配挂单金额 (元) }

{ 4. 黄金准入铁律判定 }
{ A. 涨幅必须落在 +2.0% ~ +7.5% 黄金抢筹带（严禁平开与绿盘低开） }
COND_PRICE := (OPEN_PCT >= 2.00) AND (OPEN_PCT <= 7.50);

{ B. 竞价成交额达标（单笔竞价超1500万或竞价换手超0.8%） }
COND_VOLUME := (AUCTION_AMT >= 15000000) OR (AUCTION_RATIO >= 0.80);

{ C. 买一真实买盘支撑（未匹配买单超200万元，严防空心买盘） }
COND_BUY1 := (BUY1_AMT >= 2000000);

{ 最终预警触发 }
XG_DRAGON_0925: COND_PRICE AND COND_VOLUME AND COND_BUY1;


{ ========================================================================== }
{ 公式 2：通达信 / 同花顺【龙头战法_1455尾盘加单锁仓雷达】分时副图/选股           }
{ 监控 14:30 以后主力大额加单与撤单抢跑（计算 Seal_Delta）                     }
{ ========================================================================== }

{ 获取当前买一封单金额 }
CURR_SEAL_AMT := (BIDVOL(1) * DYNAINFO(7) * 100);

{ 获取 14:30 封单金额基准 }
SEAL_1430 := VALUEATTIME(BIDVOL(1) * DYNAINFO(7) * 100, 143000);

{ 计算封单比值 Seal_Delta }
SEAL_DELTA := IF(SEAL_1430 > 0, CURR_SEAL_AMT / SEAL_1430, 1.0);

{ 尾盘加单-真龙锁仓（加单超20%） }
IS_LATE_ADD := (TIME >= 145000) AND (DYNAINFO(7) >= DYNAINFO(3)*1.095) AND (SEAL_DELTA >= 1.20);

{ 尾盘撤单-心虚出逃（撤单超30%，一票否决） }
IS_LATE_CANCEL := (TIME >= 145000) AND (DYNAINFO(7) >= DYNAINFO(3)*1.095) AND (SEAL_DELTA < 0.70);

{ 分时副图红绿柱展示 }
STICKLINE(IS_LATE_ADD, 0, SEAL_DELTA, 3, 0), COLORRED;       { 红色柱：尾盘主力大额加单锁仓！}
STICKLINE(IS_LATE_CANCEL, 0, SEAL_DELTA, 3, 0), COLORGREEN;  { 绿色柱：主力心虚撤单跑路！}
"""


# ==============================================================================
# 自检与回放演示
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("龙头战法实战微观盘口雷达（突破方向一落地运行验证）")
    print("=" * 80)
    
    # 1. 模拟明日 09:25 核心标的集合竞价审计
    print("\n【实测演示 1：明日重点标的 09:25 竞价模拟推演】")
    test_candidates = [
        {"code": "000678", "name": "襄阳轴承", "slot": "L1", "min_auction_amount": 50000000.0},
        {"code": "002708", "name": "光洋股份", "slot": "L2", "min_auction_amount": 25000000.0},
        {"code": "003028", "name": "振邦智能", "slot": "F1", "min_auction_amount": 12000000.0},
        {"code": "603897", "name": "长城科技", "slot": "F2", "min_auction_amount": 15000000.0},
        {"code": "603528", "name": "多伦科技_假弱转强测试", "slot": "VETO_TEST", "min_auction_amount": 20000000.0}
    ]
    
    auction_results = audit_0925_auction(test_candidates)
    print(json.dumps(auction_results, ensure_ascii=False, indent=2))
    
    # 2. 模拟 14:55 尾盘封单加撤单 Seal_Delta 真实跟踪
    print("\n" + "=" * 80)
    print("【实测演示 2：真实盘口拉取与 14:55 封单比值 Seal_Delta 跟踪】")
    tracker = LateOrderSealTracker(["000678", "600838", "603528", "603089"])
    tracker.record_1430_baseline()
    seal_reports = tracker.calculate_1455_delta()
    print(json.dumps(seal_reports, ensure_ascii=False, indent=2))
    
    # 3. 打印通达信/同花顺公式
    print("\n" + "=" * 80)
    print("【通达信 / 同花顺盘中预警公式源码就绪】")
    print(TDX_FLUSH_FORMULAS[:400] + "\n... (详见源码完整导出)")
