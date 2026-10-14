"""Read-only performance report for the V141 absorption forward registry."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import absorption_forward_registry_v141 as registry_code
import absorption_forward_registry_v140 as base


def trade_metrics(frame):
    returns = (pd.to_numeric(frame["net_return_after_costs"], errors="coerce").dropna()
               if "net_return_after_costs" in frame else pd.Series(dtype=float))
    if returns.empty:
        return {"completed_trades":0,"win_rate":None,"mean_net_return":None,"average_win":None,
                "average_loss":None,"payoff_ratio":None,"profit_factor":None,"large_loss_rate":None,
                "maximum_consecutive_losses":0,"trade_sequence_max_drawdown_proxy":None,
                "account_max_drawdown":None}
    wins, losses = returns[returns>0], returns[returns<=0]
    streak=maximum=0
    for value in returns:
        streak = streak+1 if value<=0 else 0; maximum=max(maximum,streak)
    equity=(1+returns).cumprod(); peak=equity.cummax(); drawdown=equity/peak-1
    return {"completed_trades":int(len(returns)),"win_rate":float((returns>0).mean()),
            "mean_net_return":float(returns.mean()),"average_win":float(wins.mean()) if len(wins) else None,
            "average_loss":float(losses.mean()) if len(losses) else None,
            "payoff_ratio":float(wins.mean()/abs(losses.mean())) if len(wins) and len(losses) and losses.mean()!=0 else None,
            "profit_factor":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else None,
            "large_loss_rate":float(returns.le(-.07).mean()),"maximum_consecutive_losses":maximum,
            "trade_sequence_max_drawdown_proxy":float(drawdown.min()),"account_max_drawdown":None}


def build(registry):
    verification=registry_code.verify(registry)
    if verification["status"]!="PASS": raise ValueError("registry verification failed")
    events=base.read_events(registry/"ledger.jsonl")
    snapshot_dates={e["trade_date"] for e in events if e.get("event_type")=="TEN_OCLOCK_SNAPSHOT"}
    settlements=[e for e in events if e.get("event_type")=="OUTCOME_SETTLEMENT"]
    settled_dates={e["trade_date"] for e in settlements}; rows=[]
    for event in settlements:
        document=json.loads((registry/event["settlement_file"]).read_text(encoding="utf-8"))
        rows.extend({**row,"trade_date":document["trade_date"]} for row in document["records"])
    frame=pd.DataFrame(rows)
    execution_counts=(frame.execution_state.value_counts().to_dict() if not frame.empty else {})
    actual=(frame[frame.execution_state.eq("ACTUAL_FILL")].sort_values("trade_date") if not frame.empty else frame)
    simulated=(frame[frame.execution_state.eq("CONSERVATIVE_SIM_FILL")].sort_values("trade_date") if not frame.empty else frame)
    candidate_groups={}
    if not frame.empty:
        observed=frame[frame.outcome_status.eq("OBSERVED")].copy()
        observed["b2_net_close_mark"]=pd.to_numeric(observed.b2_net_close_mark,errors="coerce")
        for (board,gate),part in observed.groupby(["board_group","shadow_gate"]):
            values=part.b2_net_close_mark.dropna()
            candidate_groups[f"{board}|{gate}"]={"rows":int(len(part)),"b2_known":int(len(values)),
                "b2_mean":float(values.mean()) if len(values) else None,
                "b2_positive_rate":float(values.gt(0).mean()) if len(values) else None,
                "b2_large_loss_rate":float(values.le(-.07).mean()) if len(values) else None}
    actual_metrics=trade_metrics(actual); simulated_metrics=trade_metrics(simulated)
    protocol=json.loads((registry/"protocol.json").read_text(encoding="utf-8")); gate=protocol["minimum_forward_gate"]
    each_group=(frame[frame.execution_state.isin(["ACTUAL_FILL","CONSERVATIVE_SIM_FILL"])]
                .groupby("board_group").size().to_dict() if not frame.empty else {})
    sample_gate=(actual_metrics["completed_trades"]+simulated_metrics["completed_trades"]>=gate["settled_candidate_records"]
                 and all(each_group.get(group,0)>=gate["each_board_group"]
                         for group in ("FIRST_BOARD","CONSECUTIVE_BOARD")))
    return {"version":"v142-absorption-forward-report-1","status":"DIAGNOSTIC_ONLY",
            "registry_verification":verification,"snapshot_sessions":len(snapshot_dates),
            "settled_sessions":len(settled_dates),"unsettled_sessions":sorted(snapshot_dates-settled_dates),
            "candidate_records":int(len(frame)),"execution_state_counts":execution_counts,
            "actual_fills":actual_metrics,"conservative_sim_fills":simulated_metrics,
            "candidate_b2_by_board_and_gate":candidate_groups,"minimum_sample_gate_passed":sample_gate,
            "challenge_70pct_testable":bool(sample_gate and actual_metrics["completed_trades"]>=100),
            "deployment_authorized":False,
            "limitations":["actual and simulated fills are never combined into an actual-performance claim",
                           "trade-sequence drawdown assumes one full unit per completed trade and is not account drawdown",
                           "account drawdown remains unavailable without frozen capital and position ledger"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--registry",type=Path,default=registry_code.DEFAULT_REGISTRY)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); result=build(args.registry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
