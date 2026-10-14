"""Select a validation-only threshold pair; never inspect or score blind rows."""
import argparse
import hashlib
import json
import math
from datetime import date,datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from leader_target_screen_v17 import timestamp,finite


TIE_BREAK="more_trades_then_lower_large_loss_then_higher_win_rate_then_stricter_profit"


def _wilson(success,n,z=1.96):
    if not n:
        return None
    p=success/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-radius),min(1,center+radius)]


def _week(value):
    parsed=date.fromisoformat(value)
    iso=parsed.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def validate(model_card,config):
    if (model_card.get("status")!="DEVELOPMENT_MODEL_ONLY" or model_card.get("blind_metrics") is not None
            or model_card.get("threshold_selected") is not False):
        raise ValueError("development model card required and blind metrics must remain unread")
    if model_card.get("can_rank_unselected_full_pool") is not False:
        raise ValueError("model generalization scope changed")
    try:
        if not timestamp(config["frozen_at"])<=timestamp(config["available_at"])<=timestamp(model_card["generated_at"]):
            raise ValueError("threshold grid not frozen before model result")
        target=finite(config["target_win_rate"])
        loss_cap=finite(config["max_big_loss_rate"])
        minimum=int(config["minimum_selected_trades"])
        min_week,max_week=[finite(config[key]) for key in ("minimum_signals_per_active_week","maximum_signals_per_active_week")]
        profit_grid=[finite(value) for value in config["profit_thresholds"]]
        risk_grid=[finite(value) for value in config["large_loss_thresholds"]]
        valid=(bool(config.get("config_id")) and target==.70 and loss_cap==.10 and minimum>=30
               and min_week==3 and max_week==5 and config.get("tie_break")==TIE_BREAK
               and profit_grid and risk_grid and all(v is not None and 0<=v<=1 for v in profit_grid+risk_grid)
               and profit_grid==sorted(set(profit_grid)) and risk_grid==sorted(set(risk_grid)))
    except (KeyError,TypeError,ValueError):
        valid=False
    if not valid:
        raise ValueError("invalid or unfrozen threshold validation config")
    predictions=model_card.get("validation_predictions",[])
    if len({row.get("decision_id") for row in predictions})!=len(predictions):
        raise ValueError("duplicate validation prediction")
    normalized=[]
    for row in predictions:
        try:
            profit,risk,ret=[finite(row[key]) for key in ("profit_probability","large_loss_probability","complete_trade_net_return")]
            if (profit is None or risk is None or ret is None or not 0<=profit<=1 or not 0<=risk<=1
                    or not isinstance(row["profit_label"],int) or isinstance(row["profit_label"],bool)
                    or not isinstance(row["large_loss_label"],int) or isinstance(row["large_loss_label"],bool)
                    or row["profit_label"] not in (0,1) or row["large_loss_label"] not in (0,1)
                    or row["profit_label"]!=int(ret>0) or row["large_loss_label"]!=int(ret<=-.07)
                    or not isinstance(row["emotion_active"],bool) or row.get("board_group") not in {"FIRST_BOARD","CONSECUTIVE_BOARD"}):
                raise ValueError("invalid validation prediction")
            _week(row["signal_date"])
            if timestamp(row["signal_date"]+"T15:00:00+08:00")>=timestamp(model_card["generated_at"]):
                raise ValueError("validation observation is not historical at model generation")
            normalized.append(dict(row,profit_probability=profit,large_loss_probability=risk,
                                   complete_trade_net_return=ret))
        except (KeyError,TypeError,ValueError):
            raise ValueError("invalid validation prediction row")
    signal_dates=[row["signal_date"] for row in normalized]
    if len(signal_dates)!=len(set(signal_dates)):
        raise ValueError("one-stock validation requires at most one policy trade per signal date")
    active_weeks={_week(row["signal_date"]) for row in normalized if row["emotion_active"]}
    grid=[]
    for profit_threshold in profit_grid:
        for risk_threshold in risk_grid:
            selected=[row for row in normalized if row["emotion_active"]
                      and row["profit_probability"]>=profit_threshold
                      and row["large_loss_probability"]<=risk_threshold]
            n=len(selected)
            wins=sum(row["profit_label"] for row in selected)
            losses=sum(row["large_loss_label"] for row in selected)
            win_rate=wins/n if n else None
            loss_rate=losses/n if n else None
            weekly=n/len(active_weeks) if active_weeks else None
            mean_return=sum(row["complete_trade_net_return"] for row in selected)/n if n else None
            passed=(n>=minimum and win_rate is not None and win_rate>=target and loss_rate<=loss_cap
                    and mean_return>0 and weekly is not None and min_week<=weekly<=max_week)
            grid.append(dict(profit_threshold=profit_threshold,large_loss_threshold=risk_threshold,
                             selected_trades=n,win_rate=win_rate,win_rate_wilson95=_wilson(wins,n),
                             big_loss_rate=loss_rate,big_loss_rate_wilson95=_wilson(losses,n),
                             mean_net_return=mean_return,signals_per_active_week=weekly,
                             first_board_count=sum(row["board_group"]=="FIRST_BOARD" for row in selected),
                             consecutive_board_count=sum(row["board_group"]=="CONSECUTIVE_BOARD" for row in selected),
                             passed=passed))
    feasible=[row for row in grid if row["passed"]]
    chosen=(min(feasible,key=lambda row:(-row["selected_trades"],row["big_loss_rate"],-row["win_rate"],
                                         -row["profit_threshold"],row["large_loss_threshold"])) if feasible else None)
    return dict(status="VALIDATION_THRESHOLD_FROZEN" if chosen else "NO_VALIDATED_THRESHOLD",
                config_id=config["config_id"],policy_id=model_card.get("policy_id"),
                selection_policy_id=model_card.get("policy_id"),chosen_thresholds=chosen,grid=grid,
                frozen_at=config["frozen_at"],available_at=config["available_at"],
                grid_size=len(grid),validation_observations=len(normalized),active_weeks=len(active_weeks),
                grid_frozen_before_model_output=True,
                validation_selected_trades=chosen["selected_trades"] if chosen else None,
                validation_win_rate=chosen["win_rate"] if chosen else None,
                validation_large_loss_rate=chosen["big_loss_rate"] if chosen else None,
                validation_mean_net_return=chosen["mean_net_return"] if chosen else None,
                active_weekly_signals_within_3_to_5=bool(chosen),
                deployment_allowed=False,blind_metrics=None,blind_rows_accessed=False,
                system_action="CONTINUE_TO_INDEPENDENT_BLIND" if chosen else "ABSTAIN_AND_KEEP_DIAGNOSTIC_ONLY",
                limitations=["validation threshold applies only to the frozen selection-policy population",
                             "Wilson intervals are reported but the point target remains the configured gate",
                             "no blind prediction, threshold relaxation or deployment occurs here"])


def run(model_card_path,config_path,output_path):
    model_card_path,config_path,target=map(Path,(model_card_path,config_path,output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable threshold report")
    card=json.loads(model_card_path.read_text(encoding="utf-8"))
    config=json.loads(config_path.read_text(encoding="utf-8"))
    result=validate(card,config)
    result["validated_at"]=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    result["model_artifact_sha256"]=card.get("model_artifact_sha256")
    result["source_hashes"]={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (model_card_path,config_path,Path(__file__))}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-card",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.model_card,args.config,args.output),ensure_ascii=False,indent=2))
