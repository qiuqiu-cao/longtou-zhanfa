"""Fixed exploratory interaction hypothesis, not new blind validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from duration_factor_ablation_v28 import compare, GROUPS

PAIRS = [("boards", "market_yesterday_premium"),
         ("sole_known_top_proxy", "market_failed_rate"),
         ("height_gap_to_known_top", "market_promotion_mid")]


def interactions(frame):
    result = frame.copy()
    names = []
    for first, second in PAIRS:
        name = "interaction_"+first+"_x_"+second
        if name in result:
            raise ValueError("interaction column already present")
        a = pd.to_numeric(result[first], errors="coerce").replace([np.inf,-np.inf],np.nan)
        b = pd.to_numeric(result[second], errors="coerce").replace([np.inf,-np.inf],np.nan)
        result[name] = (a*b).replace([np.inf,-np.inf],np.nan)
        names.append(name)
    return result,names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features","episodes","calendar","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (args.features,args.episodes,args.calendar,Path(__file__),Path("duration_factor_ablation_v28.py"))}
    # Persist fixed choices before fitting; no threshold or parameter search.
    protocol = dict(status="EXPLORATORY_REUSED_VALIDATION",pairs=PAIRS,source_hashes=sources,
                    training_exclusive_cutoff="2025-06-16",validation_start="2025-07-28",validation_end="2025-12-26",
                    C=1,horizons=[1,3,5],promotion_to_live_allowed=False)
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    features,names = interactions(pd.read_csv(args.features,dtype={"code":str}))
    groups = dict(GROUPS,identity_emotion_interactions=GROUPS["identity_plus_emotion"]+names)
    report,predictions = compare(features,pd.read_csv(args.episodes,dtype={"code":str}),
                                 pd.read_csv(args.calendar).date.tolist(),groups)
    report["experiment_status"] = "EXPLORATORY_REUSED_VALIDATION"
    report["source_hashes"] = sources
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    predictions.to_csv(args.output/"predictions_NOT_LIVE.csv",index=False)
    print(json.dumps({h:{g:{k:v["validation"][k] for k in ("auc","brier")} for g,v in groups.items()}
                      for h,groups in report["groups_by_horizon"].items()},indent=2))


if __name__ == "__main__":
    main()
