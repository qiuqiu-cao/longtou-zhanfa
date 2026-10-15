"""Future-window comparison with authenticated frozen instruction snapshots."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import joblib
import short_term_thinking_dataset_builder as b
from relay_multiday_scenarios import evaluate

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--record",required=True);p.add_argument("--bundle",required=True);p.add_argument("--asof",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    record=b.read(a.record);snapshots=b.read(ROOT/"current_forecast/instruction_snapshots.json")
    resolved={}
    for path,sha in record["source_bindings"].items():
        if Path(path).exists() and b.sha(path)==sha:resolved[path]=sha;continue
        matching=next((s for s in snapshots["documents"] if s["original_path"]==path and s["sha256"]==sha),None)
        if matching is None:raise ValueError("noninstruction source changed: "+path)
        resolved[matching["snapshot_path"]]=sha
    b.verify(resolved);bundle=joblib.load(a.bundle);b.verify(bundle["sources"])
    result=evaluate(record,bundle["bars"],bundle["sessions"],a.asof)
    result["source_bindings"]=resolved|{str(Path(path).resolve()):b.sha(path) for path in [a.record,a.bundle,__file__,ROOT/"current_forecast/instruction_snapshots.json"]}
    result["original_record_unchanged"]=True;b.write_new(a.output,result)
    print(dict(events=len(result["rows"]),known=result["new_mature_labels"],new_fits=0))


if __name__=="__main__":main()
