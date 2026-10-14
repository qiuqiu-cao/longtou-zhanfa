"""Train calibrated development models on policy-conditional complete trades.

Blind rows are never loaded into a feature matrix or evaluated here.  The model
cannot rank the unselected full limit-up pool and is never deployment-ready.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _matrix(rows,ids,features,target):
    selected=[]
    for decision_id in ids:
        row=rows[decision_id]
        if row.get("training_eligible") is not True or not isinstance(row.get(target),bool):
            raise ValueError("ineligible or unlabeled development row: "+decision_id)
        values=row.get("selected_features",{})
        selected.append([float(values.get(feature,np.nan)) for feature in features])
    matrix=np.asarray(selected,dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("development feature matrix contains missing or nonfinite values")
    labels=np.asarray([int(rows[decision_id][target]) for decision_id in ids],dtype=int)
    if np.unique(labels).size!=2:
        raise ValueError("each development target split requires both classes")
    return matrix,labels


def _models():
    return {
        "regularized_logistic":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=1.0,max_iter=2000,random_state=17))]),
        "hist_gradient_boosting":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=120,
                                                     l2_regularization=.1,random_state=17))]),
    }


def _logit(probability):
    clipped=np.clip(np.asarray(probability,dtype=float),1e-6,1-1e-6)
    return np.log(clipped/(1-clipped)).reshape(-1,1)


def _metrics(y,p):
    prevalence=float(np.mean(y))
    brier=float(brier_score_loss(y,p))
    baseline=float(np.mean((y-prevalence)**2))
    return dict(n=int(len(y)),prevalence=prevalence,auc=float(roc_auc_score(y,p)),brier=brier,
                brier_skill=1-brier/baseline if baseline>0 else None,
                log_loss=float(log_loss(y,p,labels=[0,1])),mean_probability=float(np.mean(p)))


def _fit_target(rows,partitions,features,target):
    x_train,y_train=_matrix(rows,partitions["train"],features,target)
    x_cal,y_cal=_matrix(rows,partitions["calibration"],features,target)
    x_val,y_val=_matrix(rows,partitions["validation"],features,target)
    fitted={}
    evaluations={}
    for name,model in _models().items():
        model.fit(x_train,y_train)
        cal_raw=model.predict_proba(x_cal)[:,1]
        calibrator=LogisticRegression(C=1000,max_iter=2000,random_state=23).fit(_logit(cal_raw),y_cal)
        val_raw=model.predict_proba(x_val)[:,1]
        val_probability=calibrator.predict_proba(_logit(val_raw))[:,1]
        fitted[name]=dict(base=model,calibrator=calibrator)
        evaluations[name]=_metrics(y_val,val_probability)
    chosen=min(evaluations,key=lambda name:(evaluations[name]["brier"],-evaluations[name]["auc"],name))
    winner=fitted[chosen]
    validation_probability=winner["calibrator"].predict_proba(
        _logit(winner["base"].predict_proba(x_val)[:,1]))[:,1]
    return dict(chosen=chosen,candidates=evaluations,validation_probability=validation_probability.tolist(),
                validation_labels=y_val.tolist()),winner


def train(dataset,split):
    if (dataset.get("training_ready") is not True
            or dataset.get("generalization_scope")!="FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY"
            or dataset.get("blind_rows_removed") is not True
            or dataset.get("blind_label_fields_present") is not False):
        raise ValueError("upstream complete-trade dataset is not ready")
    if (split.get("status")!="READY_FOR_MODEL_DEVELOPMENT" or split.get("blind_labels_not_read_by_audit") is not True
            or split.get("blind_access_control") != "SEPARATE_FEATURE_AND_LABEL_VAULT"
            or split.get("source_dataset_with_blind_labels_not_required_for_training") is not True):
        raise ValueError("purged development split is not ready")
    partitions=split["partitions"]
    development=set(partitions["train"])|set(partitions["calibration"])|set(partitions["validation"])
    if "blind" in partitions or sum(len(partitions[name]) for name in ("train","calibration","validation"))!=len(development):
        raise ValueError("sealed development split must not expose blind identities")
    rows={row["decision_id"]:row for row in dataset["rows"]}
    if development!=set(rows):
        raise ValueError("development view must contain exactly development observations")
    features=dataset.get("feature_allowlist")
    if not isinstance(features,list) or not features or len(features)!=len(set(features)):
        raise ValueError("frozen feature allowlist required")
    profit_result,profit_model=_fit_target(rows,partitions,features,"complete_trade_profit_label")
    risk_result,risk_model=_fit_target(rows,partitions,features,"large_loss_label")
    x_train,_=_matrix(rows,partitions["train"],features,"complete_trade_profit_label")
    support={feature:{"minimum":float(np.min(x_train[:,index])),"maximum":float(np.max(x_train[:,index]))}
             for index,feature in enumerate(features)}
    validation_ids=partitions["validation"]
    validation_predictions=[dict(decision_id=decision_id,
                                 signal_date=rows[decision_id].get("signal_date"),
                                 board_group=rows[decision_id].get("board_group"),
                                 emotion_active=rows[decision_id].get("emotion_active"),
                                 complete_trade_net_return=rows[decision_id].get("complete_trade_net_return"),
                                 profit_probability=profit_result["validation_probability"][index],
                                 large_loss_probability=risk_result["validation_probability"][index],
                                 profit_label=profit_result["validation_labels"][index],
                                 large_loss_label=risk_result["validation_labels"][index])
                            for index,decision_id in enumerate(validation_ids)]
    bundle=dict(feature_allowlist=features,policy_id=dataset.get("policy_id"),
                generalization_scope=dataset["generalization_scope"],profit=profit_model,risk=risk_model,
                feature_support_envelope=support,out_of_distribution_gate_frozen=True,
                trained_decision_ids=sorted(development),blind_feature_matrix_built=False,
                source_payload_including_blind_rows_parsed=False,
                blind_decision_count=split.get("blind_count"),
                blind_decision_ids_sha256=split.get("blind_decision_ids_sha256"),
                blind_labels_sha256_commitment=split.get("blind_labels_sha256_commitment"))
    card=dict(status="DEVELOPMENT_MODEL_ONLY",deployment_allowed=False,policy_id=dataset.get("policy_id"),
              generalization_scope=dataset["generalization_scope"],can_rank_unselected_full_pool=False,
              feature_allowlist=features,partition_counts={**{name:len(partitions[name]) for name in partitions},
                                                           "blind":split.get("blind_count")},
              profit_model=profit_result,risk_model=risk_result,validation_predictions=validation_predictions,
              blind_metrics=None,blind_feature_matrix_built=False,blind_label_fields_accessed=False,
              source_payload_including_blind_rows_parsed=False,blind_count=split.get("blind_count"),threshold_selected=False,
              blind_label_access_control="SEPARATE_FEATURE_AND_LABEL_VAULT",
              blind_decision_ids_sha256=split.get("blind_decision_ids_sha256"),
              blind_labels_sha256_commitment=split.get("blind_labels_sha256_commitment"),
              feature_support_envelope=support,out_of_distribution_gate_frozen=True,
              limitations=["model selection uses validation Brier score; no threshold is selected here",
                           "training inputs exclude blind rows; organizationally independent custody is still preferable",
                           "candidates outside the frozen training min/max support envelope must abstain",
                           "predictions apply only inside the frozen selection-policy population"])
    # Keep serialized arrays out of the JSON card after validation rows are built.
    for target in (card["profit_model"],card["risk_model"]):
        target.pop("validation_probability")
        target.pop("validation_labels")
    return card,bundle


def run(dataset_path,split_path,output_dir):
    dataset_path,split_path,output_dir=map(Path,(dataset_path,split_path,output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new immutable model output directory")
    dataset=json.loads(dataset_path.read_text(encoding="utf-8"))
    split=json.loads(split_path.read_text(encoding="utf-8"))
    card,bundle=train(dataset,split)
    card["generated_at"]=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True,exist_ok=False)
    artifact=output_dir/"policy_conditional_model.joblib"
    joblib.dump(bundle,artifact,compress=3)
    card["source_hashes"]={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (dataset_path,split_path,Path(__file__))}
    card["model_artifact_sha256"]=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output_dir/"model_card.json").write_text(json.dumps(card,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return card


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--split",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.split,args.output),ensure_ascii=False,indent=2))
