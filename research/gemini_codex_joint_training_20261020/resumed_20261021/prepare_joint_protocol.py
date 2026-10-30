"""Prepare a concrete shared protocol; Gemini must acknowledge before either arm fits."""
import sys
from pathlib import Path
import json
import shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import first_board_joint_seal_model as m

out=Path(__file__).resolve().parent
proposal=m.JOINT/"gemini/GEMINI_PROPOSAL_BEFORE_NEW_FITS.md"
snapshot=out/"GEMINI_ROUND2_PROPOSAL_BEFORE_JOINT_FREEZE.md"
assert not snapshot.exists()
shutil.copy2(proposal,snapshot)
qual=json.loads((m.QUAL/"qualification.json").read_text("utf-8"))
assert all(qual["qualified_arms"].values())
bindings=[m.FACTS,m.SCREEN,m.QUAL/"qualification.json",m.QUAL/"COMPLETE28503_NEW_FIELDS_NO_LABELS.csv",m.QUAL/"SPARSE_GUARD_BEFORE_ANY_FIT_NO_FEATURE_CHANGE.json",m.SOURCES/"ALL668_SOURCE_RECEIPTS.json",snapshot,
    Path(m.base.__file__),Path(m.stage.__file__),Path(m.risk.__file__),Path(m.history.__file__),Path(m.members.__file__),
    m.REFERENCE/"report.json",m.REFERENCE/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv",m.REFERENCE/"S1_ALL_PREDICTIONS.csv",m.REFERENCE/"S2_ALL_PREDICTIONS.csv",*m.base.INPUTS.values()]
protocol=dict(at=m.base.now(),status="PENDING_GEMINI_ACK_BEFORE_FITS",user_authorization="User explicitly:you and AntigravityGemini train together;monitor existing collaboration",counterpart="Antigravity Gemini3.8 Flash High,current actual selected model;not ChromeGemini4",
    hypothesis="Previously unavailable signal-day first-seal timing and dated provider closing-seal supply-demand quotes improve full-date first-close1to2Top1 beyond frozen62-field model",
    source_sha256=m.base.sha(m.__file__),qualification_sha256=m.base.sha(m.QUAL/"qualification.json"),arms={a:dict(trainer=m.OWNERS[a],fields=m.ARMS[a]) for a in m.ARMS},
    peer_proposal_snapshot=str(snapshot),peer_proposal_sha256=m.base.sha(snapshot),
    resolutions=dict(sparse_rank="Gemini's >=3observed and>=50% same-day coverage protection accepted;actual minimum7observed/min75.14%coverage,all28503newfields unchanged",
        clock="Shanghai time;09:25-09:30negative minutes,11:30-13:00openinterval NaN;afternoon deduct90min,not120;outside session NaN",
        closing_fund="Only supplier reportedSdateddailyclosingseal quote;not firstsealinstantaneous queue,verified execution or actoridentity",
        amount_floor="Codex rejects arbitrary1e6floor;allknownquote amount>=2925612yuan,zero rows affected by Gemini proposal;log1p range finite. Nonpositive/missing denominators NaN. No winsorization.",
        no_false_features="No retrospectively revised themes/reasons/group IDs,current freefloat/netflow/peakseal/finalseal/bombcount. Original62 retained,including archiveclock;olddeletion/IPOtests remain failed"),
    feature_qualification=qual["periods"],training="Same mature members as frozen baseline;daily equal total weights;HGB160/15/leaf60/lr.06/L2=8/balanced/seed42/no earlystop;last42signaldate positive-slope temporalPlattC1;raw scores rank;baseline andS2 never refit",
    validation="All original4724rows/117dates,unknowns retained. Each arm must >52screenedTop1hits,>=48non-one-priceTop1hits,>=95%decisiondatecoverage. Baseline wins ties. Save predictions before metrics.",
    selection="After both arms complete,only passing arm with maximumvalidationhits enters14monthly tests;ties choose64fields. If nonepass stop,no thresholds/parameters/features changed afterresults.",
    evaluation="One selected arm,previous252official signalpositions each month2025Jul-2026Aug;all maturelabelends strictly beforemonthstart. Preserve completepools/datecoverage/excludedpositives/unknowns. Only actual selectedfirstpromoters join savedS2 samecode second-close scores.",
    acceptance="DailyTop1next-official-close1to2>=60%,>=200known,>=95%outcomecoverage,noabstentiondates;2025H2and2026each>=55%. Historicalmultipletries not pristineblind,nofutureauctions/opens inS1.",
    stop="One fixed two-arm comparison,30minute work budget perbatch;preserve checkpoints/failures;nooldfailedarmretry,no effect-driven retuning",production_changes=False,actual_trades=0,new_forward_results=0,
    bindings={str(p.resolve()):m.base.sha(p) for p in bindings},
    acknowledgment="Gemini may change ONLY status toAGREED_BEFORE_NEW_FITS andaddagreed_at/acknowledgment_note if accepts these exact rules. Otherwise add objections to separate file;do notfit orsilentlychange fields. Codex verifies snapshot before own fit.",
    commands={a:"D:/python/python.exe first_board_joint_seal_model.py train --arm "+a+" --trainer "+m.OWNERS[a] for a in m.ARMS},evaluation_owner="Codex afterbothvalidation reports,onceonly",prospective_deployment_allowed=False)
path=m.JOINT/"JOINT_FROZEN_PROTOCOL.json"
with path.open("x",encoding="utf-8") as f:json.dump(m.base.clean(protocol),f,ensure_ascii=False,indent=2)
shutil.copy2(path,out/"PROTOCOL_DRAFT_BEFORE_GEMINI_ACK.json")
print(json.dumps(dict(status=protocol["status"],source_sha256=protocol["source_sha256"],protocol=str(path),fit_count=0),ensure_ascii=True))
