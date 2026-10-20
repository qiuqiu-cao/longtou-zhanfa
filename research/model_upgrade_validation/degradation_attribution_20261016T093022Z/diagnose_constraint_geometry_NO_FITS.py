"""Report guards on already-registered inner thresholds; never select outer rules."""
from fractions import Fraction
from pathlib import Path
import pandas as pd
import relay_monthly_oof_alignment_research as r


def main():
    out=r.MECHANISM/'constraint_geometry';out.mkdir(exist_ok=False)
    records=[];sources={}
    for path in sorted((r.ROOT/'execution').glob('*_SELECTION_BEFORE_OUTER_SCORE.json')):
        doc=r.b.read(path);sources[str(path.resolve())]=r.b.sha(path)
        if not doc['eligible']:continue
        for arm,choice in doc['choices'].items():
            base=choice['statistics']['0.0']
            bp,br,bf=[r.old.fraction(base,k) for k in ['Precision','Recall','F1']]
            for text,value in choice['statistics'].items():
                if float(text)==0:continue
                p,rec,f=[r.old.fraction(value,k) for k in ['Precision','Recall','F1']]
                a,b=base['tp']-value['tp'],base['fp']-value['fp']
                if a<0 or b<0:raise ValueError('registered filter must only remove positives')
                f1_boundary=b*base['tp']>=a*(base['fp']+base['positives'])
                precision_boundary=b*base['tp']>a*base['fp']
                if p is not None and bp is not None:
                    assert (p>bp)==precision_boundary
                if f is not None and bf is not None:
                    assert (f>=bf)==f1_boundary
                p_up=p is not None and bp is not None and p>bp
                recall_ok=rec is not None and br is not None and rec>=br-Fraction(1,20)
                fp_ok=value['fp']*10<=base['fp']*9
                f1_ok=f is not None and bf is not None and f>=bf
                records.append(dict(month=doc['month'],arm=arm,registered_threshold=float(text),
                    tp_removed=a,fp_removed=b,base_tp=base['tp'],base_fp=base['fp'],base_positives=base['positives'],
                    Precision_up=p_up,Recall_within5pp=recall_ok,FP_drop10percent=fp_ok,F1_not_lower=f1_ok,
                    all_without_F1=p_up and recall_ok and fp_ok,all_guards=p_up and recall_ok and fp_ok and f1_ok,
                    Precision_delta=float(p-bp) if p is not None and bp is not None else None,
                    Recall_delta=float(rec-br) if rec is not None and br is not None else None,
                    F1_delta=float(f-bf) if f is not None and bf is not None else None,
                    min_FP_removal_per_removed_TP_for_F1=(base['fp']+base['positives'])/base['tp'] if base['tp'] else None))
    frame=pd.DataFrame(records)
    frame.to_csv(out/'all216_registered_inner_threshold_guard_diagnostics.csv',index=False,encoding='utf-8-sig')
    candidates=frame[frame.all_without_F1]
    candidates.to_csv(out/'three_inner_candidates_blocked_by_F1_NOT_NEW_SELECTION.csv',index=False,encoding='utf-8-sig')
    keys=['Precision_up','Recall_within5pp','FP_drop10percent','F1_not_lower','all_without_F1','all_guards']
    summary={key:int(frame[key].sum()) for key in keys}
    text=f'''# 原固定过滤条件的数学与训练期资料诊断

这是0拟合、0新阈值、0外月选参的补充归因。只复算新月度252一致OOF试验已登记的24组×9个非零阈值，共{len(frame)}格；不改变已经完成的选择、预测或验收标准。

## 过滤正确率与F1的不同代价

设基准TP=T、FP=F，完整正例=P；过滤误删a个TP、删掉b个FP。Precision严格提高要求 b×T > a×F；F1不下降要求 b×T ≥ a×(F+P)。这两个不等式已与全部登记计数的实际分数逐格精确核对。

当前完整历史SOURCE T225/F303/P871：每误删1个TP，Precision提高至少要删超过1.347个FP，F1不降则要删至少约5.218个FP。用户允许Recall最多下降5pp并不撤销已批准的F1底线，两者共同作用不能理解为任何5pp漏判损失都可接受。

旧实际FINAL删a3/b2，连Precision条件都不满足；仅保留PREFIX删a2/b3，Precision略增但F1条件不满足。这些是已发生模型方向改变的算术解释，不是资金因果。

## 新训练内登记网格实际情况

全部216格各项通过数：{summary}。仅{len(candidates)}格在Precision严格增加、Recall最多下降5pp、FP至少少10%的同时被F1门单独拦下；其余阈值有其他取舍或没有优势。三个都是第三块选阈材料，不是新外月有效结果，未据它们重新选任何阈值，也未生成放宽门后的外月预测。

结论须区分两件事：综合指标底线确实限制了可接受过滤范围；现有固定分数及九字段并没有证明足够强的个股区分。不能据本诊断擅自移除F1门宣布通过，也不能把全部不滤归因于‘数据完全没有信息’。若未来改变用户目标或验收标准，须另行明确，旧失败保持。
'''
    report=out/'过滤取舍与验收条件_仅训练期诊断.md';report.write_text(text,encoding='utf-8')
    sources[str(Path(__file__).resolve())]=r.b.sha(__file__)
    r.b.write_new(out/'constraint_geometry_verified_NO_NEW_POLICY.json',dict(formed_at=r.b.now(),
        inner_registered_thresholds=len(frame),guards_count=summary,new_fits=0,new_thresholds=0,
        outer_new_rules_evaluated=False,old_acceptance_changed=False,
        source_bindings=sources,output_bindings={str(p.resolve()):r.b.sha(p) for p in out.iterdir() if p.is_file()}))
    print(summary)


if __name__=='__main__':main()
