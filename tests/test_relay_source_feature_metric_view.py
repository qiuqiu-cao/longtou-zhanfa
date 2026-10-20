import pandas as pd
from relay_source_feature_metric_view import applicability_view,STRUCTURAL_COLUMNS


def test_pairwise_has_only_structure_no_absolute_confusion_performance_or_utility():
    source=pd.DataFrame([dict(period='x',arm='PAIRWISE_DAY_RANK',scope='B',candidates=10,known=8,unknown_outcomes=2,positives=3,
        tp=0,fp=0,fn_called=0,tn=0,Recall=0.,Precision=None,F1=0.,false_positive_rate=0.,utility_TP_minus_1p5FP=0.,
        abstentions=10,decision_coverage=0.,abstained_positives=3,metric_applicability='NO_ABSOLUTE_EVENT_HEAD')])
    result=applicability_view(source)
    assert result[[c for c in source if c not in STRUCTURAL_COLUMNS]].isna().all().all()
    assert result.structural_unknown_calls_without_absolute_head.iloc[0]==10
    assert result.positives.iloc[0]==3


def test_absolute_head_metrics_exactly_unchanged():
    source=pd.DataFrame([dict(period='x',arm='CORRECTED_CAUSAL',scope='B',candidates=10,known=8,unknown_outcomes=2,positives=3,
        tp=2,fp=1,fn_called=1,tn=4,Recall=2/3,Precision=2/3,F1=2/3,false_positive_rate=.2,utility_TP_minus_1p5FP=.5,
        abstentions=0,decision_coverage=1.,abstained_positives=0,metric_applicability='ABSOLUTE_EVENT_HEAD')])
    result=applicability_view(source)
    pd.testing.assert_frame_equal(source,result[source.columns],check_exact=True)
