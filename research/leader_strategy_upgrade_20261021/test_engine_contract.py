"""Behavioral regressions for unknowns, failed-board feedback and whole-pool scope."""
from pathlib import Path
import sys,json,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import leader_emotion_engine as e

def stock(code,name,boards=1,kind='limit_up',first=93000):
 return dict(code=code,name=name,boards=boards,kind=kind,seal_first=first,
             seal_last=first,close=10.,reported_open_count=0,one_price=False)

class EmotionContract(unittest.TestCase):
 def test_unknown_promotion_not_negative(self):
  rows=[dict(code='1',name='甲',prior_boards=2,promoted=1,return_pct=10.,open_gap_pct=5.),
        dict(code='2',name='乙',prior_boards=2,promoted=None,return_pct=None,open_gap_pct=None)]
  tier=e.compute_market_ladder_emotion(rows)['tier_2_into_3']
  self.assertEqual(tier['denominator'],2)
  rate=tier.get('promotion_rate_complete_cohort',tier.get('promotion_rate'))
  self.assertIsNone(rate,'Unknown outcome cannot be silently counted as a failed promotion')

 def test_failed_name_peer_not_positive_diffusion(self):
  cur=[stock('1','东方海洋'),stock('2','海洋王',4),stock('3','盛洋科技',1,'failed_limit_up')]
  x=e.compute_theme_name_proxy_feedback(cur[0],cur)
  self.assertNotIn('3',{r['code'] for r in x['low_tier_diffusion']})

 def test_missing_seal_is_unknown_for_all_peers(self):
  cur=[stock('1','甲'),stock('2','乙',first=None)]
  self.assertIsNone(e.compute_stock_relative_competition(cur[0],cur)['seal_first_rank_in_tier'])

 def test_equal_seal_times_equal_rank(self):
  cur=[stock('1','甲'),stock('2','乙')]
  a=e.compute_stock_relative_competition(cur[0],cur)
  b=e.compute_stock_relative_competition(cur[1],cur)
  self.assertEqual(a['seal_first_rank_in_tier'],b['seal_first_rank_in_tier'])
  self.assertTrue(a['is_earliest_in_tier'] and b['is_earliest_in_tier'])

 def test_missing_fields_not_zero_or_false(self):
  row=stock('1','甲');del row['reported_open_count'];del row['one_price']
  x=e.compute_stock_relative_competition(row,[row])
  self.assertIsNone(x['reported_open_count'])
  self.assertIsNone(x['one_price'])

 def test_duplicate_code_rejected(self):
  facts=dict(signal_date='2026-10-21',target_date='2026-10-22',rows=[stock('1','甲'),stock('1','甲')])
  prior=dict(signal_date='2026-10-21',prior_date='2026-10-20',rows=[])
  with self.assertRaises(ValueError):e.build_emotion_context(facts,prior)

 def test_wrong_feedback_date_rejected(self):
  facts=dict(signal_date='2026-10-21',target_date='2026-10-22',rows=[stock('1','甲')])
  prior=dict(signal_date='2026-10-20',prior_date='2026-10-19',rows=[])
  with self.assertRaises(ValueError):e.build_emotion_context(facts,prior)

 def test_real_complete_cohort(self):
  r=ROOT/'results/joint_review_20261021T084800Z/facts'
  facts=json.loads((r/'complete_enriched_facts_SCORES_MASKED.json').read_text(encoding='utf-8'))
  prior=json.loads((r/'COMPLETE_PRIOR_COHORT_FEEDBACK.json').read_text(encoding='utf-8'))
  ctx=e.build_emotion_context(facts,prior)
  self.assertEqual(ctx['current_pool_size'],67)
  self.assertEqual(len(ctx['candidate_emotion_cards']),67)
  tier=ctx['market_ladder_emotion']['tier_2_into_3']
  self.assertEqual(tier['denominator'],7)
  self.assertEqual(tier['promoted_count'],1)
  self.assertAlmostEqual(tier['return_distribution']['mean_pct'],1.7144725413,places=2)

if __name__=='__main__':unittest.main(verbosity=2)
