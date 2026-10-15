"""Identity discovery boundaries, not measured trading/prediction improvement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stock_judgement_review as o
import supplement_identity_discovery as d
import relay_review_workbench as w


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.days = ['2026-10-08','2026-10-09','2026-10-12','2026-10-13','2026-10-14']
        self.child = dict(code='603096',name='新经典',boards=4,industry='出版传媒')
        self.peer = dict(code='002343',name='慈文传媒',boards=4,industry='影视传媒')
        self.parent = dict(code='600825',name='新华传媒',boards=8,industry='传媒')
        self.pool = dict(signal_date=self.days[-1],rows=[self.child,self.peer])
        self.snapshots = {}
        for i,day in enumerate(self.days):
            rows = {}
            if i < 3:
                rows[self.parent['code']] = dict(self.parent,boards=8+i)
            if i > 0:
                for stock in (self.child,self.peer):
                    rows[stock['code']] = dict(stock,boards=i)
            self.snapshots[day] = dict(rows=rows)

    def run_case(self,claims=(),edges=()):
        return d.analyze(self.pool,self.snapshots,self.days,claims,edges)

    def test_autonomously_finds_unseeded_parent_and_preserves_tied_competitors(self):
        r=self.run_case()
        self.assertIn('600825',[a['code'] for a in r['anchor_candidates']])
        for row in r['rows']:
            self.assertEqual(row['identity_hypothesis'],'THEME_RELAY_COMPETING')
            h=next(h for h in row['anchor_hypotheses'] if h['anchor_code']=='600825')
            self.assertEqual(h['timing_type'],'EARLY_START_SURVIVES_PARENT_BREAK')
            self.assertEqual(row['same_cycle_competitors'][0]['all_highest_streak_candidates'],['002343','603096'])
            self.assertFalse(row['confirmed_catchup_dragon'])
        self.assertFalse(r['model_scores_used'])

    def test_later_successful_peer_is_not_inferred_to_be_parent(self):
        for row in self.run_case()['rows']:
            self.assertNotIn('002343' if row['code']=='603096' else '603096',
                             [h['anchor_code'] for h in row['anchor_hypotheses']])

    def test_theme_comparison_retains_first_board_outside_frozen_cohort(self):
        self.snapshots[self.days[-1]]['rows']['601811']=dict(code='601811',name='新华文轩',boards=1,industry='出版')
        result=self.run_case()
        self.assertEqual(result['pool_size'],2)
        for row in result['rows']:
            context=next(c for c in row['current_theme_context_including_first_boards'] if c['code']=='601811')
            self.assertFalse(context['in_frozen_candidate_pool'])
            self.assertFalse(context['current_cycle_membership_confirmed'])
        self.assertEqual(len(result['current_complete_provider_closing_pool']),3)

    def test_industry_words_do_not_confuse_additives_electronics_and_cooling(self):
        self.assertNotIn('CONSUMER',d.families('乙腈涨价+食品饲料添加剂'))
        self.assertNotIn('AGRICULTURE',d.families('食品饲料添加剂'))
        self.assertNotIn('CONSUMER',d.families('消费电子'))
        self.assertNotIn('BATTERY',d.families('储能温控+冷热设备'))
        self.assertNotIn('CHEMICAL',d.families('化学制药'))
        self.assertIn('BATTERY',d.families('锂电池+储能'))

    def test_sentence_subjects_prevent_paragraph_relationship_transfer(self):
        for snapshot in self.snapshots.values():
            for row in snapshot['rows'].values():
                if row['code']!='600825': row['industry']='化工'
        claim=dict(claim_id='sector_paragraph',excerpt='新经典、慈文传媒。善水科技跟随新华传媒超预期。')
        result=self.run_case([claim])
        for row in result['rows']:
            self.assertNotEqual(row['identity_hypothesis'],'CYCLE_SUPPLEMENT_CANDIDATE')
            for h in row['anchor_hypotheses']:
                self.assertEqual(h['sentence_pair_connection_claim_ids'],[])

    def test_explicit_pair_produces_only_candidate_not_role_truth(self):
        claim=dict(claim_id='pair',excerpt='新经典接力新华传媒，补涨需要继续核验。')
        row=next(r for r in self.run_case([claim])['rows'] if r['code']=='603096')
        self.assertEqual(row['identity_hypothesis'],'CYCLE_SUPPLEMENT_CANDIDATE')
        self.assertIsNone(row['formal_role_label'])
        self.assertIsNone(row['promotion_direction'])

    def test_question_and_name_alone_do_not_produce_supported_supplement(self):
        claim=dict(claim_id='question',excerpt='新经典是接力新华传媒的补涨吗？')
        row=next(r for r in self.run_case([claim])['rows'] if r['code']=='603096')
        h=next(h for h in row['anchor_hypotheses'] if h['anchor_code']=='600825')
        self.assertFalse(h['source_connection_supported'])
        self.assertEqual(h['question_only_claim_ids'],['question'])
        self.assertNotEqual(row['identity_hypothesis'],'CYCLE_SUPPLEMENT_CANDIDATE')

    def test_missing_market_day_cannot_be_inferred_parent_break(self):
        del self.snapshots['2026-10-13']
        row=next(r for r in self.run_case()['rows'] if r['code']=='603096')
        h=next(h for h in row['anchor_hypotheses'] if h['anchor_code']=='600825')
        self.assertIsNone(h['parent_not_closing_in_complete_provider_pool_since'])
        self.assertEqual(h['missing_pool_days'],['2026-10-13'])
        self.assertFalse(row['current_streak']['complete_closing_streak_checked'])

    def test_old_observation_and_break_restart_are_distinct_from_early_start(self):
        code='603096'
        self.pool['rows'][0]['boards']=2
        for i,day in enumerate(self.days):
            self.snapshots[day]['rows'].pop(code,None)
            if i in (0,3,4):self.snapshots[day]['rows'][code]=dict(self.child,boards=1 if i in (0,3) else 2)
        claim=dict(claim_id='pair',excerpt='新经典接力新华传媒。')
        row=next(r for r in self.run_case([claim])['rows'] if r['code']==code)
        self.assertEqual(row['identity_hypothesis'],'OLD_FOLLOWER_RESTART')
        self.assertEqual(row['current_streak']['earlier_board_observed'],'2026-10-08')

    def test_suspension_or_missing_child_pool_day_does_not_certify_streak(self):
        self.snapshots['2026-10-13']['rows'].pop('603096')
        row=next(r for r in self.run_case()['rows'] if r['code']=='603096')
        self.assertFalse(row['current_streak']['complete_closing_streak_checked'])
        self.assertTrue(all(h['timing_type']=='CHILD_STREAK_UNQUALIFIED' for h in row['anchor_hypotheses']))

    def test_current_pool_cannot_omit_or_change_candidate(self):
        self.snapshots[self.days[-1]]['rows'].pop('603096')
        with self.assertRaisesRegex(ValueError,'identities/boards'):
            self.run_case()

    def test_exact_response_date_guard_remains_active(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'pool.json'
            o.write_new(p,dict(data=dict(qdate=20261013,tc=0,pool=[])))
            spec=dict(signal_date='2026-10-14',pool_snapshots=[dict(date='2026-10-14',path=str(p),
                sha256=o.sha(p),manifest='unused',kind='EASTMONEY_CLOSING_POOL')])
            with self.assertRaisesRegex(ValueError,'date'):
                d.read_snapshots(spec,o.bind(p),o.now())

    def test_price_path_uses_previous_official_session_not_last_trade(self):
        days={'2026-10-09':dict(open=100,close=110,high=110,low=100,volume_native=20),
              '2026-10-14':dict(open=120,close=120,high=120,low=120,volume_native=30)}
        path=d.price_path(dict(days=days),['2026-10-13','2026-10-14'],self.days)
        self.assertEqual(path['observations'][0]['status'],'UNKNOWN')
        signal=path['observations'][1]
        self.assertTrue(signal['one_price_session'])
        self.assertIsNone(signal['gap_vs_previous_official_close'])
        self.assertIsNone(signal['volume_ratio_vs_previous_official_session'])
        self.assertFalse(path['volume_surge_is_verified_absorption'])

    def test_raw_daily_must_rebuild_exact_normalized_prices(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw,normalized,manifest=[root/n for n in ['raw.json','normalized.json','manifest.json']]
            o.write_new(raw,dict(data=dict(sz000001=dict(day=[['2026-10-14','10','11','11','9','100']]))))
            rebuilt=d.parse_daily(o.read(raw),'000001','2026-10-08','2026-10-14')
            stamp='2026-10-14T08:00:00+00:00'
            o.write_new(normalized,{'000001':dict(rebuilt,available_at=stamp,raw_path=str(raw))})
            o.write_new(manifest,dict(requests=[dict(raw_path=str(raw),raw_sha256=o.sha(raw),kind='daily',code='000001',
                requested_at=stamp,available_at=stamp,parameters=dict(param='sz000001,day,2026-10-08,2026-10-14,40,'))]))
            spec=dict(signal_date='2026-10-14',daily_archive=dict(normalized_path=str(normalized),manifest_paths=[str(manifest)],codes=['000001']))
            proof=o.bind(raw)|o.bind(normalized)|o.bind(manifest)
            self.assertEqual(d.read_daily(spec,proof,o.now())['000001']['days'],rebuilt['days'])
            early=o.read(manifest);early['requests'][0]['requested_at']='2026-10-14T06:00:00+00:00'
            manifest.write_text(__import__('json').dumps(early),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'intraday'):
                d.read_daily(spec,proof|o.bind(manifest),o.now())
            early['requests'][0]['requested_at']=stamp
            manifest.write_text(__import__('json').dumps(early),encoding='utf-8')
            proof|=o.bind(manifest)
            data=o.read(normalized);data['000001']['days']['2026-10-14']['close']=1000
            normalized.write_text(__import__('json').dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differ'):
                d.read_daily(spec,proof|o.bind(normalized),o.now())

    def test_identity_review_must_cover_every_parent_and_competitor(self):
        context=self.run_case()
        rows=[]
        for row in context['rows']:
            rows.append(dict(code=row['code'],identity_discovery_assessment=dict(reviewed=True,
                identity_call='THEME_RELAY_COMPETING',accepted_anchor_codes=['600825'],
                rejected_or_pending_anchors=[],support='已核时序与传媒同组。',counterevidence='未核控制力。',
                cancel_condition='相对地位反证后撤销。',
                same_cycle_competitors=sorted(c for p in row['same_cycle_competitors'] for c in p['members'] if c!=row['code']))))
        with patch.object(d,'checked',return_value=context):
            d.validate_review({},dict(rows=rows))
            rows[0]['identity_discovery_assessment']['same_cycle_competitors']=[]
            with self.assertRaisesRegex(ValueError,'complete same-cycle'):
                d.validate_review({},dict(rows=rows))

    def test_workbench_rejects_pending_identity_before_original_freeze(self):
        with tempfile.TemporaryDirectory() as t:
            facts,draft=Path(t)/'facts.json',Path(t)/'draft.json'
            o.write_new(facts,dict(rows=[]));o.write_new(draft,dict(rows=[]))
            with patch.object(d,'validate_review',side_effect=ValueError('identity pending')),patch.object(w.forum,'freeze') as freeze:
                with self.assertRaisesRegex(ValueError,'identity pending'):w.freeze(facts,draft,'unused')
                freeze.assert_not_called()

    def test_compare_checks_identity_before_original_outcome_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);facts,record=root/'facts.json',root/'record.json'
            o.write_new(facts,dict(rows=[]));o.write_new(record,dict(facts_path=str(facts),rows=[]))
            with patch.object(d,'validate_review',side_effect=ValueError('identity changed')),patch.object(w.forum,'compare') as compare:
                with self.assertRaisesRegex(ValueError,'identity changed'):w.compare(record,'base','outcome',root/'comparison.json')
                compare.assert_not_called()
            self.assertEqual(o.read(root/'comparison.workbench.json')['status'],'ORIGINAL_COMPARISON_REJECTED')

    def test_prepare_forwards_optional_identity_evidence_without_losing_histories(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out'
            def forum_prepare(**kwargs):
                o.write_new(Path(kwargs['output_dir'])/'reasoning/facts_with_forum_reasoning.json',dict(rows=[]))
                return dict(rows=[])
            with patch.object(w.forum,'prepare',side_effect=forum_prepare) as prepare,patch.object(d,'attach',return_value=dict(rows=[])) as attach,patch.object(w,'render'):
                # The real identity adapter writes the enriched facts.
                def adapter(*args):
                    o.write_new(out/'identity/facts_with_identity_discovery.json',dict(rows=[]))
                    return dict(rows=[])
                attach.side_effect=adapter
                w.prepare(*(['x']*6),out,'pool','relations',history_datasets=['history'],
                    mainline_history=['mainline'],narrative_reviews='narrative',identity_evidence='packet')
                self.assertEqual(prepare.call_args.kwargs['history_datasets'],('history',))
                self.assertEqual(prepare.call_args.kwargs['mainline_history'],('mainline',))
                self.assertEqual(attach.call_args.args[1],'packet')


if __name__=='__main__':unittest.main()
