import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import runpy
import unittest
from unittest.mock import patch

import numpy as np

import leader_role_engine as e
import leader_role_research as r
import leader_role_review as review


class RolesTest(unittest.TestCase):
    def test_broad_posted_theme_is_retained_without_certifying_core_comparison(self):
        row = copy.deepcopy(self.row)
        value = row["groups"]["theme"]["value"]
        value.update(scope=e.ARCHIVE_THEME_SCOPE, theme_id="consumer", granularity="specific")
        source = self.root/"broad_theme.json"
        e.write_new(source, {"formed_at": row["available_at"], "theme": value})
        row["groups"]["theme"].update(source_path=str(source), source_sha256=e.sha(source), pointer=["theme"])
        result = e.evidence_row(row, "DIAGNOSTIC", self.days)
        self.assertFalse(result["role_eligible"]["THEME"])
        self.assertFalse(result["role_eligible"]["CYCLE"])
        self.assertIn("BROAD_THEME_COMPARISON_SCOPE", result["roles"]["THEME"]["missing"])
        self.assertEqual(result["roles"]["THEME"]["baseline_call"], None)
        self.assertIsNone(result["features"]["theme_amount_share"])
        self.assertTrue(any(v["kind"] == "OBSERVED_THEME_ASSOCIATION" for v in result["relations"]))
        self.assertEqual(result["source_groups"]["theme"]["value"], value)
        self.assertTrue(result["role_eligible"]["HEIGHT"])

    def test_original_theme_wording_and_source_confidence_are_bound_without_rewriting_projection(self):
        manifest, _ = self.fixture_source_roster_manifest()
        data = e.from_source_roster(manifest, self.cal)
        original = copy.deepcopy(data["rows"])
        result = e.analyse(data, "DIAGNOSTIC")
        definition = result["rows"][0]["theme_scope_assessment"]["original_definition"]
        self.assertEqual(definition["raw_topic"], "PCB板")
        self.assertEqual(definition["source_tier"], "SYNTHETIC_FIXTURE")
        self.assertFalse(result["rows"][0]["theme_scope_assessment"]["business_membership_certified"])
        self.assertEqual(data["rows"], original)

    def test_broad_qualification_is_specific_to_original_archive_scope(self):
        value = copy.deepcopy(self.row["groups"]["theme"]["value"])
        value["theme_id"] = "medicine"
        self.assertTrue(e._valid_group("theme", value, self.row))
        value["scope"] = e.ARCHIVE_THEME_SCOPE
        self.assertFalse(e._valid_group("theme", value, self.row))
        value["theme_id"] = "pcb"
        self.assertTrue(e._valid_group("theme", value, self.row))
        value["theme_id"] = "unreviewed_new_topic"
        self.assertFalse(e._valid_group("theme", value, self.row))
        self.assertEqual(e.theme_granularity(value), "UNKNOWN_SOURCE_TOPIC_GRANULARITY")

    def test_original_theme_definition_changed_members_are_rejected(self):
        row = copy.deepcopy(self.row)
        value = row["groups"]["theme"]["value"]
        definitions = {(row["signal_date"], value["theme_id"]): {
            "source_url": value.get("source_url"), "published_at": value.get("source_published_at"), "codes": ["000099"]}}
        with self.assertRaisesRegex(ValueError, "original definition"):
            e.evidence_row(row, "DIAGNOSTIC", self.days, definitions)

    def test_unavailable_theme_does_not_change_legacy_fact_shape(self):
        row = copy.deepcopy(self.row)
        row["groups"].pop("theme")
        result = e.evidence_row(row, "STRICT", self.days)
        self.assertNotIn("theme_scope_assessment", result)
        self.assertFalse(result["role_eligible"]["THEME"])

    def test_original_theme_confidence_is_recovered_from_bound_nested_relation_bundle(self):
        manifest, _ = self.fixture_source_roster_manifest()
        data = e.from_source_roster(manifest, self.cal)
        bundle = self.root/"source_rebuilt_relation_facts.json"
        e.write_new(bundle, {"derivation_kind": "HISTORICAL_RELATION_FACTS", "source_bindings": data["source_bindings"]})
        definitions = e.theme_source_definitions(e.bind(bundle))
        self.assertEqual(definitions[("2026-10-09", "pcb")]["source_tier"], "SYNTHETIC_FIXTURE")
        raw = e.read(manifest)["evidence_file"]
        Path(raw).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            e.theme_source_definitions(e.bind(bundle))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dates = [date(2026, 8, 3)+timedelta(days=i) for i in range(90)]
        self.days = [d.isoformat() for d in dates if d.weekday() < 5]
        self.cal = self.root/"calendar.json"
        e.write_new(self.cal, {"status": "verified", "source": "synthetic_fixture_only", "sessions": self.days})
        self.source = self.root/"raw.json"
        self.source_data = {"formed_at": "2026-10-10T10:00:00+08:00", "groups": {
            "pool": {"complete_within_scope": True, "codes": ["000001", "000002"], "boards": {"000001": 3, "000002": 2}},
            "price": {"turnover": .15, "signal_one_word": 1., "price": 10.},
            "path": {"event_sequence_complete": True, "events": [
                {"observed_at": "2026-10-09T11:25:00+08:00", "state": "OPEN"},
                {"observed_at": "2026-10-09T13:05:00+08:00", "state": "SEALED"}]},
            "theme": {"membership_verified": True, "complete_within_scope": True, "granularity": "specific",
                "theme_id": "fixture", "codes": ["000001", "000002"], "members": {
                    "000001": {"amount": 100., "return_pct": 10., "boards": 3},
                    "000002": {"amount": 50., "return_pct": 5., "boards": 2}}}}}
        e.write_new(self.source, self.source_data)
        self.row = {"candidate_id": "2026-10-09:000001", "signal_date": "2026-10-09", "code": "000001",
            "name": "fixture", "boards": 3, "scope_id": "fixture", "decision_at": "2026-10-11T10:00:00+08:00",
            "available_at": "2026-10-10T10:00:00+08:00", "actual_one_word_daily": 0., "groups": {}}
        for kind in self.source_data["groups"]:
            self.row["groups"][kind] = {"source_path": str(self.source), "source_sha256": e.sha(self.source),
                "pointer": ["groups", kind], "value": self.source_data["groups"][kind], "status": "VERIFIED",
                "observed_at": "2026-10-09T15:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        self.dataset = {"version": e.VERSION, "formed_at": e.now(), "calendar_path": str(self.cal),
            "calendar_sha256": e.sha(self.cal), "source_bindings": e.bind(self.source), "rows": [self.row]}

    def test_roles_and_multi_identity(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(set(row["roles"]), set(e.ROLES))
        self.assertTrue(row["roles"]["HEIGHT"]["baseline_call"])
        self.assertEqual(row["roles"]["THEME"]["state"], "CANDIDATE")
        self.assertEqual(row["roles"]["TURNOVER"]["state"], "CANDIDATE")
        self.assertFalse(row["role_identity_automatically_certified"])

    def market_record_fixture(self, day, limits=20, phase="WARMING", status="verified"):
        raw={k:None for k in e.MARKET_RAW_FIELDS}
        raw.update(limit_count=float(limits),consecutive_count=2.,max_board=3.,failed_count=5.,down_limit_count=2.,
                   failed_rate=.2,promotion_mid=.5,yesterday_premium=.01)
        return {"date":day,"raw":raw,"rule_phase":phase,"data_status":status}

    def test_market_context_uses_exact_past_official_sessions_and_masks_future_values(self):
        day=self.days[8]
        records=[self.market_record_fixture(self.days[i],20+i) for i in range(9)]
        records.append(self.market_record_fixture(self.days[9],9999))
        context,cycle=e.market_context_window(records,day,self.days)
        self.assertEqual(context["features"]["market_limit_count_delta_1"],1.)
        self.assertEqual(context["features"]["market_limit_count_delta_3"],3.)
        self.assertEqual(context["features"]["market_limit_count_delta_5"],5.)
        self.assertEqual(context["records"][-1]["date"],day)
        self.assertNotIn("9999",json.dumps(context))
        self.assertTrue(all(r["state"]=="WARMING" for r in cycle["records"]))
        self.assertTrue(cycle["market_phase_is_not_stock_role_truth"])

    def test_market_gap_is_retained_and_does_not_certify_continuous_cycle(self):
        day=self.days[8];records=[self.market_record_fixture(self.days[i],20+i) for i in range(9) if i!=7]
        context,cycle=e.market_context_window(records,day,self.days)
        self.assertIsNone(cycle)
        self.assertIsNone(context["features"]["market_limit_count_delta_1"])
        self.assertEqual(context["records"][-2]["data_status"],"MISSING_OFFICIAL_SESSION")
        self.assertTrue(all(v is None for v in context["records"][-2]["raw"].values()))
        records.append(self.market_record_fixture(self.days[7],27,status="partial"))
        self.assertIsNone(e.market_context_window(records,day,self.days)[1])

    def test_market_raw_allowlist_rejects_future_label_columns_duplicate_and_nonofficial_days(self):
        record=self.market_record_fixture(self.days[8])
        record["raw"]["target"]=1
        with self.assertRaisesRegex(ValueError,"forecast/label"):
            e.market_context_window([record],self.days[8],self.days)
        record["raw"].pop("target")
        with self.assertRaisesRegex(ValueError,"duplicate"):
            e.market_context_window([record,record],self.days[8],self.days)
        record["date"]="2026-08-08"
        with self.assertRaisesRegex(ValueError,"unofficial"):
            e.market_context_window([record],self.days[8],self.days)

    def test_market_ex_self_requires_archived_count_and_height_match(self):
        pool=self.source_data["groups"]["pool"];r=self.market_record_fixture(self.days[8])
        value=e.market_ex_self(r,pool,"000001")["features"]
        self.assertEqual(value["market_limit_count_ex_self"],19.)
        self.assertEqual(value["market_consecutive_count_ex_self"],1.)
        self.assertEqual(value["market_height_ex_self"],2)
        r["raw"]["consecutive_count"]=3.
        self.assertTrue(all(v is None for v in e.market_ex_self(r,pool,"000001")["features"].values()))

    def test_rich_source_read_cache_keeps_upstream_hash_checks_and_revalidates_new_digest(self):
        upstream=self.root/"upstream.json";upstream.write_text("one",encoding="utf-8")
        self.source_data.update(derivation_kind="HISTORICAL_PEER_PRICE_PATH",source_bindings=e.bind(upstream),paths={},coverage={})
        self.source.write_text(json.dumps(self.source_data),encoding="utf-8")
        for cell in self.row["groups"].values():cell["source_sha256"]=e.sha(self.source)
        with patch.object(e,"_replay_peer_bundle",return_value=copy.deepcopy(self.source_data)) as replay:
            e._group(self.row,"price","DIAGNOSTIC");e._group(self.row,"price","DIAGNOSTIC")
            self.assertEqual(replay.call_count,1)
            upstream.write_text("two",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"source hash mismatch"):
                e._group(self.row,"price","DIAGNOSTIC")
            upstream.write_text("one",encoding="utf-8")
            changed=copy.deepcopy(self.source_data);changed["groups"]["price"]["price"]=11.
            self.source.write_text(json.dumps(changed),encoding="utf-8")
            self.row["groups"]["price"]["source_sha256"]=e.sha(self.source)
            with self.assertRaisesRegex(ValueError,"original response replay"):
                e._group(self.row,"price","DIAGNOSTIC")
            self.assertEqual(replay.call_count,2)

    def test_cycle_source_cannot_skip_official_sessions_or_end_before_signal(self):
        self.source_data["groups"]["cycle"]={"records":[{"date":d,"state":"REPAIR"} for d in
            ("2026-10-06","2026-10-08","2026-10-09")]}
        self.source.write_text(json.dumps(self.source_data),encoding="utf-8")
        for cell in self.row["groups"].values():cell["source_sha256"]=e.sha(self.source)
        self.row["groups"]["cycle"]=dict(self.row["groups"]["price"],pointer=["groups","cycle"],value=self.source_data["groups"]["cycle"])
        self.dataset["source_bindings"]=e.bind(self.source)
        actual=e.analyse(self.dataset,"DIAGNOSTIC")["rows"][0]
        self.assertFalse(actual["role_eligible"]["CYCLE"])
        self.assertIn("NONCONSECUTIVE_OR_STALE_CYCLE",actual["roles"]["CYCLE"]["missing"])

    def test_operation_hash_deduplication_rehashes_on_exit_and_does_not_survive_next_call(self):
        bound=e.bind(self.source)
        @e.verified_session
        def operation():
            e.verify(bound);e.verify(bound)
        with patch.object(e,"sha",wraps=e.sha) as hasher:
            operation();operation()
            self.assertEqual(hasher.call_count,4)

    def test_operation_final_hash_detects_changed_bytes_with_restored_file_metadata(self):
        import os
        bound=e.bind(self.source);stat=self.source.stat()
        @e.verified_session
        def operation():
            e.verify(bound)
            data=self.source.read_bytes();self.source.write_bytes(data.replace(b'10.0',b'11.0',1))
            os.utime(self.source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            e.verify(bound)
        with self.assertRaisesRegex(ValueError,"changed during verified"):
            operation()

    def peer_price_fixture(self, closes, code="000001", volume=True):
        bars = [{"minute": f"2026-10-09 09:{35+5*i:02d}:00", "open": c, "high": c,
                 "low": c, "close": c, "volume_shares": 100., "amount": 1000.} for i,c in enumerate(closes)]
        return {"signal_date": "2026-10-09", "code": code, "previous_close": 10.,
            "price_reconciled": True, "grid_complete": True, "volume_reconciled": volume,
            "amount_volume_unit_consistent": volume, "bars": bars, "missing_times": []}

    def test_peer_sampled_price_strength_excludes_self_and_keeps_volume_separate(self):
        paths = {"000001": self.peer_price_fixture([10., 11.]),
                 "000002": self.peer_price_fixture([10., 10.], "000002", volume=False)}
        actual = e.relative_sampled_prices("000001", ["000002"], paths, True)
        self.assertTrue(actual["complete_price_comparison"])
        self.assertAlmostEqual(actual["features"]["relative_last_pp"], 10.)
        self.assertEqual(actual["features"]["positive_bar_share"], .5)
        self.assertIsNone(actual["features"]["amount_share"])
        self.assertTrue(actual["not_fund_causality_or_absorption"])

    def test_missing_peer_or_incomplete_scope_never_becomes_relative_win(self):
        paths = {"000001": self.peer_price_fixture([10., 11.]),
                 "000002": self.peer_price_fixture([10., 10.], "000002")}
        missing = e.relative_sampled_prices("000001", ["000002", "000003"], paths, True)
        self.assertEqual(missing["missing_peer_codes"], ["000003"])
        self.assertEqual(missing["features"]["peer_coverage"], .5)
        self.assertIsNone(missing["features"]["relative_last_pp"])
        incomplete = e.relative_sampled_prices("000001", ["000002"], paths, False)
        self.assertIsNone(incomplete["features"]["relative_last_pp"])
        paths["000002"]["grid_complete"] = False
        self.assertIsNone(e.relative_sampled_prices("000001", ["000002"], paths, True)["features"]["relative_last_pp"])

    def test_peer_dates_previous_reference_and_duplicate_rivals_are_checked(self):
        paths = {"000001": self.peer_price_fixture([10., 11.]),
                 "000002": self.peer_price_fixture([10., 10.], "000002")}
        with self.assertRaises(ValueError): e.relative_sampled_prices("000001", ["000001"], paths, True)
        with self.assertRaises(ValueError): e.relative_sampled_prices("000001", ["000002", "000002"], paths, True)
        paths["000002"]["previous_close"] = None
        self.assertIsNone(e.relative_sampled_prices("000001", ["000002"], paths, True)["features"]["relative_last_pp"])
        paths["000002"]["previous_close"] = 10.
        for p in paths.values():
            for b in p["bars"]: b["minute"] = b["minute"].replace("2026-10-09", "2026-10-12")
        with self.assertRaisesRegex(ValueError, "cross signal dates"):
            e.relative_sampled_prices("000001", ["000002"], paths, True)

    def anonymous_peer_fixture(self):
        row = copy.deepcopy(self.row); row["groups"].pop("path")
        row["groups"].pop("theme")
        paths = {"2026-10-09:"+code: self.peer_price_fixture([10., 10.],code)
                 for code in ("000001", "000002")}
        local = {p["code"]:p for p in paths.values()}
        value = {"code": row["code"], "signal_date": row["signal_date"], "current_scope": "CURRENT_CLOSERS_ONLY",
            "prior_scope": "DECLARED_PRIOR_COMPETITORS", "current": e.relative_sampled_prices("000001",["000002"],local,True),
            "prior": e.relative_sampled_prices("000001",[],local,False), "prior_current_states": {"000002":"UNKNOWN"},
            "path_refs": {code: "2026-10-09:"+code for code in local}}
        source = self.root/"peer_source.json"
        e.write_new(source, {"derivation_kind":"HISTORICAL_PEER_PRICE_PATH", "paths": paths,
                            "groups": {row["candidate_id"]:{"sampled_peers":value}}})
        row["groups"]["sampled_peers"] = {"source_path":str(source),"source_sha256":e.sha(source),"value":value}
        row["groups"]["sampled_path"] = {"value":{"provider_reported_seal":{"bomb_count":"0"}}}
        data = dict(self.dataset, rows=[row])
        return data,source

    def test_anonymous_peer_packet_has_raw_bars_without_model_features_future_or_named_ids(self):
        data,source = self.anonymous_peer_fixture()
        data["rows"].append(copy.deepcopy(self.row))
        packet,mapping = r.anonymous_peer_packet(data,"b"*64)
        self.assertEqual(len(mapping),1)
        self.assertEqual(len(packet["raw_paths"]),2)
        text=json.dumps({k:v for k,v in packet.items() if k!="formed_at"})
        for forbidden in ("2026-10-09", '"000001"', '"000002"', '"features"', '"relative_last_pp"', str(source), '"signal_one_word"'):
            self.assertNotIn(forbidden,text)
        self.assertFalse(packet["learned_scores_or_role_features_exported"])
        fact=packet["rows"][0]
        self.assertIn("UNKNOWN",fact["groups"]["peer_raw_facts"]["prior_current_states"].values())

    def test_anonymous_peer_packet_rejects_future_bars_hidden_behind_signal_date(self):
        data,source=self.anonymous_peer_fixture();bundle=e.read(source)
        bundle["paths"]["2026-10-09:000001"]["bars"][0]["minute"]="2026-10-12 09:35:00"
        source.write_text(json.dumps(bundle),encoding="utf-8")
        data["rows"][0]["groups"]["sampled_peers"]["source_sha256"]=e.sha(source)
        with self.assertRaisesRegex(ValueError,"duplicate/future/wrong-date"):
            r.anonymous_peer_packet(data,"b"*64)

    def test_single_date_market_mask_hides_rule_stage_models_and_later_case_windows(self):
        data,_=self.anonymous_peer_fixture();signal="2026-10-09"
        records=[self.market_record_fixture(day) for day in ("2026-10-07","2026-10-08","2026-10-09")]
        context,cycle=e.market_context_window(records,signal,self.days)
        data["rows"][0]["groups"]["market_context"]={"value":context}
        data["rows"][0]["groups"]["cycle"]={"value":cycle}
        future=copy.deepcopy(data["rows"][0]);future.update(candidate_id="2026-10-12:000001",signal_date="2026-10-12",boards=999)
        future["groups"]["pool"]["value"]["boards"]["000001"]=999
        data["rows"].append(future)
        packet,mapping=r.anonymous_market_packet(data,"c"*64,signal)
        self.assertEqual(len(mapping),1)
        self.assertFalse(packet["inter_case_later_signal_windows_included"])
        self.assertEqual(max(r["session_offset"] for r in packet["market_observations"]),0)
        text=json.dumps({k:v for k,v in packet.items() if k!="formed_at"})
        for forbidden in (signal,"2026-10-12",'"state"','"rule_phase"','"features"',"999",'"raw_score"'):
            self.assertNotIn(forbidden,text)
        self.assertIsNone(packet["rows"][0]["past_ladder_observations"][0]["own_boards"])

    def test_partial_anonymous_peer_review_keeps_unreviewed_candidates_and_roles_unknown(self):
        data,_=self.anonymous_peer_fixture()
        other=copy.deepcopy(self.row)
        other.update(candidate_id="2026-10-09:000002",code="000002",boards=2)
        data["rows"].append(other)
        dataset_path=self.root/"full_peer_dataset.json";e.write_new(dataset_path,data)
        with patch.object(e,"load_dataset",return_value=data):
            manifest=r.prepare_anonymous_peer_review(dataset_path,self.root/"peer_review")
            packet=e.read(manifest["packet_path"]);alias=packet["rows"][0]["candidate_id"]
            receipts=[]
            for name in ("independent_a","independent_b"):
                moment=e.now()
                receipt={"reviewer_id":name,"packet_path":manifest["packet_path"],"packet_sha256":manifest["packet_sha256"],
                    "review_started_at":moment,"reviewed_at":e.now(),"model_masked":True,"future_masked":True,
                    "independent_of_prediction":True,"formal_blind_label_eligible":True,"case_identity_masked":True,
                    "rows":[{"candidate_id":alias,"role":role,"verdict":True if role=="HEIGHT" else None,
                             "basis":"Synthetic raw height3 versus2; other identity evidence unavailable",
                             "evidence_refs":[manifest["packet_path"]]} for role in e.ROLES]}
                path=self.root/(name+".json");e.write_new(path,receipt);receipts.append(path)
            output=self.root/"partial_peer_labels.json"
            bank=r.merge_identity_reviews(dataset_path,manifest["packet_path"],receipts,output,self.root/"peer_review/operator_mapping.json")
            self.assertEqual(len(bank["rows"]),22)
            labels=r.load_labels(output,data)
            self.assertTrue(labels[(self.row["candidate_id"],"HEIGHT")]["value"])
            self.assertTrue(all(c["value"] is None for (ident,role),c in labels.items() if ident==other["candidate_id"]))
            self.assertEqual(labels[(other["candidate_id"],"HEIGHT")]["unknown_reason"],"NOT_IN_CURRENT_RAW_PEER_REVIEW_PACKET")

    def sampled_fixture(self, closes, lows=None, complete=True, volume=True):
        bars = [{"minute": f"2026-10-09 09:{35+5*i:02d}:00", "open": c,
                 "high": 10., "low": (lows or closes)[i], "close": c,
                 "volume_shares": 100., "amount": c*100} for i, c in enumerate(closes)]
        check = {"date": "2026-10-09", "code": "000001", "interval_minutes": 5,
                 "price_reconciled": True, "observed_bars": len(bars), "grid_complete": complete,
                 "missing_times": [] if complete else ["09:50"], "volume_reconciled": volume,
                 "amount_volume_unit_consistent": volume}
        return bars, check

    def test_sampled_recovery_is_observed_price_sequence_not_seal_or_absorption(self):
        bars, check = self.sampled_fixture([10., 9.8, 10.])
        actual = e.sampled_price_path(bars, check, 10.)
        self.assertEqual(actual["features"]["sampled_later_limit_close_after_drop"], 1.)
        self.assertAlmostEqual(actual["features"]["sampled_after_touch_drawdown_pct"], -2.)
        self.assertIsNone(actual["actual_open_duration_minutes"])
        self.assertIsNone(actual["actual_reseal_verified"])
        self.assertFalse(actual["absorption_verified"])
        self.assertFalse(actual["order_book_seal_verified"])

    def test_sampled_within_bar_low_does_not_create_drop_then_reseal(self):
        bars, check = self.sampled_fixture([10., 10.], lows=[9.8, 10.])
        actual = e.sampled_price_path(bars, check, 10.)
        self.assertEqual(actual["features"]["sampled_close_drop_after_limit"], 0.)
        self.assertEqual(actual["features"]["sampled_later_limit_close_after_drop"], 0.)
        self.assertFalse(actual["actual_one_word_observed_OHLC"])

    def test_sampled_missing_grid_does_not_prove_negative_pattern_or_one_word(self):
        bars, check = self.sampled_fixture([10., 10.], complete=False)
        actual = e.sampled_price_path(bars, check, 10.)
        self.assertIsNone(actual["features"]["sampled_close_drop_after_limit"])
        self.assertIsNone(actual["features"]["sampled_later_limit_close_after_drop"])
        self.assertIsNone(actual["actual_one_word_observed_OHLC"])

    def test_sampled_volume_mismatch_is_not_volume_or_absorption_evidence(self):
        bars, check = self.sampled_fixture([10., 9.8, 10.], volume=False)
        actual = e.sampled_price_path(bars, check, 10.)
        self.assertIsNone(actual["observed_volume_shares"])
        self.assertIsNone(actual["volume_in_bars_closing_below_limit"])
        self.assertFalse(actual["absorption_verified"])

    def test_sampled_wrong_signal_and_cent_price_rejected(self):
        bars, check = self.sampled_fixture([10., 10.])
        with self.assertRaises(ValueError): e.sampled_price_path(bars, check, 9.99)
        with self.assertRaises(ValueError): e.sampled_price_path(bars, check, 10.001)
        bars[-1]["minute"] = "2026-10-12 09:40:00"
        with self.assertRaises(ValueError): e.sampled_price_path(bars, check, 10.)

    def test_sampled_group_never_unlocks_turnover_or_exact_duration(self):
        bars, check = self.sampled_fixture([10., 9.8, 10.])
        fact = e.sampled_price_path(bars, check, 10.)
        self.source_data["groups"]["sampled_path"] = fact
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values(): cell["source_sha256"] = e.sha(self.source)
        self.row["groups"].pop("path")
        self.row["groups"]["sampled_path"] = dict(self.row["groups"]["price"],
            pointer=["groups", "sampled_path"], value=fact)
        self.dataset["source_bindings"] = e.bind(self.source)
        actual = e.analyse(self.dataset, "DIAGNOSTIC")["rows"][0]
        self.assertFalse(actual["role_eligible"]["TURNOVER"])
        self.assertEqual(actual["roles"]["TURNOVER"]["state"], "UNKNOWN")
        self.assertIsNone(actual["features"]["actual_path_open_minutes"])
        self.assertEqual(actual["features"]["sampled_later_limit_close_after_drop"], 1.)

    def test_sampled_rebuilt_facts_reject_tampering_even_with_new_bundle_hash(self):
        original = copy.deepcopy(self.source_data)
        self.source_data["derivation_kind"] = "HISTORICAL_SAMPLED_PRICE_PATH"
        self.source_data["capture_metadata_bound"] = True
        self.source_data["groups"]["price"]["price"] = 11.
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        cell = self.row["groups"]["price"]
        cell.update(source_sha256=e.sha(self.source), value=self.source_data["groups"]["price"])
        with patch.object(e, "_replay_sampled_bundle", return_value=original):
            with self.assertRaisesRegex(ValueError, "original response replay"):
                e._group(self.row, "price", "DIAGNOSTIC")

    def test_reported_seal_requires_original_capture_date_hash_and_full_count(self):
        metadata = {"date": "20261009", "datasets": {"limit_up": {"sha256": "original", "rows": 10}}}
        e.validate_reported_capture(metadata, "2026-10-09", "original", 10)
        with self.assertRaisesRegex(ValueError, "date mismatch"):
            e.validate_reported_capture(metadata, "2026-10-08", "original", 10)
        with self.assertRaisesRegex(ValueError, "hash/count"):
            e.validate_reported_capture(metadata, "2026-10-09", "changed", 10)
        with self.assertRaisesRegex(ValueError, "hash/count"):
            e.validate_reported_capture(metadata, "2026-10-09", "original", 9)
        metadata["datasets"]["limit_up"].pop("sha256")
        self.assertFalse(e.validate_reported_capture(metadata, "2026-10-09", "v275_retrospective_hash", 10))

    def test_sampled_historical_reconstruction_remains_unavailable_in_strict_track(self):
        bars, check = self.sampled_fixture([10., 9.8, 10.])
        fact = e.sampled_price_path(bars, check, 10.)
        fact["provider_reported_seal"] = {"first_last_clock_gap_minutes": 60., "bomb_count": 1}
        self.source_data["groups"]["sampled_path"] = fact
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values(): cell["source_sha256"] = e.sha(self.source)
        self.row["groups"].pop("path")
        self.row["groups"]["sampled_path"] = dict(self.row["groups"]["price"],
            pointer=["groups", "sampled_path"], value=fact, reconstructed=True)
        self.dataset["source_bindings"] = e.bind(self.source)
        actual = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertIsNone(actual["features"]["sampled_later_limit_close_after_drop"])
        self.assertIsNone(actual["features"]["actual_path_open_minutes"])
        self.assertIsNone(actual["roles"]["TURNOVER"]["baseline_call"])

    def test_cli_dispatch_uses_importable_module_for_model_persistence(self):
        with patch.object(r, "main") as imported_entry:
            runpy.run_path(str(Path(r.__file__).resolve()), run_name="__main__")
        imported_entry.assert_called_once_with()

    def test_unadjusted_return_requires_exact_cent_quote_and_previous_official_day(self):
        bars = [{"date": "2026-10-08", "close": "10.00"}, {"date": "2026-10-09", "close": "11.00"},
                {"date": "2026-10-12", "close": "99.00"}]
        actual = e.unadjusted_quote_change("000001", "2026-10-09", 11., bars, self.days)
        self.assertEqual(actual["value"], 10.)
        self.assertEqual(actual["previous_session"], "2026-10-08")
        self.assertEqual(e.unadjusted_quote_change("000001", "2026-10-09", 11.01, bars, self.days)["reason"], "SIGNAL_CLOSE_MISMATCH")
        self.assertIsNone(e.unadjusted_quote_change("000001", "2026-10-09", 11.001, bars, self.days)["value"])
        self.assertIsNone(e.unadjusted_quote_change("000001", "2026-10-09", None, bars, self.days)["value"])
        self.assertIsNone(e.unadjusted_quote_change("000001", "2026-10-09", 11., bars[1:], self.days)["value"])
        self.assertEqual(e.unadjusted_quote_change("000001", "2026-10-09", 11., bars+[bars[0]], self.days)["reason"], "DUPLICATE_CURRENT_OR_PREVIOUS_BAR")

    def test_historical_relation_projection_preserves_original_inputs_and_replays_returns(self):
        raw = self.root/"september_unadjusted_20261002_v3/raw/sz000001.json"
        raw.parent.mkdir(parents=True)
        e.write_new(raw, {"code": 0, "data": {"sz000001": {"day": [
            ["2026-10-08", "9", "9", "9", "9", "100"],
            ["2026-10-09", "10", "10", "10", "10", "100"],
            ["2026-10-12", "99", "99", "99", "99", "100"]]}}})
        relation = copy.deepcopy(self.dataset)
        relation["rows"][0]["groups"]["theme"]["value"]["members"]["000001"]["return_pct"] = None
        # The source pointer must retain its own original facts.
        source_value = copy.deepcopy(self.source_data)
        source_value["groups"]["theme"]["members"]["000001"]["return_pct"] = None
        self.source.write_text(json.dumps(source_value), encoding="utf-8")
        for cell in relation["rows"][0]["groups"].values(): cell["source_sha256"] = e.sha(self.source)
        relation["source_bindings"] = e.bind(self.source)
        relation_path = self.root/"relations.json"; e.write_new(relation_path, relation)
        parent = copy.deepcopy(relation)
        parent["rows"][0]["groups"] = {kind: cell for kind, cell in parent["rows"][0]["groups"].items() if kind in ("pool", "price")}
        parent["source_bindings"] |= e.bind(raw)
        parent_path = self.root/"original_history.json"; e.write_new(parent_path, parent)
        result = e.relate_history(parent_path, relation_path, self.root/"rebuild")
        loaded = e.load_dataset(self.root/"rebuild/history_with_relations.json")
        old, new = parent["rows"][0], loaded["rows"][0]
        self.assertEqual(new["groups"]["pool"], old["groups"]["pool"])
        self.assertEqual(new["groups"]["price"], old["groups"]["price"])
        self.assertEqual(new["original_decision_at"], old["decision_at"])
        self.assertAlmostEqual(new["groups"]["theme"]["value"]["members"]["000001"]["return_pct"], 100/9)
        self.assertFalse(e.analyse(loaded, "STRICT")["rows"][0]["role_eligible"]["THEME"])
        source = self.root/"rebuild/source_rebuilt_relation_facts.json"
        tampered = e.read(source)
        tampered["groups"][new["candidate_id"]]["theme"]["members"]["000001"]["return_pct"] = 999.
        source.write_text(json.dumps(tampered), encoding="utf-8")
        changed = copy.deepcopy(new)
        changed["groups"]["theme"].update(source_sha256=e.sha(source), value=tampered["groups"][new["candidate_id"]]["theme"])
        with self.assertRaisesRegex(ValueError, "original source replay"):
            e._group(changed, "theme", "DIAGNOSTIC")

    def test_saved_first_board_quotes_units_ties_and_no_target_fields(self):
        roster = [{"code": "000001", "boards": 1, "price_cents": 1234, "amount": 100,
                   "turnover_pct": 2.5, "target_promoted": True},
                  {"code": "000002", "boards": 1, "price_cents": 1234, "amount": 300,
                   "turnover_pct": 7.5}]
        facts = e.closing_price_facts(roster, {})
        self.assertEqual(facts["000001"]["price"], 12.34)
        self.assertEqual(facts["000001"]["turnover"], .025)
        self.assertEqual(facts["000001"]["signal_amount"], 100)
        self.assertEqual(facts["000001"]["price_same_height_percentile"], .75)
        self.assertEqual(facts["000001"]["signal_amount_same_height_percentile"], .5)
        self.assertEqual(set(facts["000001"]), set(e.RELAY_FIELDS))
        for key in ("signal_one_word", "turnover_vs_prev", "signal_amplitude"):
            self.assertIsNone(facts["000001"].get(key))

    def test_first_board_projection_preserves_original_relay_features(self):
        original = {key: float(i) for i, key in enumerate(e.RELAY_FIELDS)}
        relay = {"000001": {"features": original}}
        roster = [{"code": "000001", "boards": 3, "price_cents": 99999, "amount": 100,
                   "turnover_pct": 99},
                  {"code": "000002", "boards": 1, "price_cents": 1000, "amount": 10,
                   "turnover_pct": 2}]
        before = copy.deepcopy(relay)
        facts = e.closing_price_facts(roster, relay)
        self.assertEqual(facts["000001"], original)
        self.assertEqual(relay, before)
        self.assertEqual(facts["000002"]["height_gap"], 2)
        self.assertEqual(facts["000002"]["same_height_rivals"], 0)
        self.assertEqual(facts["000002"]["is_space_leader"], 0)

    def test_invalid_quotes_stay_unknown_and_incomplete_ranks_not_imputed(self):
        roster = [{"code": "000001", "boards": 1, "price_cents": 0, "amount": -1,
                   "turnover_pct": True},
                  {"code": "000002", "boards": 1, "price_cents": 1000, "amount": 10,
                   "turnover_pct": 2}]
        facts = e.closing_price_facts(roster, {})
        for key in ("price", "signal_amount", "turnover", "log_signal_amount"):
            self.assertIsNone(facts["000001"][key])
        for key in ("price_same_height_percentile", "signal_amount_same_height_percentile", "turnover_same_height_percentile"):
            self.assertIsNone(facts["000002"][key])
        with self.assertRaisesRegex(ValueError, "nonduplicated"):
            e.closing_price_facts([roster[0], roster[0]], {})

    def test_lunch_excluded_and_proxy_separate(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(row["features"]["actual_path_open_minutes"], 10.)
        self.assertEqual(row["actual_one_word_daily"], 0.)
        self.assertEqual(row["low_turnover_proxy"], 1.)

    def test_missing_path_not_volume_confirmation(self):
        del self.row["groups"]["path"]
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(row["roles"]["TURNOVER"]["state"], "UNKNOWN")

    def test_late_strict_but_diagnostic_retained(self):
        self.row["decision_at"] = "2026-10-09T16:30:00+08:00"
        self.assertFalse(e.analyse(self.dataset, "STRICT")["rows"][0]["role_eligible"]["HEIGHT"])
        self.assertTrue(e.analyse(self.dataset, "DIAGNOSTIC")["rows"][0]["role_eligible"]["HEIGHT"])

    def test_source_tamper_rejected(self):
        self.source.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            e.analyse(self.dataset, "STRICT")

    def test_projection_tamper_rejected(self):
        cell = copy.deepcopy(self.row["groups"]["pool"])
        cell["value"]["boards"]["000001"] = 8
        self.row["groups"]["pool"] = cell
        with self.assertRaisesRegex(ValueError, "differs"):
            e.analyse(self.dataset, "STRICT")

    def test_normalized_fact_bundle_checks_bound_upstream_sources(self):
        self.source_data["source_bindings"] = {str(self.root/"missing_raw.json"): "a"*64}
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values():
            cell["source_sha256"] = e.sha(self.source)
        self.dataset["source_bindings"] = e.bind(self.source)
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            e.analyse(self.dataset, "DIAGNOSTIC")

    def test_wrong_group_date_rejected(self):
        self.row["signal_date"] = "2026-10-08"
        self.row["candidate_id"] = "2026-10-08:000001"
        with self.assertRaises(ValueError):
            e.analyse(self.dataset, "STRICT")

    def test_fake_trusted_adapter_rejected(self):
        self.row["groups"]["pool"]["adapter"] = "SAVED_BASE_ALLOWLIST"
        p = self.root/"dataset.json"; e.write_new(p, self.dataset)
        with self.assertRaisesRegex(ValueError, "trusted"):
            e.load_dataset(p)

    def test_label_unknowns_preserved(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p)
        l = self.root/"labels.json"; e.write_new(l, labels)
        loaded = r.load_labels(l, self.dataset)
        self.assertEqual(len(loaded), 11)
        self.assertTrue(all(v["value"] is None for v in loaded.values()))

    def test_masked_packet_cannot_link_model_or_future_source_archives(self):
        data = copy.deepcopy(self.dataset)
        data["source_bindings"]["operator_only_future_labels.json"] = "a"*64
        data["source_bindings"]["operator_only_model_scores.json"] = "b"*64
        before = copy.deepcopy(data)
        packet = r.masked_packet(data)
        self.assertEqual(packet["source_bindings"], {})
        self.assertNotIn("operator_only_future", json.dumps(packet))
        for group in packet["rows"][0]["groups"].values():
            self.assertNotIn("source_path", group)
            self.assertNotIn("pointer", group)
        self.assertEqual(data, before)
        self.assertIn("source_path", data["rows"][0]["groups"]["pool"])

    def test_matching_reviews_reference_reconstructed_masked_facts(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p)
        cell = labels["rows"][0]; moment = e.now()
        cell.update(value=True, available_at=moment, reviews=[{
            "reviewer_id": name, "verdict": True, "model_masked": True,
            "future_masked": True, "independent_of_prediction": True,
            "reviewed_at": moment, "evidence_refs": [str(p.resolve())],
            "basis": "Complete scoped roster has height3 above height2."}
            for name in ("independent_a", "independent_b")])
        labels["formed_at"] = e.now()
        good = self.root/"reviewed.json"; e.write_new(good, labels)
        self.assertTrue(r.load_labels(good, self.dataset)[(self.row["candidate_id"], "HEIGHT")]["value"])
        cell["reviews"][0]["evidence_refs"] = [str(self.source)]
        bad = self.root/"unmasked_ref.json"; e.write_new(bad, labels)
        with self.assertRaisesRegex(ValueError, "masked fact"):
            r.load_labels(bad, self.dataset)

    def test_actual_review_import_preserves_context_exposure_and_disagreement(self):
        dataset = self.root/"dataset.json"; e.write_new(dataset, self.dataset)
        packet = self.root/"packet.json"; e.write_new(packet, r.masked_packet(self.dataset))
        paths = []
        for name, height in (("a", True), ("b", True)):
            moment = e.now()
            receipt = {"reviewer_id": name, "packet_path": packet.resolve().as_posix(), "packet_sha256": e.sha(packet),
                "review_started_at": moment, "reviewed_at": moment, "model_masked": True, "future_masked": True,
                "independent_of_prediction": True, "rows": [{"candidate_id": self.row["candidate_id"],
                    "role": role, "verdict": height if role == "HEIGHT" else None,
                    "basis": "synthetic fixture scoped height evidence or missing required role facts",
                    "evidence_refs": [packet.resolve().as_posix()]} for role in e.ROLES]}
            path = self.root/(name+".json"); e.write_new(path, receipt); paths.append(path)
        good = r.merge_identity_reviews(dataset, packet, paths, self.root/"known.json")
        self.assertTrue(good["rows"][0]["value"])
        exposed = e.read(paths[1]); exposed["future_masked"] = False
        exposed_path = self.root/"exposed.json"; e.write_new(exposed_path, exposed)
        unqualified = r.merge_identity_reviews(dataset, packet, [paths[0], exposed_path], self.root/"unqualified.json")
        self.assertIsNone(unqualified["rows"][0]["value"])
        self.assertEqual(unqualified["rows"][0]["unknown_reason"], "REVIEW_CONTEXT_NOT_BLINDED")
        disagreement = e.read(paths[1]); disagreement["rows"][0]["verdict"] = False
        different = self.root/"disagreed.json"; e.write_new(different, disagreement)
        disputed = r.merge_identity_reviews(dataset, packet, [paths[0], different], self.root/"disputed.json")
        self.assertIsNone(disputed["rows"][0]["value"])
        self.assertEqual(disputed["rows"][0]["unknown_reason"], "REVIEW_DISAGREEMENT")
        with self.assertRaisesRegex(ValueError, "different actual"):
            r.merge_identity_reviews(dataset, packet, [paths[0], paths[0]], self.root/"duplicate.json")

    def test_label_cannot_omit_failed_or_unknown(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p); labels["rows"].pop()
        l = self.root/"labels.json"; e.write_new(l, labels)
        with self.assertRaisesRegex(ValueError, "omits"):
            r.load_labels(l, self.dataset)

    def test_blind_review_raw_quotes_exclude_role_shortcuts_and_model_features(self):
        self.row["groups"]["price"]["value"].update(is_space_leader=1., signal_one_word=1.,
            emotion_score=.8, height_gap=0., signal_amount=123.)
        packet = r.masked_packet(self.dataset)
        quote = packet["rows"][0]["groups"]["price"]["value"]
        self.assertEqual(set(quote), {"price", "turnover", "signal_amount"})
        self.assertEqual(packet["rows"][0]["actual_one_word_daily"], 0.)
        self.assertEqual(packet["mask_schema"], "RAW_FACTS_NO_ROLE_SHORTCUTS_V3")

    def test_raw_multiday_review_cannot_be_blind_and_day_packets_preserve_whole_dates(self):
        data = self._height_sequence_fixture()
        path = self.root/"all_days.json"; e.write_new(path, data)
        packet = r.masked_packet(data)
        self.assertEqual(packet["review_scope"], "MULTI_DATE_OPERATOR_ARCHIVE_NOT_BLIND_REVIEW")
        attest = {"model_masked": True, "future_masked": True, "independent_of_prediction": True}
        self.assertFalse(any(r.review_context_eligibility(attest,
            {row["candidate_id"] for row in data["rows"]}, "fixture_hash").values()))
        days = r.prepare_day_reviews(path, self.root/"daily")
        self.assertEqual(days["candidate_count"], 6)
        for entry in days["entries"]:
            raw = e.read(entry["packet_path"])
            self.assertEqual({row["signal_date"] for row in raw["rows"]}, {entry["signal_date"]})
            self.assertEqual(e.load_dataset(entry["dataset_path"])["rows"],
                [row for row in data["rows"] if row["signal_date"] == entry["signal_date"]])

    def test_context_scope_preserves_unrelated_exposure_but_excludes_overlap_and_unknown(self):
        candidates = {"2024-01-02:000001", "2024-01-02:000002"}
        receipt = {"model_masked": True, "future_masked": True, "independent_of_prediction": True,
            "inherited_AGENTS_contains_mature_outcome_summary": True,
            "context_exposure_scope": {"inventory_complete": True, "unknown_or_undated_exposure": False,
                "packet_sha256": "fixture", "exposed_candidate_ids": [], "basis": "Synthetic disclosed context",
                "exposed_periods": [{"start": "2026-10-09", "end": "2026-10-12", "kind": "FUTURE_OUTCOME", "basis": "Synthetic2026only"}]}}
        self.assertTrue(all(r.review_context_eligibility(receipt, candidates, "fixture").values()))
        receipt["context_exposure_scope"]["exposed_candidate_ids"] = ["2024-01-02:000001"]
        eligibility = r.review_context_eligibility(receipt, candidates, "fixture")
        self.assertFalse(eligibility["2024-01-02:000001"])
        self.assertTrue(eligibility["2024-01-02:000002"])
        receipt["context_exposure_scope"]["unknown_or_undated_exposure"] = True
        self.assertFalse(any(r.review_context_eligibility(receipt, candidates, "fixture").values()))

    def test_anonymous_height_labels_require_raw_source_mapping_and_actual_independent_reviews(self):
        data = self._height_sequence_fixture()
        path = self.root/"original_history.json"; e.write_new(path, data)
        manifest = r.prepare_anonymous_height_review(path, self.root/"anonymous")
        mapping_path = self.root/"anonymous/operator_mapping.json"
        packet = e.read(manifest["packet_path"])
        raw_text = json.dumps(packet["rows"])
        self.assertNotIn("2026-10-09", raw_text)
        self.assertNotIn('"000001"', raw_text)
        self.assertEqual(len({row["code"] for row in packet["rows"]}), 6)
        paths = []
        for name in ("anon_a", "anon_b"):
            moment = e.now()
            receipt = {"reviewer_id": name, "packet_path": manifest["packet_path"],
                "packet_sha256": manifest["packet_sha256"], "review_started_at": moment, "reviewed_at": moment,
                "model_masked": True, "future_masked": True, "independent_of_prediction": True,
                "case_identity_masked": True, "rows": []}
            for row in packet["rows"]:
                maximum = max(row["groups"]["pool"]["value"]["boards"].values())
                receipt["rows"].extend({"candidate_id": row["candidate_id"], "role": role,
                    "verdict": row["boards"] == maximum if role == "HEIGHT" else None,
                    "basis": "Synthetic independent raw height roster or required role facts absent",
                    "evidence_refs": [manifest["packet_path"]]} for role in e.ROLES)
            reviewer_path = self.root/(name+".json"); e.write_new(reviewer_path, receipt); paths.append(reviewer_path)
        output = self.root/"actual_anon_labels.json"
        bank = r.merge_identity_reviews(path, manifest["packet_path"], paths, output, mapping_path)
        known = r.load_labels(output, data)
        self.assertTrue(all(known[(row["candidate_id"], "HEIGHT")]["value"] is not None for row in data["rows"]))
        self.assertTrue(all(cell["value"] is None for key, cell in known.items() if key[1] != "HEIGHT"))
        altered = copy.deepcopy(bank); altered["rows"][0]["reviews"][0]["basis"] = "changed assertion"
        changed_path = self.root/"altered_review.json"; e.write_new(changed_path, altered)
        with self.assertRaisesRegex(ValueError, "actual reviewer evidence"):
            r.load_labels(changed_path, data)
        with self.assertRaisesRegex(ValueError, "rich role evidence"):
            r.anonymous_height_packet(self.dataset, "a"*64)

    def _original_reviewed_bank(self):
        dataset = self.root/"original_dataset.json"; e.write_new(dataset, self.dataset)
        packet = self.root/"original_packet.json"; e.write_new(packet, r.masked_packet(self.dataset))
        bank = r.label_template(self.dataset, packet)
        moment = e.now()
        bank["rows"][0].update(value=True, available_at=moment, reviews=[{
            "reviewer_id": name, "verdict": True, "model_masked": True, "future_masked": True,
            "independent_of_prediction": True, "reviewed_at": moment, "evidence_refs": [str(packet.resolve())],
            "basis": "Synthetic fixture complete scoped height3 versus2."} for name in ("a", "b")])
        bank["formed_at"] = e.now()
        labels = self.root/"original_labels.json"; e.write_new(labels, bank)
        return dataset, labels, bank

    def test_original_daily_labels_aggregate_without_redating_review_or_evidence(self):
        dataset, labels, original = self._original_reviewed_bank()
        combined = copy.deepcopy(self.dataset)
        other = copy.deepcopy(self.row); other.update(code="000002", candidate_id="2026-10-09:000002", boards=2)
        combined["rows"].append(other)
        combined_path = self.root/"combined.json"; e.write_new(combined_path, combined)
        output = self.root/"aggregate.json"
        bank = r.aggregate_labels(combined_path, [(dataset, labels)], output)
        loaded = r.load_labels(output, combined)
        cell = loaded[(self.row["candidate_id"], "HEIGHT")]
        self.assertTrue(cell["value"])
        self.assertEqual(cell["reviews"], original["rows"][0]["reviews"])
        self.assertEqual(cell["available_at"], original["rows"][0]["available_at"])
        self.assertLess(e.stamp(cell["reviews"][0]["reviewed_at"]), e.stamp(e.read(bank["packet_path"])["formed_at"]))
        self.assertIsNone(loaded[(other["candidate_id"], "HEIGHT")]["value"])
        for key, value in (("value", False), ("available_at", bank["formed_at"])):
            tampered = copy.deepcopy(bank); tampered["rows"][0][key] = value
            path = self.root/(key+"_changed.json"); e.write_new(path, tampered)
            with self.assertRaisesRegex(ValueError, "verdict or timestamp changed"):
                r.load_labels(path, combined)

    def test_label_origins_cannot_change_facts_hashes_or_choose_duplicate_verdicts(self):
        dataset, labels, original = self._original_reviewed_bank()
        with self.assertRaisesRegex(ValueError, "duplicate label origins"):
            r.aggregate_labels(dataset, [(dataset, labels), (dataset, labels)], self.root/"duplicated.json")
        output = self.root/"aggregate.json"; r.aggregate_labels(dataset, [(dataset, labels)], output)
        bank = e.read(output)
        bank["rows"][0]["label_origin"]["labels_sha256"] = "0"*64
        forged = self.root/"forged.json"; e.write_new(forged, bank)
        with self.assertRaisesRegex(ValueError, "source bindings"):
            r.load_labels(forged, self.dataset)
        changed = copy.deepcopy(self.dataset); changed["rows"][0]["decision_at"] = "2026-10-11T11:00:00+08:00"
        changed_path = self.root/"changed_dataset.json"; e.write_new(changed_path, changed)
        with self.assertRaisesRegex(ValueError, "candidate facts changed"):
            r.aggregate_labels(changed_path, [(dataset, labels)], self.root/"changed_origin.json")

    def test_frozen_warmup_labels_preserve_verdict_unknowns_reviews_and_availability(self):
        key = ("fixture", "HEIGHT")
        original = {key: {"value": None, "available_at": e.now(), "reviews": []}}
        current = copy.deepcopy(original); current[key]["label_origin"] = {"source": "fixture"}
        r._assert_warmup_labels_preserved(current, original)
        current[key]["value"] = True
        with self.assertRaisesRegex(ValueError, "warmup label"):
            r._assert_warmup_labels_preserved(current, original)

    def test_complete_date_slices_rebuild_parent_and_reject_dropped_candidates(self):
        data = self._height_sequence_fixture()
        parent = self.root/"all_dates.json"; e.write_new(parent, data)
        output = self.root/"slice.json"
        subset = e.slice_dates(parent, data["rows"][0]["signal_date"], data["rows"][-1]["signal_date"], output)
        self.assertEqual(e.load_dataset(output)["rows"], data["rows"])
        subset["rows"].pop()
        changed = self.root/"subset_changed.json"; e.write_new(changed, subset)
        with self.assertRaisesRegex(ValueError, "original candidates"):
            e.load_dataset(changed)

    def test_known_label_requires_independent_review(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p); labels["rows"][0]["value"] = True
        l = self.root/"labels.json"; e.write_new(l, labels)
        with self.assertRaisesRegex(ValueError, "independent"):
            r.load_labels(l, self.dataset)

    def test_role_metrics_abstention_and_unknown_truth(self):
        m = r.binary_metrics({"a": None, "b": False, "c": True}, {"a": True, "b": False, "c": None})
        self.assertEqual(m["registered"], 3)
        self.assertEqual(m["unknown_labels"], 1)
        self.assertEqual(m["complete_positive_Recall"], 0.)
        self.assertEqual(m["positive_abstentions"], 1)
        self.assertIsNone(m["Brier"])

    def diagnostic_record(self):
        return {"candidate_id": self.row["candidate_id"], "signal_date": self.row["signal_date"],
            "code": self.row["code"], "track": "DIAGNOSTIC", "fold": "fixture",
            "role_truth": dict.fromkeys(e.ROLES),
            "C1": {"calls": dict.fromkeys(e.ROLES), "scores": dict.fromkeys(e.ROLES)},
            "role_input_eligible": dict.fromkeys(e.ROLES, False),
            "missing_role_inputs": {role: ["MISSING_SOURCE"] for role in e.ROLES},
            "BASE_promotion": False, "C2_promotion": False, "promotion_truth": True}

    def append_fixture_group(self, kind, value):
        self.source_data["groups"][kind] = value
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values():
            cell["source_sha256"] = e.sha(self.source)
        self.row["groups"][kind] = {"value": value, "status": "VERIFIED", "source_path": str(self.source),
            "source_sha256": e.sha(self.source), "pointer": ["groups", kind],
            "observed_at": "2026-10-09T15:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        self.dataset["source_bindings"] = e.bind(self.source)

    def test_cross_cycle_requires_actual_pre_and_post_pressure_observations(self):
        value = {"event_id": "fixture", "pre_event_role": "THEME", "started_at": "2026-10-08T10:00:00+08:00",
            "ended_at": "2026-10-09T10:00:00+08:00", "post_event_competition_verified": True,
            "pre_cycle_id": "old", "post_cycle_id": "new"}
        self.append_fixture_group("pressure", value)
        weak = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertIn("INCOMPLETE_PRESSURE", weak["roles"]["CROSS_CYCLE"]["missing"])
        self.assertIsNone(weak["features"]["pressure_survived"])
        value.update(pre_event_observation={"code": "000001", "state": "CORE", "relative_rank": 3,
            "price": 10., "observed_at": "2026-10-08T09:35:00+08:00"},
            post_event_observation={"code": "000001", "state": "CORE", "relative_rank": 1,
            "price": 11., "observed_at": "2026-10-09T15:00:00+08:00"})
        value["post_competition"] = {"complete_within_scope": True, "peer_codes": ["000002", "000003"],
            "prior_at": "2026-10-08T09:35:00+08:00", "current_at": "2026-10-09T15:00:00+08:00",
            "prior": {"000001": "STRONG", "000002": "STRONG", "000003": "STRONG"},
            "current": {"000001": "STRONG", "000002": "FAILED", "000003": "COMPETING"}}
        self.append_fixture_group("pressure", value)
        facts = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(facts["features"]["pressure_rank_improvement"], 2)
        self.assertAlmostEqual(facts["features"]["pressure_price_change_pct"], 10.)
        self.assertEqual(facts["features"]["pressure_cycle_changed"], 1.)
        self.assertFalse(facts["role_identity_automatically_certified"])
        value["post_competition"]["current"].pop("000003")
        self.append_fixture_group("pressure", value)
        incomplete = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertIsNone(incomplete["features"]["pressure_survived"])

    def test_supplement_divergence_can_precede_anchor_break_and_future_break_is_rejected(self):
        value = {"anchor_code": "000002", "same_specific_theme": True, "anchor_identity_source": "synthetic_prior_review",
            "anchor_stable": True, "anchor_start": "2026-10-06", "own_start": "2026-10-08",
            "anchor_stage": "DIVERGENCE", "anchor_break_day": "2026-10-09"}
        self.append_fixture_group("dependency", value)
        facts = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertTrue(facts["role_eligible"]["SUPPLEMENT"])
        self.assertEqual(facts["features"]["dependency_anchor_is_divergence"], 1.)
        self.assertEqual(facts["features"]["dependency_start_after_anchor_break_sessions"], -1)
        self.assertEqual(facts["roles"]["SUPPLEMENT"]["state"], "CANDIDATE")
        value["anchor_break_day"] = "2026-10-12"
        self.append_fixture_group("dependency", value)
        with self.assertRaisesRegex(ValueError, "dependency date follows signal"):
            e.analyse(self.dataset, "DIAGNOSTIC")

    def test_promotion_miss_cannot_be_rewritten_as_role_false_negative(self):
        case = self.diagnostic_record()
        d = r.role_error_diagnosis([case])
        self.assertEqual(d["role_cells"], 11)
        self.assertEqual(d["cases"][0]["promotion_evaluation"]["BASE"], "FALSE_NEGATIVE")
        self.assertTrue(all(v["evaluation"] == "TRUTH_UNKNOWN" for v in d["cases"][0]["roles"].values()))

    def test_saved_role_diagnosis_rejects_changed_prediction_matrix(self):
        protocol = self.root/"protocol.json"; e.write_new(protocol, {"source_bindings": {}})
        matrix = self.root/"full_candidate_comparison.json"; e.write_new(matrix, [self.diagnostic_record()])
        report = self.root/"report.json"
        e.write_new(report, {"protocol_sha256": e.sha(protocol), "predictions_sha256": e.sha(matrix),
            "tracks": {"DIAGNOSTIC": {"folds": {"fixture": {"C1_fit_logs": {}}}}}})
        result = r.diagnose_saved_comparison(report, protocol, self.root/"diagnosis.json")
        self.assertEqual(result["candidate_records"], 1)
        self.assertFalse(result["deployment_allowed"])
        matrix.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            r.diagnose_saved_comparison(report, protocol, self.root/"tampered.json")

    def test_role_diagnosis_preserves_multirole_abstention_and_missing_inputs(self):
        case = self.diagnostic_record()
        case["role_truth"].update(HEIGHT=True, SUPPLEMENT=True, CROSS_CYCLE=False)
        case["C1"]["calls"].update(HEIGHT=True, CROSS_CYCLE=True)
        case["role_input_eligible"]["HEIGHT"] = True
        d = r.role_error_diagnosis([case], {("DIAGNOSTIC", "fixture"):
            {"SUPPLEMENT": {"status": "INSUFFICIENT_KNOWN_CLASSES", "attempted": False}}})
        roles = d["cases"][0]["roles"]
        self.assertEqual(roles["HEIGHT"]["evaluation"], "TRUE_POSITIVE")
        self.assertEqual(roles["SUPPLEMENT"]["evaluation"], "POSITIVE_ABSTENTION")
        self.assertIn("REQUIRED_ROLE_INPUTS_UNAVAILABLE", roles["SUPPLEMENT"]["reasons"])
        self.assertIn("ROLE_HEAD_NOT_FITTED:INSUFFICIENT_KNOWN_CLASSES", roles["SUPPLEMENT"]["reasons"])
        self.assertEqual(roles["CROSS_CYCLE"]["evaluation"], "FALSE_POSITIVE")
        self.assertIn("ROLE_CALL_WITHOUT_REQUIRED_INPUTS", roles["CROSS_CYCLE"]["reasons"])

    def test_training_member_role_score_rejected(self):
        model = r.RoleModel(); model.training_ids["HEIGHT"] = [self.row["candidate_id"]]
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        with self.assertRaisesRegex(ValueError, "training member"):
            model.predict([row], reject_training_members=True)

    def test_frozen_role_score_explanation_reconstructs_actual_logit(self):
        x = np.zeros((40, len(e.FEATURES)))
        index = e.FEATURES.index("price")
        x[:20, index] = -1.; x[20:, index] = 1.
        pipeline = r._pipeline().fit(x, np.asarray([0]*20+[1]*20))
        model = r.RoleModel(); model.models["HEIGHT"] = pipeline
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        result = model.predict([row])[row["candidate_id"]]
        explanation = result["explanations"]["HEIGHT"]
        self.assertAlmostEqual(explanation["intercept"]+explanation["sum_feature_contributions"],
                               float(pipeline.decision_function(r._matrix([row]))[0]), places=10)
        self.assertTrue(explanation["not_causal_effect"])
        self.assertTrue(explanation["not_calibrated_probability"])
        self.assertEqual(explanation["top_factors"][0]["feature"], "price")
        self.assertIsNone(result["explanations"]["SUPPLEMENT"])

    def test_no_role_labels_means_no_fit(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        labels = {(row["candidate_id"], role): {"value": None, "available_at": e.now()} for role in e.ROLES}
        m = r.RoleModel().fit([row], labels, "2026-10-12T00:00:00+08:00", self.days, "DIAGNOSTIC")
        self.assertFalse(m.models)
        self.assertTrue(all(not log["attempted"] for log in m.logs.values()))

    def future_date_fixture(self):
        today = e.stamp(e.now()).astimezone(r.ZoneInfo("Asia/Shanghai")).date()
        days = [(today+timedelta(days=i)).isoformat() for i in range(90) if (today+timedelta(days=i)).weekday()<5]
        future = [day for day in days if day>today.isoformat()]
        folds = [{"id": "development_"+str(i), "start": future[offset], "end": future[offset+1]}
                 for i,offset in enumerate((0,7,14))]
        return days, folds

    def test_strict_future_dates_are_separate_and_preserve_original_gates(self):
        days, folds = self.future_date_fixture()
        r._validate_future_folds(folds, days, e.now())
        protocol = {"strict_folds": folds}
        self.assertEqual(r.validation_folds(protocol,"DIAGNOSTIC"),list(r.FOLDS))
        self.assertEqual(r.validation_folds(protocol,"STRICT"),folds)
        result = {"tracks": {"STRICT": {"folds": {f["id"]: {
            "historical_screen_passed": True, "historical_C2_screen_passed": True} for f in folds}}}}
        self.assertTrue(r.strict_comparison_passed(protocol,result))
        result["tracks"]["STRICT"]["folds"].pop(folds[0]["id"])
        self.assertFalse(r.strict_comparison_passed(protocol,result))
        self.assertFalse(r.strict_comparison_passed(protocol,{"tracks":{"STRICT":{"folds":{}}}}))
        self.assertEqual(r.GATE["sessions"],120)
        self.assertEqual(r.GATE["min_complete_days_per_role"],60)

    def test_strict_date_plan_rejects_past_overlap_and_unverified_calendar(self):
        days, folds = self.future_date_fixture()
        bad=copy.deepcopy(folds);bad[0]["start"]=days[0]
        with self.assertRaisesRegex(ValueError,"precede"):
            r._validate_future_folds(bad,days,e.now())
        bad=copy.deepcopy(folds);bad[1]["start"]=folds[0]["end"]
        with self.assertRaisesRegex(ValueError,"embargo"):
            r._validate_future_folds(bad,days,e.now())
        bad=copy.deepcopy(folds);bad[-1]["end"]="2099-01-01"
        with self.assertRaisesRegex(ValueError,"official calendar"):
            r._validate_future_folds(bad,days,e.now())

    def test_future_strict_plan_cannot_freeze_unknown_or_late_training(self):
        days, folds = self.future_date_fixture()
        # Keep the original synthetic historical days and append official-grid fixture dates.
        calendar_path=self.root/"extended_calendar.json"
        e.write_new(calendar_path,{"status":"verified","source":"synthetic_fixture_only",
            "sessions":sorted(set(self.days+days))})
        data=copy.deepcopy(self.dataset);data["calendar_path"]=str(calendar_path);data["calendar_sha256"]=e.sha(calendar_path)
        data_path=self.root/"data.json";e.write_new(data_path,data)
        packet=self.root/"fact_packet.json";e.write_new(packet,r.masked_packet(data))
        labels=self.root/"unknown_labels.json";e.write_new(labels,r.label_template(data,packet))
        spec=self.root/"dates.json";e.write_new(spec,{"folds":folds})
        plan=self.root/"unqualified_plan.json"
        result=r.plan_strict(data_path,labels,spec,plan)
        self.assertEqual(result["status"],"NOT_FREEZABLE_TRAINING_QUALIFICATION_MISSING")
        self.assertEqual(result["fits"],0)
        self.assertTrue(all(not c["qualified"] for c in result["training_support"].values()))
        with self.assertRaisesRegex(ValueError,"qualified frozen"):
            r.load_strict_plan(plan)
        result["status"]="FROZEN_FUTURE_STRICT_DEVELOPMENT"
        forged=self.root/"forged_plan.json";e.write_new(forged,result)
        with self.assertRaisesRegex(ValueError,"flags alone"):
            r.load_strict_plan(forged)

    def test_diagnostic_lags_allow_late_reconstruction_but_not_changed_theme_denominator(self):
        self.row["decision_at"] = "2026-10-09T16:30:00+08:00"
        self.source_data["groups"]["theme"]["comparison_scope_id"] = "fixed_fixture_members"
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values(): cell["source_sha256"] = e.sha(self.source)
        previous_data = copy.deepcopy(self.source_data)
        previous_data["groups"].pop("path")
        previous_data["groups"]["price"]["price"] = 8.
        previous_data["groups"]["theme"]["members"]["000001"]["amount"] = 50.
        previous_source = self.root/"previous.json"; e.write_new(previous_source, previous_data)
        previous = copy.deepcopy(self.row)
        previous.update(signal_date="2026-10-08", candidate_id="2026-10-08:000001", decision_at="2026-10-08T16:30:00+08:00")
        previous["groups"].pop("path")
        for kind, cell in previous["groups"].items():
            cell.update(source_path=str(previous_source), source_sha256=e.sha(previous_source),
                        value=previous_data["groups"][kind], observed_at="2026-10-08T15:00:00+08:00")
        data = copy.deepcopy(self.dataset)
        data["rows"] = [previous, self.row]
        data["source_bindings"] = e.bind(self.source) | e.bind(previous_source)
        current = e.analyse(data, "DIAGNOSTIC")["rows"][1]
        self.assertEqual(current["features"]["price_delta_1"], 2.)
        self.assertAlmostEqual(current["features"]["theme_amount_share_delta_1"], 1/6)
        self.assertIsNone(e.analyse(data, "STRICT")["rows"][1]["features"]["price_delta_1"])
        previous_data["groups"]["theme"]["comparison_scope_id"] = "different_members"
        previous_source.write_text(json.dumps(previous_data), encoding="utf-8")
        for kind, cell in previous["groups"].items():
            cell.update(source_sha256=e.sha(previous_source), value=previous_data["groups"][kind])
        data["rows"][0] = previous
        data["source_bindings"] = e.bind(self.source) | e.bind(previous_source)
        changed = e.analyse(data, "DIAGNOSTIC")["rows"][1]
        self.assertIsNone(changed["features"]["theme_amount_share_delta_1"])
        self.assertEqual(changed["features"]["price_delta_1"], 2.)

    def test_no_closing_height_cannot_certify_all_zero_board_candidates_as_leaders(self):
        self.source_data["groups"]["pool"]["boards"] = {"000001": 0, "000002": 0}
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        self.row["boards"] = 0
        self.row["groups"]["pool"]["value"] = self.source_data["groups"]["pool"]
        for cell in self.row["groups"].values(): cell["source_sha256"] = e.sha(self.source)
        self.dataset["source_bindings"] = e.bind(self.source)
        actual = e.analyse(self.dataset, "DIAGNOSTIC")["rows"][0]
        self.assertFalse(actual["roles"]["HEIGHT"]["baseline_call"])
        self.assertEqual(actual["peer_codes"], [])

    def _height_sequence_fixture(self):
        pos = self.days.index("2026-10-09")
        days = self.days[pos-5:pos+1]
        data = copy.deepcopy(self.dataset)
        data["rows"], data["source_bindings"] = [], {}
        for i, day in enumerate(days):
            value = {"formed_at": day+"T15:05:00+08:00", "groups": {
                "pool": {"complete_within_scope": True, "codes": ["000001", "000002"],
                         "boards": {"000001": i+1, "000002": [2, 2, 3, 3, 4, 4][i]}}}}
            source = self.root/("height_"+day+".json")
            e.write_new(source, value)
            row = {"candidate_id": day+":000001", "signal_date": day, "code": "000001", "boards": i+1,
                "scope_id": "fixture", "available_at": day+"T15:10:00+08:00", "decision_at": day+"T16:30:00+08:00",
                "groups": {"pool": {"value": value["groups"]["pool"], "source_path": str(source),
                    "source_sha256": e.sha(source), "pointer": ["groups", "pool"], "status": "VERIFIED",
                    "observed_at": day+"T15:00:00+08:00", "available_at": day+"T15:10:00+08:00"}}}
            data["rows"].append(row)
            data["source_bindings"] |= e.bind(source)
        return data

    def test_height_sequence_uses_exact_official_sessions_and_complete_chain(self):
        data = self._height_sequence_fixture()
        actual = e.analyse(data, "STRICT")["rows"][-1]["features"]
        self.assertEqual(actual["scope_max_boards"], 6)
        self.assertEqual(actual["scope_top_count"], 1)
        self.assertEqual(actual["scope_height_delta_5"], 4)
        self.assertEqual(actual["own_boards_delta_5"], 5)
        self.assertEqual(actual["height_expansion_steps_5"], 4)
        self.assertEqual(actual["height_lead_retained_5"], 0.)
        self.assertEqual(actual["height_lead_retained_3"], 1.)
        data["rows"].pop(3)
        actual = e.analyse(data, "STRICT")["rows"][-1]["features"]
        self.assertEqual(actual["scope_height_delta_5"], 4)
        self.assertIsNone(actual["height_expansion_steps_5"])
        self.assertIsNone(actual["height_lead_retained_5"])

    def test_height_missing_stock_is_unknown_not_identity_failure(self):
        data = self._height_sequence_fixture()
        prior = data["rows"][0]
        source = Path(prior["groups"]["pool"]["source_path"])
        value = e.read(source)
        value["groups"]["pool"].update(codes=["000002", "000003"], boards={"000002": 2, "000003": 1})
        source.write_text(json.dumps(value), encoding="utf-8")
        prior.update(code="000003", candidate_id=prior["signal_date"]+":000003")
        prior["groups"]["pool"].update(value=value["groups"]["pool"], source_sha256=e.sha(source))
        data["source_bindings"] |= e.bind(source)
        actual = e.analyse(data, "STRICT")["rows"][-1]["features"]
        self.assertEqual(actual["scope_height_delta_5"], 4)
        self.assertEqual(actual["height_expansion_steps_5"], 4)
        self.assertIsNone(actual["own_boards_delta_5"])
        self.assertIsNone(actual["height_lead_retained_5"])

    def test_height_scope_and_strict_late_sources_cannot_supply_continuity(self):
        data = self._height_sequence_fixture()
        data["rows"][0]["scope_id"] = "different_archive_scope"
        self.assertIsNone(e.analyse(data, "STRICT")["rows"][-1]["features"]["scope_height_delta_5"])
        data["rows"][0]["scope_id"] = "fixture"
        for row in data["rows"]:
            row["available_at"] = "2026-10-10T10:00:00+08:00"
            row["groups"]["pool"]["available_at"] = row["available_at"]
            row["groups"]["pool"]["reconstructed"] = True
        self.assertEqual(e.analyse(data, "DIAGNOSTIC")["rows"][-1]["features"]["scope_height_delta_5"], 4)
        self.assertIsNone(e.analyse(data, "STRICT")["rows"][-1]["features"]["scope_height_delta_5"])

    def test_height_rejects_conflicting_scopes_and_own_board_disagreement(self):
        data = self._height_sequence_fixture()
        data["rows"][-1]["boards"] = 99
        self.assertFalse(e.analyse(data, "STRICT")["rows"][-1]["role_eligible"]["HEIGHT"])
        data["rows"][-1]["boards"] = 6
        other = copy.deepcopy(data["rows"][-1])
        source = self.root/"conflicting_pool.json"
        value = {"formed_at": other["signal_date"]+"T15:05:00+08:00", "groups": {
            "pool": {"complete_within_scope": True, "codes": ["000001", "000002"],
                     "boards": {"000001": 5, "000002": 4}}}}
        e.write_new(source, value)
        other.update(code="000002", candidate_id=other["signal_date"]+":000002", boards=4)
        other["groups"]["pool"].update(value=value["groups"]["pool"], source_path=str(source), source_sha256=e.sha(source))
        data["rows"].append(other)
        data["source_bindings"] |= e.bind(source)
        with self.assertRaisesRegex(ValueError, "conflicting height pools"):
            e.analyse(data, "STRICT")

    def test_annual_audit_retains_all_candidates_and_unknown_role_truth(self):
        data = self._height_sequence_fixture()
        dataset = self.root/"height_dataset.json"
        e.write_new(dataset, data)
        result = r.audit(dataset)
        year = result["annual_summary"]["2026"]
        self.assertEqual(year["candidate_count"], 6)
        self.assertEqual(year["label_support"]["HEIGHT"], {"UNKNOWN": 6})
        self.assertEqual(year["tracks"]["STRICT"]["height_temporal_available"]["scope_height_delta_5"], 1)
        self.assertEqual(len(result["height_temporal_records"]), 12)
        self.assertFalse(result["height_temporal_facts_are_independent_identity_labels"])
        self.assertEqual(result["fits"], 0)

    def fixture_source_roster_manifest(self):
        import pandas as pd
        from build_source_bound_theme_roles_v75 import build
        entries = []
        for code, boards, price, amount in [("000001", 3, 10., 100), ("000002", 1, 8., 50), ("000003", 0, 7., 75)]:
            entries.append({"signal_date": "2026-10-09", "code": code, "name": "synthetic_fixture",
                "boards": boards, "pool_assignment": "CLOSING_LIMIT_UP" if boards else "FAILED_BOARD_OBSERVATION",
                "snapshot_kind": "closing", "observed_at": "2026-10-09T15:00:00+08:00",
                "available_at": "2026-10-10T10:00:00+08:00", "last_price": price, "amount": amount,
                "turnover": 2. if boards else "", "change_pct": 10. if boards else 3., "first_seal_time": 93000+int(code),
                "limit_up_status": "VERIFIED_TWO_SOURCE_CLOSE" if boards else "FAILED_TO_CLOSE_AT_LIMIT"})
        pool = self.root/"full_pool.csv"; pd.DataFrame(entries).to_csv(pool, index=False)
        evidence = self.root/"members.json"
        value = {"signal_date": "2026-10-09", "groups": [{"theme_group_id": "pcb", "theme": "PCB板",
            "codes": ["000001", "000002", "000003", "000004"], "url": "https://fixture.invalid/members",
            "published_at": "2026-10-09T14:00:00+08:00", "source_tier": "SYNTHETIC_FIXTURE"}]}
        e.write_new(evidence, value)
        output = self.root/"candidate_theme_roles.csv"
        build(pd.read_csv(pool, dtype={"code": str}), value, "2026-10-10T10:00:00+08:00").to_csv(output,index=False)
        manifest = self.root/"manifest.json"
        e.write_new(manifest, {"version": "v75-source-bound-theme-roles-1", "signal_date": "2026-10-09",
            "asof": "2026-10-10T10:00:00+08:00", "source_pool": str(pool), "source_pool_sha256": e.sha(pool),
            "evidence_file": str(evidence), "evidence_sha256": e.sha(evidence), "output_sha256": e.sha(output)})
        return manifest, pool

    def test_saved_source_roster_preserves_failed_candidates_and_unknown_external_members(self):
        manifest, pool = self.fixture_source_roster_manifest()
        path = self.root/"source_dataset.json"; d = e.from_source_roster(manifest, self.cal, path)
        e.load_dataset(path)
        self.assertEqual(len(d["rows"]), 3)
        failed = next(row for row in d["rows"] if row["code"] == "000003")
        self.assertEqual(failed["boards"], 0)
        self.assertEqual(failed["pool_kind"], "failed_board")
        self.assertIsNone(failed["groups"]["price"]["value"]["turnover"])
        self.assertIsNone(failed["actual_one_word_daily"])
        self.assertIsNone(failed["groups"]["theme"]["value"]["members"]["000004"]["boards"])
        observed = e.analyse(d, "DIAGNOSTIC")
        self.assertEqual(observed["qualification"]["THEME"]["eligible"], 3)
        self.assertIsNone(observed["rows"][0]["features"]["theme_first_count"])
        self.assertEqual(e.analyse(d, "STRICT")["qualification"]["THEME"]["eligible"], 0)
        altered = copy.deepcopy(d); altered["rows"][0]["groups"]["price"]["value"]["price"] = 999.
        bad = self.root/"altered_source_projection.json"; e.write_new(bad, altered)
        with self.assertRaisesRegex(ValueError, "projection changed"):
            e.load_dataset(bad)

    def test_saved_source_roster_rejects_original_pool_hash_or_date_changes(self):
        import pandas as pd
        manifest, pool = self.fixture_source_roster_manifest()
        frame = pd.read_csv(pool, dtype={"code": str}); frame["signal_date"] = "2026-10-08"
        frame.to_csv(pool, index=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            e.from_source_roster(manifest, self.cal)
        declaration = e.read(manifest); declaration["source_pool_sha256"] = e.sha(pool)
        changed = self.root/"changed_date_manifest.json"; e.write_new(changed, declaration)
        with self.assertRaisesRegex(ValueError, "wrong dates"):
            e.from_source_roster(changed, self.cal)

    def test_calendar_horizon_exact(self):
        self.assertEqual(e.future_day(self.days, "2026-10-09", 3), "2026-10-14")

    def test_capacity_needs_non_limit_members(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertFalse(row["role_eligible"]["CAPACITY"])
        self.assertFalse(row["role_eligible"]["TREND"])

    def test_complete_competition_missing_peer_not_failure(self):
        value = {"complete_within_scope": True, "peer_codes": ["000002"],
                 "prior": {"000001": "STRONG", "000002": "STRONG"}, "current": {"000001": "STRONG"},
                 "prior_at": "2026-10-09T10:00:00+08:00", "current_at": "2026-10-09T14:00:00+08:00"}
        self.assertFalse(e._valid_group("competition", value, self.row))

    def test_bootstrap_abstention_counts_fn(self):
        records = [{"candidate_id": str(i), "signal_date": day,
            "role_truth": {role: True for role in e.ROLES}, "baseline_roles": {role: None for role in e.ROLES},
            "C1": {"calls": {role: True for role in e.ROLES}}} for i, day in enumerate(self.days[:6])]
        self.assertEqual(r.paired_macro_interval(records, self.days[:6]), [1., 1.])

    def test_confirmation_future_rejected(self):
        rec = self.root/"record.json"; conf = self.root/"confirmation.json"
        e.write_new(rec, {"signal_date": "2026-10-09", "target_date": "2026-10-12", "formed_at": "2026-10-11T10:00:00+08:00", "rows": [{"code": "000001"}]})
        e.write_new(conf, {"signal_date": "2026-10-09", "target_date": "2026-10-12", "observed_at": "2026-10-13T10:00:00+08:00", "rows": [], "source_bindings": {}})
        with self.assertRaisesRegex(ValueError, "actual"):
            review.confirm(rec, conf, self.root/"append.json")

    def test_augment_rebuild_and_original_cutoff_preserved(self):
        parent = self.root/"parent.json"; output = self.root/"augmented.json"
        e.write_new(parent, self.dataset)
        data = e.augment(parent, output)
        loaded = e.load_dataset(output)
        self.assertEqual(data["rows"], loaded["rows"])
        self.assertEqual(loaded["rows"][0]["original_decision_at"], self.row["decision_at"])
        self.assertGreater(e.stamp(loaded["rows"][0]["decision_at"]), e.stamp(self.row["decision_at"]))

    def test_rich_evidence_cannot_replace_price(self):
        parent, bank = self.root/"parent.json", self.root/"bank.json"
        e.write_new(parent, self.dataset)
        e.write_new(bank, {"version": e.VERSION, "formed_at": e.now(), "rows": [
            {"candidate_id": self.row["candidate_id"], "groups": {"price": self.row["groups"]["price"]}}]})
        with self.assertRaisesRegex(ValueError, "protected"):
            e.augment(parent, self.root/"bad.json", evidence_path=bank)

    def test_augmented_group_tamper_rejected(self):
        parent, output = self.root/"parent.json", self.root/"augmented.json"
        e.write_new(parent, self.dataset)
        data = e.augment(parent, output)
        data["rows"][0]["boards"] = 5
        output.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "projection"):
            e.load_dataset(output)

    def test_partial_current_dataset_not_strict_available(self):
        self.row["available_at"] = "2026-10-12T16:00:00+08:00"
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertFalse(row["role_eligible"]["HEIGHT"])

    def promotion_data(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        data = copy.deepcopy(self.dataset)
        data["adapter"] = {"kind": "AUGMENTED"}
        data["baseline_binding"] = {"original_codes": ["000001"]}
        data["baseline_promotion"] = {row["candidate_id"]: {"call": True, "raw_score": .8,
            "truth": None, "origin": "SOURCE_BOUND_PREAUCTION", "available_at": "2026-10-10T10:00:00+08:00"}}
        return data, row

    def test_C2_missing_role_falls_back_same_BASE(self):
        data, row = self.promotion_data()
        roles = {row["candidate_id"]: {"scores": {role: None for role in e.ROLES}}}
        pred = r.promotion_predictions(data, [row], roles, None)
        self.assertEqual(pred["ROLE_C2"][0]["raw_score"], .8)
        self.assertTrue(pred["ROLE_C2"][0]["prediction"])
        self.assertTrue(pred["ROLE_C2"][0]["fallback"])

    def test_C2_complete_roles_use_registered_width(self):
        data, row = self.promotion_data()
        class FixedFixture:
            n_features_in_ = len(e.RELAY_FIELDS)+len(e.ROLES)
            def predict_proba(self, x):
                self.x = x
                return np.asarray([[.7, .3]])
        model = FixedFixture()
        roles = {row["candidate_id"]: {"scores": {role: .6 for role in e.ROLES}}}
        pred = r.promotion_predictions(data, [row], roles, model)
        self.assertFalse(pred["ROLE_C2"][0]["prediction"])
        self.assertIsNone(pred["ROLE_C2"][0]["fallback"])
        self.assertEqual(model.x.shape[1], model.n_features_in_)

    def test_C2_cannot_delete_original_candidate(self):
        data, row = self.promotion_data(); data["baseline_binding"]["original_codes"].append("000002")
        with self.assertRaisesRegex(ValueError, "complete"):
            r.promotion_predictions(data, [row], {}, None)

    def test_original_gate_adapter_no_new_window_is_not_passed(self):
        value = r.assess_promotion([], {}, [], self.days)
        self.assertFalse(value["passed"])

    def test_original_gate_adapter_preserves_pure_modules(self):
        import relay_context_research as context
        original = context.ARMS.copy()
        facts = {"code": "000001", "boards": 2, "actual_one_word_daily": 0., "prediction": True, "raw_score": .8}
        record = {"signal_date": self.days[0], "target_dates": {"1": self.days[1]},
            "baseline_binding": {"original_codes": ["000001"]}, "promotion_predictions": {"BASE": [facts], "ROLE_C2": [dict(facts)]}}
        result = r.assess_promotion([record], {self.days[0]: {"000001": True}}, self.days[:5], self.days)
        self.assertFalse(result["passed"])
        self.assertEqual(context.ARMS, original)
        self.assertIsNone(result["probability_Brier"])

    def test_invalidation_delayed_observation_and_censoring(self):
        key = self.row["candidate_id"]
        rec = {"rows": [{"candidate_id": key}]}
        def event(h, truth, state):
            return {"registration_sha256": "fixture", "horizon": h, "rows": [{"candidate_id": key,
                "role_truth": {role: truth for role in e.ROLES}, "target_role_states": {role: state for role in e.ROLES}}]}
        result = r.invalidation_latency({"fixture": rec}, [event(0, True, "CANDIDATE"),
            event(3, False, "CANDIDATE"), event(5, False, "INVALIDATED")], self.days)
        self.assertEqual(result["mean_observed_delay_sessions"], 2.)
        self.assertEqual(result["observed_losses"], 11)
        self.assertIsNone(result["exact_intraday_loss_time_or_latency"])

    def test_unknown_initial_identity_cannot_claim_loss(self):
        key = self.row["candidate_id"]
        event = {"registration_sha256": "fixture", "horizon": 3, "rows": [{"candidate_id": key,
            "role_truth": {role: False for role in e.ROLES}, "target_role_states": {role: "INVALIDATED" for role in e.ROLES}}]}
        value = r.invalidation_latency({"fixture": {"rows": [{"candidate_id": key}]}}, [event], self.days)
        self.assertEqual(value["observed_losses"], 0)
        self.assertEqual(value["unknown_or_censored"], 11)

    def test_foreign_role_settlement_rejected(self):
        protocol = self.root/"p.json"; e.write_new(protocol, {})
        with self.assertRaisesRegex(ValueError, "registered"):
            r.verify_settlement({"registration_sha256": "not_registered", "protocol_sha256": e.sha(protocol)}, protocol, {})

    def test_abstaining_baseline_precision_gate_explicit_reference(self):
        a = r.binary_metrics({"a": True}, {"a": True})
        b = r.binary_metrics({"a": None}, {"a": True})
        self.assertTrue(r.metric_not_lower(a, b, "Precision"))
        self.assertIsNone(b["Precision"])

    def test_rich_competition_appended_from_exact_source(self):
        parent, bank, raw = self.root/"parent.json", self.root/"bank.json", self.root/"competition.json"
        value = {"complete_within_scope": True, "peer_codes": ["000002"],
            "prior": {"000001": "COMPETING", "000002": "STRONG"},
            "current": {"000001": "STRONG", "000002": "FAILED"},
            "prior_at": "2026-10-09T10:00:00+08:00", "current_at": "2026-10-09T14:00:00+08:00"}
        e.write_new(parent, self.dataset)
        e.write_new(raw, {"formed_at": "2026-10-10T10:00:00+08:00", "competition": value})
        cell = {"source_path": str(raw), "source_sha256": e.sha(raw), "pointer": ["competition"], "value": value,
            "status": "VERIFIED", "observed_at": "2026-10-09T14:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        e.write_new(bank, {"version": e.VERSION, "formed_at": e.now(), "rows": [{"candidate_id": self.row["candidate_id"], "groups": {"competition": cell}}]})
        out = self.root/"aug.json"; e.augment(parent, out, evidence_path=bank)
        row = e.analyse(e.load_dataset(out), "STRICT")["rows"][0]
        self.assertTrue(row["role_eligible"]["POSITION"])
        self.assertEqual(row["features"]["competition_failed_share"], 1.)
        self.assertEqual(row["roles"]["POSITION"]["state"], "CANDIDATE")

    def test_multirole_observation_does_not_certify_identity(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        p = {row["candidate_id"]: {"calls": {role: role in ("HEIGHT", "TURNOVER") for role in e.ROLES},
            "scores": {role: .8 if role == "HEIGHT" else .6 for role in e.ROLES}}}
        result = r.role_observations([row], p)[0]
        self.assertEqual(result["primary_role"], "HEIGHT")
        self.assertEqual(result["auxiliary_roles"], ["TURNOVER"])
        self.assertFalse(result["identity_certified"])

    def test_exact_promotion_successor_rejected_before_source_adapter(self):
        record = {"baseline_binding": {"compatibility_path": "fixture", "target_date": "2026-10-13"},
                  "source_bindings": {}, "dataset_path": "fixture", "signal_date": "2026-10-09",
                  "target_dates": {"1": "2026-10-13"}}
        with patch.object(e, "load_dataset", return_value={"calendar_path": str(self.cal)}):
            with self.assertRaisesRegex(ValueError, "successor"):
                r.reconstruct_promotion_outcomes(record, "no-source-needed")


if __name__ == "__main__":
    unittest.main()
