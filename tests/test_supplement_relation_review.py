import copy
from pathlib import Path
import tempfile
import unittest

import stock_judgement_review as o
import supplement_relation_review as r


def fixture(out):
    dates = ["2026-10-09","2026-10-12","2026-10-13","2026-10-14"]
    calendar = out/"calendar.json"
    o.write_new(calendar,{"status":"verified","sessions":dates+["2026-10-15"]})
    text = out/"forum.txt"
    text.write_text("父锚带动低位，另一父锚也有关。\n评论：低位独立上涨也可能。",encoding="utf-8")
    identity = out/"identity.json"
    o.write_new(identity,[{"code":"000001","name":"父锚"},{"code":"000003","name":"另一父锚"}])
    price = dict(open=100,close=110,high=110,low=99)
    normalized = out/"daily.json"
    daily = {code:{"days":{d:price for d in dates}} for code in ("000001","000002","000004")}
    o.write_new(normalized,daily)
    proof = o.bind(calendar) | o.bind(text) | o.bind(identity) | o.bind(normalized)
    raw_paths = {}
    for code in daily:
        path = out/(code+"_raw.json")
        o.write_new(path,{"data":{"sz"+code:{"day":[[d,"1.00","1.10","1.10","0.99"] for d in dates]}}})
        raw_paths[code] = path
        proof |= o.bind(path)
    boards = out/"boards.json"
    board_values = {"000002":dict(zip(dates,[1,2,3,4])),"000001":dict(zip(dates,[3,6,None,None])),"000004":{dates[-1]:4},"broken":True}
    o.write_new(boards,board_values)
    proof |= o.bind(boards)
    available = "2026-10-14T16:00:00+00:00"
    observed = []
    for code in daily:
        for date in dates:
            n = board_values[code].get(date)
            observed.append({"code":code,"date":date,"ohlc_cents":price,"available_at":available,
                "raw_path":str(raw_paths[code]),"archived_boards":n,
                "ohlc_source":{"path":str(normalized),"sha256":o.sha(normalized),"json_pointer":f"/{code}/days/{date}"},
                "board_source":{"path":str(boards),"sha256":o.sha(boards),"json_pointer":f"/{code}/{date}"} if n else None})
    pool = out/"pool.json"
    o.write_new(pool,{"signal_date":dates[-1],"target_date":"2026-10-15","rows":[
        {"code":"000002","name":"低位","boards":4,"same_height_peers":["000004"]},
        {"code":"000004","name":"对手","boards":4,"same_height_peers":["000002"]}]})
    claim = {"claim_id":"body","source_id":"s","source_section":"AUTHOR_BODY","offset":0,
        "excerpt":"父锚带动低位，另一父锚也有关。","reviewed_at":"2026-10-14T16:10:00+00:00"}
    edge = {"edge_id":"a","child_code":"000002","anchor_code":"000001",
        "connection_strength":"EXPLICIT_CO_MENTION","relation_types":["NAME_EMOTION"],"claim_ids":["body"]}
    bank = {"version":r.BANK_VERSION,"formed_at":"2026-10-14T16:20:00+00:00","signal_date":dates[-1],"target_date":"2026-10-15",
        "calendar_path":str(calendar),"anchors_are_forecast_candidates":False,"source_bindings":proof,
        "sources":[{"source_id":"s","kind":"ORIGINAL_FORUM","body_available":True,"author":"作者","url":"https://example.com",
            "text_path":str(text),"text_sha256":o.sha(text),"published_at":"2026-10-12T12:00:00+00:00","obtained_at":available}],
        "claims":[claim],"edges":[edge],"temporal_facts":observed,
        "events":[{"code":"000001","date":"2026-10-13","event":"STREAK_BREAK","value":True,"available_at":available,
            "source":{"path":str(boards),"sha256":o.sha(boards),"json_pointer":"/broken"}}],
        "anchors":[{"code":code,"name":name,"identity_source":{"path":str(identity),"sha256":o.sha(identity),"json_pointer":f"/{i}"}}
            for i,(code,name) in enumerate((("000001","父锚"),("000003","另一父锚")))]}
    bank_path = out/"bank.json"
    o.write_new(bank_path,bank)
    return pool,bank_path,bank


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.pool,self.path,self.bank = fixture(self.out)

    def project(self,bank=None):
        if bank is not None:
            path = self.out/("bank_changed_"+str(len(list(self.out.glob("bank_changed*"))))+".json")
            o.write_new(path,bank)
        else:
            path = self.path
        return r.project(self.pool,path)

    def test_off_pool_parent_and_early_child_retains_all_peers(self):
        d = self.project()
        self.assertEqual(d["pool_size"],2)
        child = d["rows"][0]
        self.assertEqual(child["same_height_peers"],["000004"])
        edge = child["relations"][0]
        self.assertFalse(edge["anchor_in_current_pool"])
        self.assertEqual(edge["recognition_state"],"EARLY_START_CONTINUATION_CANDIDATE")
        self.assertEqual(edge["timing"]["anchor_boards_at_current_start"],3)
        self.assertIsNone(edge["promotion_direction"])
        self.assertIsNone(edge["training_role_label"])
        self.assertFalse(edge["confirmed_catchup_dragon"])

    def test_missing_parent_board_does_not_mean_streak_break(self):
        bank = copy.deepcopy(self.bank);bank["events"] = []
        edge = self.project(bank)["rows"][0]["relations"][0]
        self.assertEqual(edge["recognition_state"],"RELATION_BACKGROUND_TIMING_UNRESOLVED")

    def test_missing_child_day_does_not_certify_continuity(self):
        bank = copy.deepcopy(self.bank)
        bank["temporal_facts"] = [x for x in bank["temporal_facts"] if (x["code"],x["date"]) != ("000002","2026-10-12")]
        timing = self.project(bank)["rows"][0]["relations"][0]["timing"]
        self.assertFalse(timing["complete_streak_checked"])
        self.assertIsNone(timing["current_streak_start"])

    def test_alternative_parent_kept_when_chronology_unknown(self):
        bank = copy.deepcopy(self.bank)
        bank["edges"].append({**bank["edges"][0],"edge_id":"b","anchor_code":"000003"})
        edges = self.project(bank)["rows"][0]["relations"]
        self.assertEqual(len(edges),2)
        self.assertEqual(edges[1]["recognition_state"],"RELATION_BACKGROUND_TIMING_UNRESOLVED")

    def test_name_match_alone_cannot_be_supported_candidate(self):
        bank = copy.deepcopy(self.bank);bank["edges"][0]["connection_strength"] = "NAME_MATCH_ONLY"
        edge = self.project(bank)["rows"][0]["relations"][0]
        self.assertEqual(edge["recognition_state"],"NAME_LEAD_ONLY_NOT_SUPPORTED_RELATION")

    def test_comment_has_own_clock_and_max_review_availability(self):
        bank = copy.deepcopy(self.bank);bank["claims"][0]["source_section"] = "COMMENT"
        bank["claims"][0]["comment_published_at"] = "2026-10-13T01:00:00+00:00"
        claim = self.project(bank)["rows"][0]["relations"][0]["discussion_evidence"][0]
        self.assertEqual(claim["discussion_published_at"],"2026-10-13T01:00:00+00:00")
        self.assertEqual(claim["available_at"],"2026-10-14T16:10:00+00:00")
        bank["claims"][0].pop("comment_published_at")
        self.assertEqual(self.project(bank)["rows"][0]["relations"][0]["discussion_evidence"][0]["use"],"COMMENT_DATE_UNKNOWN")

    def test_claim_span_and_hash_changes_rejected(self):
        bank = copy.deepcopy(self.bank);bank["claims"][0]["offset"] += 1
        with self.assertRaisesRegex(ValueError,"span"):
            self.project(bank)
        Path(self.bank["sources"][0]["text_path"]).write_text("changed",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"source changed"):
            self.project()

    def test_late_review_cannot_use_earlier_declared_available_at(self):
        bank = copy.deepcopy(self.bank)
        bank["claims"][0].update(reviewed_at="2099-01-01T00:00:00+00:00",available_at="2026-10-14T16:00:00+00:00")
        with self.assertRaisesRegex(ValueError,"unavailable"):
            self.project(bank)

    def test_future_daily_row_rejected(self):
        bank = copy.deepcopy(self.bank);bank["temporal_facts"][0]["date"] = "2026-10-15"
        with self.assertRaisesRegex(ValueError,"future"):
            self.project(bank)

    def test_ohlc_and_board_have_separate_source_checks(self):
        bank = copy.deepcopy(self.bank);bank["temporal_facts"][0]["archived_boards"] = 100
        with self.assertRaisesRegex(ValueError,"board projection"):
            self.project(bank)
        bank = copy.deepcopy(self.bank);bank["temporal_facts"][0]["ohlc_cents"]["close"] = 999
        with self.assertRaisesRegex(ValueError,"OHLC"):
            self.project(bank)

    def test_pool_scores_and_dropped_peers_rejected(self):
        pool = o.read(self.pool);pool["rows"][0]["raw_score"] = .99
        path = self.out/"scored_pool.json";o.write_new(path,pool)
        with self.assertRaisesRegex(ValueError,"masked"):
            r.project(path,self.path)
        pool["rows"][0].pop("raw_score");pool["rows"][0]["same_height_peers"] = []
        path = self.out/"incomplete_pool.json";o.write_new(path,pool)
        with self.assertRaisesRegex(ValueError,"same-height"):
            r.project(path,self.path)

    def test_review_requires_every_edge_and_full_cohort(self):
        context = self.project()
        draft = {"rows":[{"code":x["code"],"supplement_assessment":r.assessment_template(x)} for x in context["rows"]]}
        with self.assertRaisesRegex(ValueError,"explicit"):
            r.validate_assessments(context,draft)
        for row in draft["rows"]:
            a = row["supplement_assessment"];a.update(reviewed=True,reason="独立复核")
            for edge in a["relations"]:
                edge.update(decision="PENDING",reason="父子因果未知")
        r.validate_assessments(context,draft)
        draft["rows"][0]["supplement_assessment"]["relations"][0]["reason"] = None
        with self.assertRaisesRegex(ValueError,"explicit reason"):
            r.validate_assessments(context,draft)
        draft["rows"][0]["supplement_assessment"]["relations"][0]["reason"] = "因果未知"
        draft["rows"][0]["supplement_assessment"]["relations"] = []
        with self.assertRaisesRegex(ValueError,"alternative"):
            r.validate_assessments(context,draft)

    def test_restart_keeps_old_first_observation_separate(self):
        dates = ["2026-10-09","2026-10-12","2026-10-13","2026-10-14","2026-10-15"]
        observations = {("000002","2026-10-13"):{"boards":1},("000002","2026-10-14"):{"boards":2}}
        events = [{"code":"000001","date":"2026-10-13","event":"STREAK_BREAK","value":True},
                  {"code":"000002","date":"2026-09-30","event":"PRIOR_FIRST_OBSERVED_BOARD","value":True}]
        timing = r._timing("000002","000001",{"boards":2},observations,events,dates,"2026-10-14")
        self.assertEqual(timing["state"],"OLD_FOLLOWER_RESTART_AT_ANCHOR_BREAK")
        self.assertEqual(timing["current_streak_start"],"2026-10-13")
        self.assertEqual(timing["prior_first_observed_board"],"2026-09-30")
        self.assertFalse(timing["prior_observation_is_lifetime_first"])

    def test_multiple_anchor_cycles_cannot_choose_a_convenient_break(self):
        bank = copy.deepcopy(self.bank)
        bank["events"].append({**bank["events"][0],"date":"2026-10-12"})
        timing = self.project(bank)["rows"][0]["relations"][0]["timing"]
        self.assertEqual(timing["state"],"UNKNOWN")
        self.assertIn("multiple anchor cycles require separately bound cycle scope",timing["missing"])

    def test_shanghai_publication_day_not_utc_day(self):
        bank = copy.deepcopy(self.bank)
        bank["sources"][0]["published_at"] = "2026-10-13T20:30:00+00:00"
        claim = self.project(bank)["rows"][0]["relations"][0]["discussion_evidence"][0]
        self.assertEqual(claim["use"],"SIGNAL_DAY_DISCUSSION")


if __name__ == "__main__":
    unittest.main()
