"""Complete same-height views of existing model scores; no new prediction."""
import math


def summarize(rows):
    codes = [r["code"] for r in rows]
    if len(codes) != len(set(codes)):
        raise ValueError("complete cohort contains duplicate codes")
    groups = {}
    for row in rows:
        if (type(row["boards"]) is not int or row["boards"] < 2
                or type(row["raw_score"]) not in (int, float)
                or not math.isfinite(row["raw_score"])
                or type(row["model_positive"]) is not bool
                or row["model_positive"] != (row["raw_score"] > .5)
                or row["independent_call"] not in {"PROMOTE", "FAIL", "UNKNOWN"}):
            raise ValueError("original finite score, board and direction required")
        groups.setdefault(row["boards"], []).append(row)
    full_nonword = sorted((r for r in rows if r.get("actual_one_word_daily") == 0),
                          key=lambda r: (-r["raw_score"], r["code"]))
    full_nonword_ranks = {r["code"]: rank for rank, r in enumerate(full_nonword, 1)}
    output = []
    for boards, group in sorted(groups.items(), reverse=True):
        ordered = sorted(group, key=lambda r: (-r["raw_score"], r["code"]))
        nonword = [r for r in ordered if r.get("actual_one_word_daily") == 0]
        nonword_ranks = {r["code"]: rank for rank, r in enumerate(nonword, 1)}
        members = []
        for rank, row in enumerate(ordered, 1):
            members.append({"code": row["code"], "name": row["name"], "same_height_rank": rank,
                "raw_score": row["raw_score"], "display_score": round(100*row["raw_score"], 1),
                "original_model_positive": row["model_positive"],
                "independent_call": row["independent_call"],
                "actual_one_word_daily": row.get("actual_one_word_daily"),
                "low_turnover_proxy": row.get("low_turnover_proxy"),
                "full_pool_nonword_rank": full_nonword_ranks.get(row["code"]),
                "same_height_nonword_rank": nonword_ranks.get(row["code"]),
                "display_score_gap_to_group_first": 100*(row["raw_score"]-ordered[0]["raw_score"]),
                "display_score_gap_to_nonword_first": (100*(row["raw_score"]-nonword[0]["raw_score"])
                    if row["code"] in nonword_ranks else None),
                "relative_nonword_observation": bool(nonword and row["code"] in nonword_ranks
                    and row["raw_score"] == nonword[0]["raw_score"]),
                "missing_model_fields": row.get("missing_model_fields", []),
                "missing_facts": row.get("missing_facts", []),
                "independent_conflict": row["independent_call"] != "UNKNOWN" and
                    (row["independent_call"] == "PROMOTE") != row["model_positive"]})
        positive_nonword = [r for r in nonword if r["model_positive"]]
        output.append({"boards": boards, "target_boards": boards+1,
            "candidate_count": len(members), "members": members,
            "relative_rank_first": ordered[0]["code"],
            "rank_first_original_model_positive": ordered[0]["model_positive"],
            "model_positive_codes": [r["code"] for r in ordered if r["model_positive"]],
            "all_model_negative": not any(r["model_positive"] for r in ordered),
            "nonword_relative_first": nonword[0]["code"] if nonword else None,
            "nonword_relative_tied_first_codes": [r["code"] for r in nonword
                if r["raw_score"] == nonword[0]["raw_score"]],
            "nonword_positive_first": positive_nonword[0]["code"] if positive_nonword else None,
            "top_two_original_display_gap": 100*(ordered[0]["raw_score"]-ordered[1]["raw_score"]) if len(ordered)>1 else None,
            "nonword_top_two_original_display_gap": (100*(nonword[0]["raw_score"]-nonword[1]["raw_score"])
                if len(nonword)>1 else None),
            "relative_observation_uses_positive_threshold": False,
            "relative_rank_confirms_active_leadership": False,
            "paths_retained": ["JOINT_PROMOTION", "JOINT_FAILURE", "DIFFERENTIATION"],
            "ranking_score_is_probability": False, "new_ranking_model_fitted": False,
            "entry_selection_authorized": False, "group_roster_scope": "COMPLETE_ORIGINAL_SAME_HEIGHT_POOL"})
    return output
