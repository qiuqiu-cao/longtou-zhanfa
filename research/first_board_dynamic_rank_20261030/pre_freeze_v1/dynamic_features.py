"""Daily reported theme cohorts and prior anchors, without future outcomes."""
from collections import defaultdict
import json
import numpy as np
import pandas as pd

KEYS = ['signal_date', 'code']
THEME = ['theme_peer_breadth', 'prior_cohort_relative_return',
         'prior_cohort_relative_survival', 'relative_reported_first_seal']
ANCHOR = ['prior_anchor_relative_return', 'prior_anchor_relative_survival', 'prior_anchor_height_gap']
FIELDS = THEME + ANCHOR


def no_targets(frame):
    if any('LABEL' in str(c) or c in {'outcome', 'label_end', 'target', 'reach4'} for c in frame):
        raise ValueError('future outcome columns are forbidden in the feature builder')


def feedback(codes, current):
    """No missing member is dropped to manufacture a complete cohort."""
    if not codes: return np.nan, np.nan, 'NO_OBSERVED_COHORT'
    rows = [current.get(c) for c in sorted(codes)]
    if any(r is None or not (r['eligible'] and r['valid_price'] and r['regular_price'])
           or not np.isfinite(r['day_return']) or pd.isna(r['limit']) for r in rows):
        return np.nan, np.nan, 'PARTIAL_OR_UNKNOWN_CURRENT_PRICES'
    return float(np.mean([r['day_return'] for r in rows])), float(np.mean([r['limit'] for r in rows])), 'COMPLETE_OBSERVED_COHORT'


def build(candidates, provider, states, sessions):
    for frame in (candidates, provider, states): no_targets(frame)
    if candidates.duplicated(KEYS).any() or provider.duplicated(KEYS).any() or states.duplicated(['date', 'code']).any():
        raise ValueError('duplicate candidate or context identity')
    positions = {str(day): i for i, day in enumerate(sessions)}
    if not set(candidates.signal_date).issubset(positions): raise ValueError('unknown official signal session')
    days = set(candidates.signal_date)
    days |= {sessions[positions[d]-1] for d in days if positions[d] > 0}
    state_days = {d: g.set_index('code').to_dict('index') for d, g in states[states.date.isin(days)].groupby('date')}
    raw_days = {d: g.set_index('code').to_dict('index') for d, g in provider[provider.signal_date.isin(days)].groupby('signal_date')}
    groups, observed = {}, {}
    for day, raw in raw_days.items():
        price = state_days.get(day, {})
        eligible = {c for c in raw if c in price and price[c]['eligible'] and price[c]['valid_price']
                    and price[c]['regular_price'] and price[c]['limit']}
        observed[day] = eligible
        topics = defaultdict(set)
        for code in sorted(eligible):
            for topic in raw[code]['themes']: topics[topic].add(code)
        groups[day] = dict(topics)
    rows = []
    for day, candidates_today in candidates.groupby('signal_date', sort=True):
        p = positions[day]; previous = sessions[p-1] if p else None
        today_raw, prior_raw = raw_days.get(day, {}), raw_days.get(previous, {})
        today_price, prior_price = state_days.get(day, {}), state_days.get(previous, {})
        current_groups, prior_groups = groups.get(day, {}), groups.get(previous, {})
        for candidate in candidates_today.sort_values('code').itertuples(index=False):
            code = candidate.code; raw = today_raw.get(code)
            theme_list = raw['themes'] if raw and code in observed.get(day, set()) else []
            global_previous = observed.get(previous, set()) - {code}
            global_high = {c for c in global_previous if prior_price[c]['height'] >= 3}
            market_return, market_survival, market_status = feedback(global_previous, today_price)
            high_return, high_survival, high_status = feedback(global_high, today_price)
            details = []
            for theme in theme_list:
                peers = current_groups.get(theme, set()) - {code}
                denominator = len(observed.get(day, set()) - {code})
                detail = {'theme': theme, 'current_peer_codes': sorted(peers),
                          'background_denominator': denominator, **{f: np.nan for f in FIELDS}}
                detail['theme_peer_breadth'] = np.log1p(len(peers)) / np.log1p(denominator) if denominator else np.nan
                times = [today_raw[c]['first_seal_minutes'] for c in sorted(peers)]
                own_time = raw['first_seal_minutes']
                if times and np.isfinite(own_time) and np.isfinite(times).all():
                    detail['relative_reported_first_seal'] = float(np.mean(np.sign(np.array(times) - own_time)))
                    detail['seal_status'] = 'COMPLETE_OBSERVED_PEER_TIMES'
                else: detail['seal_status'] = 'NO_PEERS_OR_INCOMPLETE_TIMES'
                prior_members = prior_groups.get(theme, set()) - {code}
                group_return, group_survival, status = feedback(prior_members, today_price)
                detail.update(prior_cohort_codes=sorted(prior_members), prior_cohort_status=status,
                              market_status=market_status, prior_source_present=bool(prior_raw))
                if status == market_status == 'COMPLETE_OBSERVED_COHORT':
                    detail['prior_cohort_relative_return'] = group_return - market_return
                    detail['prior_cohort_relative_survival'] = group_survival - market_survival
                anchors = set()
                if prior_members:
                    maximum = max(prior_price[c]['height'] for c in prior_members)
                    if maximum >= 3:
                        anchors = {c for c in prior_members if prior_price[c]['height'] == maximum}
                        detail['prior_anchor_height_gap'] = float(maximum - 1)
                    else:
                        detail['prior_anchor_height_gap'] = 0.0
                ar, alive, anchor_status = feedback(anchors, today_price)
                if anchor_status == high_status == 'COMPLETE_OBSERVED_COHORT':
                    detail['prior_anchor_relative_return'] = ar - high_return
                    detail['prior_anchor_relative_survival'] = alive - high_survival
                detail.update(prior_anchor_codes=sorted(anchors), prior_anchor_status=(anchor_status if anchors
                    else 'NO_OBSERVED_QUALIFYING_ANCHOR' if prior_members else 'NO_PRIOR_GROUP_OR_SOURCE'),
                    high_market_status=high_status)
                details.append(detail)
            row = {'signal_date': day, 'code': code, 'theme_count': len(details),
                   'context_status': 'RETROSPECTIVE_REPORTED_THEME' if details else 'UNKNOWN_THEME_OR_PRICE_CONTEXT',
                   'theme_version_verified_at_signal': False,
                   'source_path': raw['source_path'] if raw else None,
                   'source_sha256': raw['source_sha256'] if raw else None,
                   'source_received_at': raw['source_received_at'] if raw else None}
            for field in FIELDS:
                valid = [d[field] for d in details if np.isfinite(d[field])]
                row[field] = float(np.mean(valid)) if valid else np.nan
                row[field + '_valid_themes'] = len(valid)
            # JSON nulls, not invented zero observations.
            safe = [{k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                     for k, v in d.items()} for d in details]
            row['theme_details_json'] = json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
            rows.append(row)
    out = pd.DataFrame(rows)
    return candidates[KEYS].merge(out, on=KEYS, how='left', validate='one_to_one')
