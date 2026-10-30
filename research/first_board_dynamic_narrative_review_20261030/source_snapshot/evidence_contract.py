"""Research-only narrative evidence contract, not a signal or scoring model.

Evidence is supplied by reviewers, never inferred from matching names. A late
retrieval can document history, but is not an actual-time archived signal.
"""
from datetime import datetime
from collections import defaultdict
import hashlib
from pathlib import Path

THEME_TYPES = {'industry', 'policy', 'event', 'name_narrative'}
RELATIONS = {'catch_up_hypothesis', 'common_catalyst', 'independent_line', 'ordinary_follower', 'unknown'}
ROLES = {'pioneer', 'receiving_core', 'catch_up_candidate', 'follower', 'unknown'}
PHASES = {'startup', 'diffusion', 'acceleration', 'divergence', 'repair', 'retreat', 'unknown'}


def instant(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('time must include timezone')
    return t


def source_state(source, cutoff):
    """Return separate publication and acquisition gates, with no assumed clocks."""
    deadline = instant(cutoff)
    acquired = instant(source['obtained_at'])
    published = source.get('published_at')
    if not published:
        return {'published_by_cutoff': None, 'acquired_by_cutoff': acquired <= deadline,
                'actual_archived_eligible': False, 'reason': 'publication_time_unknown'}
    pub = instant(published)
    if acquired < pub:
        raise ValueError('acquisition precedes publication')
    return {'published_by_cutoff': pub <= deadline, 'acquired_by_cutoff': acquired <= deadline,
            'actual_archived_eligible': pub <= deadline and acquired <= deadline,
            'reason': 'ok' if pub <= deadline and acquired <= deadline else 'late_publication_or_acquisition'}


def check_claims(bundle):
    """Validate explicit claims and preserve multiple independent topic memberships."""
    cutoff = bundle['cutoff']
    instant(cutoff)
    sources = bundle['sources']
    seen = set()
    out = []
    for claim in bundle['claims']:
        key = claim['claim_id']
        if key in seen:
            raise ValueError('duplicate claim')
        seen.add(key)
        if claim['theme_type'] not in THEME_TYPES:
            raise ValueError('unsupported theme type')
        if claim['role'] not in ROLES or claim['phase'] not in PHASES:
            raise ValueError('unsupported role or phase')
        if claim['relation'] not in RELATIONS:
            raise ValueError('unsupported relation')
        if not all(claim.get(k) for k in ('code', 'theme_id', 'evidence_text', 'source_id', 'counterevidence', 'reviewer')):
            raise ValueError('claim, reviewer, source and counterevidence required')
        source = sources[claim['source_id']]
        if not source.get('url') or not source.get('source_description'):
            raise ValueError('source provenance required')
        raw = Path(source['saved_text_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['saved_text_sha256']:
            raise ValueError('saved source text hash mismatch')
        text = raw.decode('utf-8')
        if claim.get('source_span') not in text or not claim.get('source_span'):
            raise ValueError('exact source span required')
        if instant(claim['reviewed_at']) < instant(source['obtained_at']):
            raise ValueError('review cannot precede source acquisition')
        if instant(claim['reviewed_at']) > instant(bundle['formed_at']):
            raise ValueError('review cannot follow bundle formation')
        eligibility = source_state(source, cutoff)
        # Explicit unknowns are not negative observations, or a numerical veto.
        out.append({**claim, 'source_eligibility': eligibility,
                    'claim_actually_available_by_cutoff': bool(eligibility['actual_archived_eligible']
                        and instant(claim['reviewed_at']) <= instant(cutoff)
                        and instant(bundle['formed_at']) <= instant(cutoff)),
                    'is_confirmed_causal_leadership': False,
                    'claim_is_author_interpretation': True})
    grouped = defaultdict(list)
    for claim in out:
        grouped[claim['code']].append(claim['theme_id'])
    return {'claims': out, 'multi_memberships': dict(grouped),
            'numeric_features_generated': False, 'ranking_changed': False}
