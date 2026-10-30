"""Prepare an isolated controller without opening a numerical batch or fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
original = ROOT / "first_board_joint_long_high_model.py"
destination = ROOT / "first_board_joint_event_path_model.py"
if destination.exists():
    raise FileExistsError("Do not overwrite an existing candidate controller")
text = original.read_text("utf-8")
changes = [
    ('"""One fixed joint experiment using a newly qualified signal-day reopening flag."""',
     '"""Fixed joint event-aligned correction paths; research only, no actor labels."""'),
    ('OUTPUT = ROOT / "research/gemini_codex_joint_long_high_20261021T022449Z"',
     'OUTPUT = None  # Explicit --output required; no numerical batch is opened on import.'),
    ('import first_board_long_high_features as features\nNEW_FIELDS_FILE = OUTPUT / "COMPLETE28503_ADMITTED_NOMINAL_HIGH_FEATURES_NO_TARGETS.csv"\nARMS = {"LONG_HIGH120_63": prior.ORIGINAL + [features.DISTANCE],\n        "LONG_HIGH120_AGE_64": prior.ORIGINAL + features.FIELDS}\nOWNERS = {"LONG_HIGH120_63": "Gemini", "LONG_HIGH120_AGE_64": "Codex"}',
     'import first_board_event_path_features as features\nDESIGN = ROOT / "research/forum_factor_design_20261021T025139Z"\nNEW_FIELDS_FILE = DESIGN / "COMPLETE28503_EVENT_PATH_FEATURES_NO_TARGETS.csv"\nALL_FIELDS = features.FIELDS["sealed"] + features.FIELDS["probe"]\n# Age/presence are deterministic transforms of original inputs and remain diagnostic.\nARMS = {"EVENT_SEALED_PATH_67": prior.ORIGINAL + features.FIELDS["sealed"][2:],\n        "EVENT_DUAL_PATH_72": prior.ORIGINAL + features.FIELDS["sealed"][2:] + features.FIELDS["probe"][2:]}\nOWNERS = {"EVENT_SEALED_PATH_67": "Gemini", "EVENT_DUAL_PATH_72": "Codex"}'),
    ('new[KEYS + features.FIELDS]', 'new[KEYS + ALL_FIELDS]'),
    ('assert result[features.AGE].dropna().between(1, 120).all()',
     'assert result[features.FIELDS["sealed"][1]].dropna().between(2, 20).all()\n    assert result[features.FIELDS["probe"][1]].dropna().between(1, 20).all()'),
    ('assert freeze["qualification_sha256"] == base.sha(OUTPUT / "qualification.json")',
     'assert freeze["qualification_sha256"] == base.sha(DESIGN / "EVENT_PATH_QUALIFICATION.json")'),
    ('args = parser.parse_args()\n    if args.command == "evaluate":',
     'parser.add_argument("--output", required=True, help="Exact new batch directory bound by its frozen protocol")\n    args = parser.parse_args()\n    OUTPUT = Path(args.output).resolve()\n    assert OUTPUT.is_relative_to((ROOT / "research").resolve())\n    if args.command == "evaluate":')
]
for before, after in changes:
    if text.count(before) != 1:
        raise AssertionError(f"Controller template drift: {before[:80]}")
    text = text.replace(before, after, 1)
with destination.open("x", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("ISOLATED_EVENT_PATH_CONTROLLER_PREPARED_NO_BATCH_NO_FITS")
