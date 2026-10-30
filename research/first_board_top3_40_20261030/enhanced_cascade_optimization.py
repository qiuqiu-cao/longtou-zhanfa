"""Replay the enhanced cascade's saved weights; no automatic weight fitting.

Original source/results are preserved in
research/first_board_enhanced_cascade_review_20261030/source_snapshot/.
All saved weights are in-sample research artifacts, not validated production models.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      "first_board_enhanced_cascade_review_20261030/replay_enhanced.py"), run_name="__main__")
