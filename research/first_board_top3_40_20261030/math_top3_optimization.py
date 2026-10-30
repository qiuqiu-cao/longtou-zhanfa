"""Replay archived cascade weights; no automatic in-sample optimization.

Original optimizer, weights and reported results remain preserved at
research/first_board_math_20261030/gemini_pre_fix/.
The weights saw evaluation outcomes and are diagnostic only.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      "first_board_math_20261030/review_gemini_cascade.py"), run_name="__main__")
