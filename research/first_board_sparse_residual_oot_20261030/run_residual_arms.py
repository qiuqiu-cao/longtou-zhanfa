"""Safe entry point: replay archived weights only; no implicit optimization.

New MILP fits use fit_joint_residual.py with a separately frozen protocol.
The original DE source and original report remain in review_source_snapshot/.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("review_residual_arms.py")), run_name="__main__")
