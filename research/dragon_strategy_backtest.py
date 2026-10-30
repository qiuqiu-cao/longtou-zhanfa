"""Corrected entry point: use one shared engine and named configurations.
The original ad hoc 6:4 implementation is preserved in the review snapshot.
No fitting, weight search, live orders, or automatic deployment.
"""
from pathlib import Path
import importlib.util
import sys

def main():
    path = Path(__file__).resolve().parent / "first_board_optimization_20261030/compare_fixed_overnight.py"
    spec = importlib.util.spec_from_file_location("fixed_overnight_comparison", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()

if __name__ == "__main__":
    main()
