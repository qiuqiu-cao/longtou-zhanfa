from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace matching dated rows in a market-emotion history with an immutable reconciled overlay.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    history = pd.read_csv(args.history, dtype={"date": str})
    overlay = pd.read_csv(args.overlay, dtype={"date": str})
    if overlay.empty or overlay["date"].duplicated().any():
        raise ValueError("overlay must contain unique dated rows")
    merged = history.loc[~history["date"].isin(overlay["date"])].copy()
    merged = pd.concat([merged, overlay], ignore_index=True, sort=False)
    merged["date"] = pd.to_datetime(merged["date"], errors="raise").dt.strftime("%Y-%m-%d")
    merged = merged.sort_values("date", kind="stable")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(output.resolve())


if __name__ == "__main__":
    main()
