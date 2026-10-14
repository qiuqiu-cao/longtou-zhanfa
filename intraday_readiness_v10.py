#!/usr/bin/env python3
"""V10 CLI alias for the readiness audit implemented in intraday_readiness_v9."""

import json

from intraday_readiness_v9 import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(
        run(args.snapshots, args.shadow, args.session, args.forward_v10, args.output),
        ensure_ascii=False, indent=2,
    ))
