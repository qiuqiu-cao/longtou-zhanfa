"""Read-only pre-publication inventory: preserve evidence bytes and exclude ML runtimes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = Path(__file__).resolve().parent / "github_sync"
OUT = REPORT_ROOT / "compact_final"
TRACKED = [".gitignore", ".gitattributes", "AGENTS.md", "README.md", "templates/复盘_涨停优先与多日持续上涨.md"]
PATTERNS = {
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_pat": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "google_api_key": rb"AIza[0-9A-Za-z_-]{30,}",
    "openai_key": rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}",
    "aws_access_key": rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "slack_token": rb"xox[baprs]-[0-9A-Za-z-]{20,}",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    untracked = [p.decode("utf-8") for p in subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).split(b"\0") if p]
    staged = [p.decode("utf-8") for p in subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT
    ).split(b"\0") if p]
    candidates = sorted(set(TRACKED + untracked + staged))
    candidates = [p for p in candidates if (ROOT / p).is_file() and not p.startswith(REPORT_ROOT.relative_to(ROOT).as_posix() + "/")]
    rows, findings, oversized = [], [], []
    compiled = {k: re.compile(v) for k, v in PATTERNS.items()}
    for rel in candidates:
        p = ROOT / rel
        sha, previous = hashlib.sha256(), b""
        found = set()
        with p.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha.update(chunk)
                for name, pattern in compiled.items():
                    if pattern.search(previous + chunk):
                        found.add(name)
                previous = chunk[-512:]
        size = p.stat().st_size
        rows.append({"path": rel, "bytes": size, "sha256": sha.hexdigest()})
        for kind in sorted(found):
            findings.append({"path": rel, "pattern": kind})
        if size > 100_000_000:
            attr = subprocess.check_output(["git", "check-attr", "filter", "--", rel], cwd=ROOT, text=True).strip()
            oversized.append({"path": rel, "bytes": size, "lfs": attr.endswith(": lfs")})
    report = {
        "formed_at": datetime.now(timezone.utc).isoformat(),
        "head_before_sync": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": len(rows), "bytes": sum(r["bytes"] for r in rows),
        "credential_pattern_findings": findings,
        "oversized_files": oversized,
        "excluded": ["Local downloaded .whl and extracted dependency runtimes via .gitignore", "Existing unrelated BATTERY.response deletion is not staged"],
        "scope": "Working-tree inventory only; pattern screening is not a complete security certification",
        "status": "PASS" if not findings and all(x["lfs"] for x in oversized) else "REVIEW_REQUIRED",
    }
    (OUT / "FILES_BEFORE_SYNC.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "AUDIT_BEFORE_SYNC.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
