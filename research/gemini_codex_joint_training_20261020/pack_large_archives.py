"""Losslessly package three oversized, redundant scorecard JSON snapshots."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = "research/model_upgrade_validation/ranking_optimization_20261015T173045Z/scorecard/"
DIRECTORIES = ("final_verified22662_research_view", "final_verified26439_research_view", "full22662_research_view")
MANIFEST = Path(__file__).resolve().parent / "github_sync/LARGE_ARCHIVES.json"


def digest(path, compressed=False):
    sha = hashlib.sha256()
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(data)
    return sha.hexdigest()


def main():
    entries = []
    for directory in DIRECTORIES:
        source = ROOT / BASE / directory / "complete_scorecard.json"
        target = source.with_suffix(".json.gz")
        if target.exists():
            raise FileExistsError("Preserve an existing compressed archive")
        with source.open("rb") as data, target.open("xb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=6, mtime=0) as compressed:
                shutil.copyfileobj(data, compressed, length=1024 * 1024)
        original_sha = digest(source)
        assert original_sha == digest(target, compressed=True)
        assert target.stat().st_size < source.stat().st_size * .2
        entries.append({"original_path": source.relative_to(ROOT).as_posix(), "compressed_path": target.relative_to(ROOT).as_posix(),
                        "original_bytes": source.stat().st_size, "compressed_bytes": target.stat().st_size,
                        "original_sha256": original_sha, "compressed_sha256": digest(target)})
        print(json.dumps(entries[-1]), flush=True)
    MANIFEST.write_text(json.dumps({"format": "gzip, deterministic header, original bytes preserved", "files": entries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
