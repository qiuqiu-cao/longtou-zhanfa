"""Restore the three lossless scorecards after clone, without replacing existing evidence."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = Path(__file__).resolve().parent / "github_sync/LARGE_ARCHIVES.json"
    for row in json.loads(manifest.read_text("utf-8"))["files"]:
        target, archive = ROOT / row["original_path"], ROOT / row["compressed_path"]
        assert target.resolve().is_relative_to(ROOT) and archive.resolve().is_relative_to(ROOT)
        if target.exists():
            assert sha(target) == row["original_sha256"], "Existing evidence differs; refusing overwrite"
            print("Already verified:", row["original_path"])
            continue
        assert sha(archive) == row["compressed_sha256"]
        temporary = target.with_suffix(".json.restoring")
        with gzip.open(archive, "rb") as stream, temporary.open("xb") as output:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                output.write(block)
        assert temporary.stat().st_size == row["original_bytes"] and sha(temporary) == row["original_sha256"]
        assert not target.exists(), "Another process created the target; original is left untouched"
        temporary.rename(target)
        print("Restored and verified:", row["original_path"])


if __name__ == "__main__":
    main()
