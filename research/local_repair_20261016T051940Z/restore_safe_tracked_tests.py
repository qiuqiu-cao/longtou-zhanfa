"""Restore missing tracked XML only after IP-pattern and exact-blob checks."""
from pathlib import Path
import hashlib
import ipaddress
import json
import re
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path(__file__).resolve().parent


def run():
    receipt=OUTPUT/'exact_tracked_test_restore.json'
    if receipt.exists():raise FileExistsError('restore receipt already exists')
    deleted=subprocess.run(['git','ls-files','--deleted','-z'],cwd=ROOT,capture_output=True,check=True).stdout.decode().split('\0')
    restored=[];preserved=[]
    for name in deleted:
        if not name:continue
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('restore target outside intended project')
        if path.suffix.lower()!='.xml' or 'test' not in path.name.lower():
            preserved.append(dict(path=name,action='LEFT_DELETED_NO_PRIVATE_RAW_RESPONSE_RESTORE'));continue
        raw=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True,check=True).stdout
        for item in re.findall(rb'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',raw):
            try:ipaddress.ip_address(item.decode())
            except ValueError:continue
            raise ValueError('IP pattern detected: refuse restoration without separate redaction')
        if re.search(rb'(?i)(?:[0-9a-f]{0,4}:){2,}[0-9a-f:]{1,39}',raw):
            # XML timestamps contain colons; require successful IPv6 parsing.
            for item in re.findall(rb'(?i)(?:[0-9a-f]{0,4}:){2,}[0-9a-f:]{1,39}',raw):
                try:ipaddress.IPv6Address(item.decode())
                except ValueError:continue
                raise ValueError('IPv6 pattern detected: refuse raw restoration')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
        digest=hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('restore hash mismatch')
        restored.append(dict(path=name,sha256=digest,exact_HEAD_bytes=True,IP_patterns_detected=0))
    value=dict(formed_at=datetime.now(timezone.utc).isoformat(),restored=restored,left_deleted=preserved,
        network_requests=0,raw_IP_values_output=False,old_receipts_modified=False)
    receipt.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(exact_test_files_restored=len(restored),non_test_deleted_files_left_unrestored=len(preserved),IP_values_output=False)))


if __name__=='__main__':run()
