"""One public anonymous dated-industry probe, bounded by the parent process."""
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import socket
import hashlib

OUT=Path(__file__).resolve().parent
WHEEL=OUT/"baostock-0.9.4-py3-none-any.whl"
assert hashlib.sha256(WHEEL.read_bytes()).hexdigest()=="0bf71c6069ab5890ff3596632f9c3f8f1fbc6bfcac582c2f9d6a5c11ab2cfaf8"
sys.path.insert(0,str(WHEEL))
socket.setdefaulttimeout(8)
import baostock as bs
import baostock.util.socketutil as transport

def now():return datetime.now(timezone.utc).isoformat()

date=sys.argv[1]
receipt_path=OUT/(date+"_request_receipt.json")
if receipt_path.exists():raise FileExistsError("Refuse repeated unchanged probe")
record=dict(requested_at=now(),method="query_stock_industry",requested_date=date,code_scope="provider full classification snapshot",client_version="0.9.4",
            authentication="provider documented public anonymous default;no private API key",timeout_seconds=8,one_attempt=True)
receipt_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
frames=[]
original_send=transport.send_msg
def record_send(message):
    # Do not capture login/password traffic, only industry-query responses.
    response=original_send(message)
    if "query_stock_industry" in message and response:
        p=OUT/(date+f"_industry_response_{len(frames)+1}.utf8")
        p.write_bytes(response.encode("utf-8"))
        frames.append(dict(file=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                           received_at=now(),medium="API decoded UTF8 message;not a wire-packet capture"))
    return response
transport.send_msg=record_send
try:
    login=bs.login()
    record["login_error_code"]=login.error_code
    record["login_message"]=login.error_msg
    if login.error_code!="0":
        record["status"]="LOGIN_FAILED_NO_INDUSTRY_REQUEST"
    else:
        result=bs.query_stock_industry(date=date)
        record["query_error_code"]=result.error_code
        record["query_message"]=result.error_msg
        rows=[]
        while result.error_code=="0" and result.next():rows.append(result.get_row_data())
        record["fields"]=result.fields
        record["rows"]=len(rows)
        (OUT/(date+"_parsed_response.json")).write_text(json.dumps(dict(fields=result.fields,rows=rows),ensure_ascii=False,indent=2),encoding="utf-8")
        if rows:
            import pandas as pd
            data=pd.DataFrame(rows,columns=result.fields)
            data.to_csv(OUT/(date+"_snapshot.csv"),index=False,encoding="utf-8-sig")
            record["update_date_range"]=[data.updateDate.min(),data.updateDate.max()]
            record["future_update_rows"]=int(data.updateDate.gt(date).sum())
            record["status"]="DATED_SNAPSHOT_CANDIDATE" if record["future_update_rows"]==0 else "REJECTED_FUTURE_DATED_RESPONSE"
        else:record["status"]="EMPTY_OR_ERROR_NOT_A_SNAPSHOT"
        bs.logout()
except Exception as exc:
    record.update(status="EXCEPTION",error_type=type(exc).__name__,error=str(exc)[:400])
finally:
    record.update(completed_at=now(),response_frames=frames)
    receipt_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:record.get(k) for k in ["status","requested_date","rows","update_date_range","query_error_code","login_error_code"]},ensure_ascii=True))
