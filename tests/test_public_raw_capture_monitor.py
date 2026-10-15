import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import public_raw_capture_monitor as raw


class Response:
    def __init__(self, body, status=200):
        self.content, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError("HTTP "+str(self.status_code))


class RawCaptureTests(unittest.TestCase):
    def test_actual_bytes_and_stock_field_pointers_survive_native_parser(self):
        fields = ["测试股", "10", "9.9", "10.1", "10.2", "9.8", "10", "10.1", "12300", "123456"]+["0"]*20+["2026-09-07", "09:25:01", "00"]
        body = ('var hq_str_sh600001="'+','.join(fields)+'";').encode("gb18030")
        with tempfile.TemporaryDirectory() as tmp:
            recorder = raw.RecordingGet(Path(tmp)/"raw", "quote", get=lambda *a, **k: Response(body))
            result = raw.quotes.fetch_sina_quotes(["600001"], get=recorder, retries=1)
            self.assertEqual(result.quotes.iloc[0]["code"], "600001")
            self.assertEqual(result.raw_hashes, [hashlib.sha256(body).hexdigest()])
            self.assertTrue(raw.verify_receipts(recorder.entries))
            entry = json.loads(Path(recorder.entries[0]["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(Path(entry["raw_path"]).read_bytes(), body)
            ref = entry["field_references"][0]["fields"]["last"]
            self.assertEqual(body[ref["byte_offset"]:ref["byte_offset"]+ref["byte_length"]], b"10.1")
            self.assertEqual(entry["field_references"][0]["observed_at_native"], "2026-09-07 09:25:01")
            Path(entry["raw_path"]).write_bytes(body+b" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                raw.verify_receipts(recorder.entries)

    def test_transport_failure_and_one_retry_leave_two_receipts_no_fabricated_raw(self):
        def unavailable(*args, **kwargs):
            raise TimeoutError("bounded provider failure")
        with tempfile.TemporaryDirectory() as tmp:
            recorder = raw.RecordingGet(Path(tmp)/"raw", "quote", get=unavailable)
            result = raw.quotes.fetch_sina_quotes(["600001"], get=recorder, retries=1)
            self.assertEqual(len(recorder.entries), 2)
            self.assertTrue(result.quotes.empty)
            for ref in recorder.entries:
                item = json.loads(Path(ref["receipt_path"]).read_text(encoding="utf-8"))
                self.assertEqual(item["status"], "TRANSPORT_OR_ARCHIVE_FAILURE")
                self.assertNotIn("raw_path", item)
            self.assertTrue(raw.verify_receipts(recorder.entries))

    def test_minute_native_array_pointer_and_http_failure_bodies_are_saved(self):
        body = b'var _data=([{"day":"2026-09-07 09:35:00","open":"10","close":"10.1","high":"10.2","low":"10","volume":"100","amount":"1000"}]);'
        with tempfile.TemporaryDirectory() as tmp:
            recorder = raw.RecordingGet(Path(tmp)/"minute", "minute", get=lambda *a, **k: Response(body))
            fetched = raw.minutes.fetch_minute_bars(["600001"], get=recorder, retries=1)
            self.assertEqual(len(fetched.bars), 1)
            entry = json.loads(Path(recorder.entries[0]["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(entry["field_references"][0]["code"], "600001")
            self.assertEqual(entry["field_references"][0]["fields"]["close"]["array_json_pointer"], "/0/close")
            self.assertTrue(raw.verify_receipts(recorder.entries))
            error = raw.RecordingGet(Path(tmp)/"http_error", "quote", get=lambda *a, **k: Response(b"provider error", 501))
            fetched = raw.quotes.fetch_sina_quotes(["600001"], get=error, retries=1)
            self.assertEqual(len(error.entries), 2)
            self.assertTrue(fetched.quotes.empty)
            self.assertTrue(raw.verify_receipts(error.entries))

    def test_wrong_session_or_silently_reduced_candidate_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calendar=root/"calendar.json"; pool=root/"pool.csv"
            calendar.write_text(json.dumps({"status":"verified","source":"fixture", "sessions":["2026-09-04","2026-09-07","2026-09-08"]}))
            pool.write_text("signal_date,code,name,boards\n2026-09-04,600001,A,2\n",encoding="utf-8")
            self.assertEqual(len(raw.validate_pool(pool,calendar,"2026-09-07")[0]),1)
            with self.assertRaisesRegex(ValueError,"predecessor"):
                raw.validate_pool(pool,calendar,"2026-09-08")
            with pool.open("a",encoding="utf-8") as f:
                f.write("2026-09-04,300001,filtered,2\n")
            with self.assertRaisesRegex(ValueError,"denominator"):
                raw.validate_pool(pool,calendar,"2026-09-07")

    def test_native_capture_cannot_bind_an_unrelated_raw_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            recorder=raw.RecordingGet(root/"raw", "quote", get=lambda *a, **k: Response(b"body"))
            recorder("https://example.invalid/fixture")
            manifest=root/"native.json"
            manifest.write_text(json.dumps({"provider_response_hashes":[hashlib.sha256(b"different").hexdigest()]}))
            link={"request_receipts":recorder.entries,"native_manifest":str(manifest),"native_sha256":raw.sha(manifest)}
            with self.assertRaisesRegex(ValueError,"no matching original"):
                raw.verify_capture_link(link)

    def test_malformed_minute_body_remains_a_verifiable_failed_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder=raw.RecordingGet(Path(tmp)/"raw", "minute", get=lambda *a, **k: Response(b'var _data=([{"invalid":}]);'))
            recorder("https://example.invalid/fixture",params={"symbol":"sh600001"})
            self.assertTrue(raw.verify_receipts(recorder.entries))
            entry=json.loads(Path(recorder.entries[0]["receipt_path"]).read_text(encoding="utf-8"))
            self.assertIn("pointer_error",entry)
            self.assertEqual(entry["field_references"],[])


if __name__ == "__main__":
    unittest.main()
