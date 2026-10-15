"""Preserve issuer-document mirrors and extract all six candidates' recent notices."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests
import fitz

OUT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def one(item):
    code, notice = item
    art = notice["art_code"]
    url = f"https://pdf.dfcfw.com/pdf/H2_{art}_1.pdf"
    result = dict(code=code, title=notice["title"], indexed_publication_date=notice["notice_date"],
                  precise_first_publication_time=None, url=url, requested_at=now(), maximum_attempts=1,
                  source_scope="ISSUER_DOCUMENT_ON_PUBLIC_MIRROR_NOT_EXCHANGE_HOST_ATTESTATION")
    try:
        response = requests.get(url, timeout=(4, 8), headers={"User-Agent":"Mozilla/5.0"})
        result.update(received_at=now(), http_status=response.status_code)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError("not PDF bytes")
        path = OUT / "notice_fulltext" / (code + "_" + art + ".pdf")
        path.write_bytes(response.content)
        with fitz.open(stream=response.content, filetype="pdf") as document:
            text = "\n".join(page.get_text(sort=True) for page in document)
            first = document[0].get_pixmap(matrix=fitz.Matrix(0.9, 0.9))
            image_path = path.with_suffix(".png")
            first.save(image_path)
        if len(text.strip()) < 80:
            raise ValueError("insufficient extracted text; no fulltext clearance")
        text_path = path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        result.update(status="PDF_TEXT_AND_FIRST_PAGE_SAVED", pdf_path=str(path),
                      text_path=str(text_path), preview_path=str(image_path),
                      pdf_sha256=hashlib.sha256(response.content).hexdigest(),
                      text_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest())
    except Exception as exc:
        result.update(status="UNAVAILABLE", error=str(exc)[:400])
    return result


def main():
    (OUT / "notice_fulltext").mkdir(exist_ok=False)
    manifest=json.loads((OUT / "bounded_sources/manifest.json").read_text(encoding="utf-8"))
    jobs=[(entry["code"], entry["notices"][0]) for entry in manifest["entries"]
          if entry["kind"]=="notices" and entry.get("notices")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows=list(executor.map(one,jobs))
    with (OUT / "notice_fulltext/manifest.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(formed_at=now(),rows=rows),stream,ensure_ascii=False,indent=2)
    print(json.dumps(rows,ensure_ascii=False))


if __name__=="__main__":
    main()
