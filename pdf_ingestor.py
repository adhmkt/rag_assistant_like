import argparse
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}, "failed": {}, "meta": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": {}, "failed": {}, "meta": {}}


def _save_progress(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _range_key(start_page: int, end_page: int) -> str:
    return f"{start_page}-{end_page}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the VisitAssist RAG ingest API in batches.")
    parser.add_argument("--pdf", dest="pdf_path", default=r"C:\\rag-project\\rag_assistant_like\\visitassist_rag\\itaipu.pdf", help="Path to PDF file to ingest.")
    parser.add_argument("--kb-id", default="itaipu")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--batch-size", type=int, default=10, help="Pages per batch (smaller is safer).")
    parser.add_argument("--start-page", type=int, default=1, help="1-based inclusive")
    parser.add_argument("--end-page", type=int, default=0, help="1-based inclusive; 0 means until end")
    parser.add_argument("--timeout", type=int, default=900, help="Read timeout seconds (embedding can be slow).")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=5.0, help="Base backoff seconds between retries")
    parser.add_argument("--sleep-between", type=float, default=0.5, help="Sleep seconds between successful batches")
    parser.add_argument("--language", default="pt")
    parser.add_argument("--source-type", default="pdf")
    parser.add_argument("--title-prefix", default="itaipu.pdf")
    parser.add_argument("--doc-date", default="2020-01-01")
    parser.add_argument("--doc-year", type=int, default=2020)
    parser.add_argument(
        "--progress-file",
        default="pdf_ingestor_progress.json",
        help="Tracks completed page ranges to allow resume without duplicates.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue ingesting subsequent ranges even if one batch fails.",
    )

    args = parser.parse_args()

    pdf_path = str(args.pdf_path)
    progress_path = Path(args.progress_file)
    progress = _load_progress(progress_path)
    progress.setdefault("completed", {})
    progress.setdefault("failed", {})
    progress.setdefault("meta", {})
    progress["meta"].update({
        "pdf": pdf_path,
        "kb_id": args.kb_id,
        "url": args.url,
        "batch_size": args.batch_size,
        "language": args.language,
        "doc_date": args.doc_date,
        "doc_year": args.doc_year,
        "title_prefix": args.title_prefix,
    })
    _save_progress(progress_path, progress)

    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        print(f"Opened PDF: {pdf_path} with {num_pages} pages.")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return 1

    start_page = max(1, int(args.start_page))
    end_page = int(args.end_page) if int(args.end_page) > 0 else num_pages
    end_page = min(end_page, num_pages)
    if start_page > end_page:
        print(f"Invalid range: start_page={start_page} > end_page={end_page}")
        return 1

    session = requests.Session()
    ingest_url = f"{args.url.rstrip('/')}/kb/{args.kb_id}/ingest/text"

    # requests timeout can be (connect, read)
    timeout = (10, int(args.timeout))

    # Iterate 1-based pages, convert to 0-based indices for PyMuPDF.
    p = start_page
    while p <= end_page:
        batch_start = p
        batch_end = min(p + int(args.batch_size) - 1, end_page)
        key = _range_key(batch_start, batch_end)

        if key in progress.get("completed", {}):
            print(f"Skipping pages {key} (already completed)")
            p = batch_end + 1
            continue

        print(f"Processing pages {batch_start} to {batch_end}...")

        try:
            batch_text = "\n\n".join(doc[i - 1].get_text() for i in range(batch_start, batch_end + 1))
        except Exception as e:
            print(f"Error reading pages {key}: {e}")
            progress.setdefault("failed", {})[key] = {"error": f"read_error: {e}"}
            _save_progress(progress_path, progress)
            if args.continue_on_error:
                p = batch_end + 1
                continue
            return 1

        if not batch_text.strip():
            print(f"No text found in pages {key}, skipping.")
            progress.setdefault("completed", {})[key] = {"doc_id": None, "status": "empty"}
            _save_progress(progress_path, progress)
            p = batch_end + 1
            continue

        payload = {
            "title": f"{args.title_prefix} (pages {batch_start}-{batch_end})",
            "text": batch_text,
            "source_type": args.source_type,
            "source_uri": pdf_path,
            "language": args.language,
            "doc_date": args.doc_date,
            "doc_year": args.doc_year,
        }

        last_err = None
        for attempt in range(0, int(args.retries) + 1):
            try:
                resp = session.post(ingest_url, json=payload, timeout=timeout)
                if resp.status_code >= 400:
                    # Server rejected; don't spam retries unless explicitly desired.
                    print(f"Batch {key}: HTTP {resp.status_code}")
                    try:
                        print(resp.json())
                    except Exception:
                        print(resp.text[:500])
                    last_err = f"http_{resp.status_code}"
                    break

                data = resp.json()
                print(f"Batch {key}: Status {resp.status_code}")
                print(data)
                progress.setdefault("completed", {})[key] = {
                    "doc_id": data.get("doc_id"),
                    "status": "ok",
                    "ts": time.time(),
                }
                # If this range previously failed, clear it.
                progress.setdefault("failed", {}).pop(key, None)
                _save_progress(progress_path, progress)
                last_err = None
                break
            except requests.exceptions.ReadTimeout as e:
                last_err = f"read_timeout: {e}"
            except requests.exceptions.ConnectTimeout as e:
                last_err = f"connect_timeout: {e}"
            except requests.exceptions.ConnectionError as e:
                last_err = f"connection_error: {e}"
            except Exception as e:
                last_err = f"error: {e}"

            if attempt < int(args.retries):
                wait = float(args.backoff) * (2 ** attempt)
                print(f"Batch {key}: attempt {attempt+1} failed ({last_err}). Retrying in {wait:.1f}s...")
                time.sleep(wait)

        if last_err:
            print(f"Error processing pages {key}: {last_err}")
            progress.setdefault("failed", {})[key] = {"error": last_err, "ts": time.time()}
            _save_progress(progress_path, progress)
            if not args.continue_on_error:
                print("Stopping due to error. Re-run with --continue-on-error to keep going.")
                return 1

        time.sleep(float(args.sleep_between))
        p = batch_end + 1

    print("Done.")
    print(f"Progress saved to: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())