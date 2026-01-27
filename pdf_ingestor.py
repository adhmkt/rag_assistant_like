import fitz  # PyMuPDF
import requests

pdf_path = "C:\\rag-project\\rag_assistant_like\\visitassist_rag\\itaipu.pdf"
batch_size = 50  # Number of pages per batch

try:
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"Opened PDF: {pdf_path} with {num_pages} pages.")
except Exception as e:
    print(f"Error opening PDF: {e}")
    exit(1)

for start in range(0, num_pages, batch_size):
    end = min(start + batch_size, num_pages)
    print(f"Processing pages {start+1} to {end}...")
    try:
        batch_text = "\n\n".join(doc[i].get_text() for i in range(start, end))
        if not batch_text.strip():
            print(f"No text found in pages {start+1}-{end}, skipping.")
            continue
        resp = requests.post(
            "http://localhost:8000/ingest/text",
            json={
                "title": f"Bublia Itaipu (pages {start+1}-{end})",
                "text": batch_text,
                "source_type": "pdf",
                "source_uri": pdf_path,
                "kb_id": "itaipu",
                "language": "pt"
            },
            timeout=120
        )
        print(f"Batch {start+1}-{end}: Status {resp.status_code}")
        print(resp.json())
    except Exception as e:
        print(f"Error processing pages {start+1}-{end}: {e}")