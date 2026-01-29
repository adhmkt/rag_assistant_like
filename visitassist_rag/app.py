from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from visitassist_rag.api.routes_ingest import router as ingest_router
from visitassist_rag.api.routes_query import router as query_router
from visitassist_rag.api.routes_admin import router as admin_router

app = FastAPI(title="VisitAssist RAG Engine")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.get("/ui/test", response_class=HTMLResponse)
def test_ui():
	return """<!doctype html>
<html lang=\"en\">
	<head>
		<meta charset=\"utf-8\" />
		<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
		<title>VisitAssist RAG — Test UI</title>
		<style>
			body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; max-width: 900px; }
			label { display: block; font-weight: 600; margin: 14px 0 6px; }
			input, textarea, select { width: 100%; padding: 10px; font-size: 14px; }
			textarea { min-height: 120px; }
			.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
			button { margin-top: 14px; padding: 10px 14px; font-size: 14px; cursor: pointer; }
			pre { background: #0b1020; color: #e6edf3; padding: 12px; overflow: auto; }
			.hint { color: #555; font-size: 13px; }
		</style>
	</head>
	<body>
		<h1>VisitAssist RAG — Test UI</h1>
		<p class=\"hint\">This page calls <code>/v1/kb/{kb_id}/query/answer</code> and shows the JSON response.</p>

		<div class=\"row\">
			<div>
				<label>kb_id</label>
				<input id=\"kb\" value=\"itaipu\" />
			</div>
			<div>
				<label>language</label>
				<select id=\"lang\">
					<option value=\"pt\" selected>pt</option>
					<option value=\"en\">en</option>
					<option value=\"es\">es</option>
				</select>
			</div>
		</div>

		<div class=\"row\">
			<div>
				<label>mode</label>
				<select id=\"mode\">
					<option value=\"tourist_chat\" selected>tourist_chat</option>
					<option value=\"faq_first\">faq_first</option>
					<option value=\"events\">events</option>
					<option value=\"directory\">directory</option>
					<option value=\"coupons\">coupons</option>
				</select>
			</div>
			<div>
				<label>debug</label>
				<select id=\"debug\">
					<option value=\"false\" selected>false</option>
					<option value=\"true\">true</option>
				</select>
			</div>
		</div>

		<label>question</label>
		<textarea id=\"q\">Qual é o total de instrumentos e o total de drenos citados no documento?</textarea>

		<button id=\"run\">Run query</button>

		<h2>Response</h2>
		<pre id=\"out\">(click “Run query”)</pre>

		<script>
			const out = document.getElementById('out');
			document.getElementById('run').addEventListener('click', async () => {
				out.textContent = 'Loading…';
				const kb = document.getElementById('kb').value.trim();
				const payload = {
					question: document.getElementById('q').value,
					language: document.getElementById('lang').value,
					mode: document.getElementById('mode').value,
					debug: document.getElementById('debug').value === 'true'
				};
				try {
					const res = await fetch(`/v1/kb/${encodeURIComponent(kb)}/query/answer`, {
						method: 'POST',
						headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
						body: JSON.stringify(payload)
					});
					const text = await res.text();
					out.textContent = `HTTP ${res.status}\n\n${text}`;
				} catch (e) {
					out.textContent = String(e);
				}
			});
		</script>
	</body>
</html>"""

app.include_router(ingest_router, prefix="/v1")
app.include_router(query_router, prefix="/v1")
app.include_router(admin_router, prefix="/v1")
