from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CITATION_RE = re.compile(r"\[S\d+\]")


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    checks: dict[str, bool]
    answer: str
    snippet_count: int
    elapsed_ms: float
    debug: dict[str, Any] | None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON on {path}:{i}: {e}")
    return cases


def _contains_any(haystack: str, needles: list[str]) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles)


def _contains_all(haystack: str, needles: list[str]) -> bool:
    h = haystack.lower()
    return all(n.lower() in h for n in needles)


def _contains_none(haystack: str, needles: list[str]) -> bool:
    h = haystack.lower()
    return all(n.lower() not in h for n in needles)


def _count_citations(text: str) -> int:
    return len(set(_CITATION_RE.findall(text or "")))


def _run_one(case: dict[str, Any], default_kb_id: str, default_mode: str, debug: bool) -> CaseResult:
    from visitassist_rag.rag.engine import rag_query

    case_id = str(case.get("id") or "") or "(missing-id)"
    kb_id = str(case.get("kb_id") or default_kb_id)
    question = str(case.get("question") or "").strip()
    mode = str(case.get("mode") or default_mode)
    language = str(case.get("language") or "pt")
    expect = case.get("expect") or {}

    if not question:
        return CaseResult(
            case_id=case_id,
            ok=False,
            checks={"has_question": False},
            answer="",
            snippet_count=0,
            elapsed_ms=0.0,
            debug=None,
        )

    t0 = time.perf_counter()
    resp = rag_query(question=question, kb_id=kb_id, mode=mode, language=language, debug=debug)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    answer = getattr(resp, "answer", "") or ""
    snippets = getattr(resp, "snippets", []) or []
    snippet_count = len(snippets)

    checks: dict[str, bool] = {}

    min_citations = int(expect.get("min_citations", 1))
    checks["min_citations"] = _count_citations(answer) >= min_citations

    max_snippets = int(expect.get("max_snippets", 4))
    checks["max_snippets"] = snippet_count <= max_snippets

    must_any = expect.get("must_contain_any") or []
    if must_any:
        checks["must_contain_any"] = _contains_any(answer, list(must_any))

    must_all = expect.get("must_contain_all") or []
    if must_all:
        checks["must_contain_all"] = _contains_all(answer, list(must_all))

    must_not = expect.get("must_not_contain") or []
    if must_not:
        checks["must_not_contain"] = _contains_none(answer, list(must_not))

    ok = all(checks.values()) if checks else True
    debug_obj = getattr(resp, "debug", None) if debug else None

    return CaseResult(
        case_id=case_id,
        ok=ok,
        checks=checks,
        answer=answer,
        snippet_count=snippet_count,
        elapsed_ms=elapsed_ms,
        debug=debug_obj,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight RAG evaluation harness")
    parser.add_argument("--cases", required=True, help="Path to JSONL cases file")
    parser.add_argument("--kb-id", required=True, help="Default kb_id (can be overridden per-case)")
    parser.add_argument("--mode", default="tourist_chat", help="Default mode (can be overridden per-case)")
    parser.add_argument("--debug", action="store_true", help="Include engine debug traces")
    parser.add_argument("--out", default="", help="Optional path to write JSON report")

    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    cases = _load_cases(cases_path)

    results: list[CaseResult] = []
    passed = 0

    for c in cases:
        r = _run_one(c, default_kb_id=args.kb_id, default_mode=args.mode, debug=args.debug)
        results.append(r)
        passed += 1 if r.ok else 0

        status = "PASS" if r.ok else "FAIL"
        checks_str = ", ".join(f"{k}={'ok' if v else 'fail'}" for k, v in r.checks.items())
        print(f"{status}  {r.case_id}  snippets={r.snippet_count}  {r.elapsed_ms:.0f}ms  {checks_str}")
        if not r.ok:
            print("  Answer:")
            print("  " + r.answer.replace("\n", "\n  "))

    total = len(results)
    print(f"\nSummary: {passed}/{total} passed")

    if args.out:
        payload = {
            "cases": str(cases_path),
            "passed": passed,
            "total": total,
            "results": [
                {
                    "id": r.case_id,
                    "ok": r.ok,
                    "checks": r.checks,
                    "snippet_count": r.snippet_count,
                    "elapsed_ms": r.elapsed_ms,
                    "answer": r.answer,
                    "debug": r.debug,
                }
                for r in results
            ],
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
