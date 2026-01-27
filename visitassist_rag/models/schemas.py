from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

class IngestTextRequest(BaseModel):
    title: str
    text: str
    source_type: str = "txt"
    source_uri: str = ""
    language: str = "pt"

class IngestResponse(BaseModel):
    success: bool
    doc_id: Optional[str]
    message: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    language: str = "pt"
    mode: Optional[Literal["tourist_chat", "faq_first", "events", "directory", "coupons"]] = "tourist_chat"
    debug: Optional[bool] = False
    debug_no_filter: Optional[bool] = False
    less_strict: Optional[bool] = False

class Snippet(BaseModel):
    type: str  # Allow any chunk_type (e.g., 'section', 'fine', etc.)
    title: Optional[str]
    text: str
    source: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    snippets: List[Snippet]
    debug: Optional[Dict[str, Any]] = None
