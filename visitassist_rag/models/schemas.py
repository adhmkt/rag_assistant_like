from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

class IngestTextRequest(BaseModel):
    title: str
    text: str
    source_type: str = "txt"
    source_uri: str = ""
    language: str = "pt"
    doc_date: Optional[str] = None
    doc_year: Optional[int] = None

class IngestResponse(BaseModel):
    success: bool
    doc_id: Optional[str]
    message: Optional[str] = None


class IngestUrlPreviewRequest(BaseModel):
    url: str
    language: str = "pt"


class IngestUrlPreviewResponse(BaseModel):
    url: str
    final_url: str
    title: str
    text_preview: str
    content_hash: str
    char_count: int
    word_count: int
    link_count: int
    link_density: float
    warnings: List[str] = []


class IngestUrlConfirmRequest(BaseModel):
    url: str
    title: Optional[str] = None
    language: str = "pt"
    doc_date: Optional[str] = None
    doc_year: Optional[int] = None


class IngestUrlPasteRequest(BaseModel):
    url: str
    text: str
    title: Optional[str] = None
    language: str = "pt"
    doc_date: Optional[str] = None
    doc_year: Optional[int] = None

class QueryRequest(BaseModel):
    question: str
    language: str = "pt"
    mode: Optional[str] = "tourist_chat"
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


class AnswerOnlyResponse(BaseModel):
    answer: str
