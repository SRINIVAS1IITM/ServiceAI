from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None # For potential future context management
    # Could add more context like company_name_known, etc. if sales flow becomes multi-turn

class KBArticle(BaseModel):
    id: str
    topic: str
    answer: str
    score: Optional[float] = None

class AgentResponse(BaseModel):
    message: str
    intent: str
    escalate: bool
    escalation_reason: Optional[str] = None
    kb_article_found: Optional[KBArticle] = None
    debug_info: Optional[dict] = None