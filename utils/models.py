from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LocationContext:
    """
    Explicit location state for a chat session.

    Tracks the confirmed state/county that the conversation is about.
    Updated by LocationStateManager; passed through the agent pipeline so
    every LLM call has a canonical location without re-running fuzzy search.
    """
    state: Optional[str] = None        # e.g. "California"
    county: Optional[str] = None       # e.g. "Los Angeles County, California"
    grain: Optional[str] = None        # "state" | "county"
    source: str = "none"               # "explicit" | "high_fuzzy" | "user_confirmed"
    confirmed: bool = False

    def is_set(self) -> bool:
        return bool(self.state or self.county)

    def as_annotation(self) -> str:
        """Compact inline annotation appended to a question string."""
        parts: list = []
        if self.county:
            parts.append(f"{self.county} (county)")
        elif self.state:
            parts.append(f"{self.state} (state)")
        return f"  [location context: {', '.join(parts)}]" if parts else ""

    def as_prompt_block(self) -> str:
        """Multi-line block for injection into LLM system/user prompts."""
        if not self.is_set():
            return ""
        parts: list = []
        if self.state:
            parts.append(f"state={self.state}")
        if self.county:
            parts.append(f"county={self.county}")
        if self.grain:
            parts.append(f"grain={self.grain}")
        return f"[ACTIVE LOCATION: {', '.join(parts)}]\n"


@dataclass
class AgentStatus:
    ok: bool
    status: str
    message: str


@dataclass
class RouteResult:
    route: str  # "qa_agent" | "conversational_agent" | "exception_agent"
    reason: str
    intent: Optional[str] = None


@dataclass
class TableSelection:
    ok: bool
    tables: List[str] = field(default_factory=list)
    grain: Optional[str] = None
    filters_needed: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class SqlGenerationResult:
    ok: bool
    sql: Optional[str] = None
    message: str = ""


@dataclass
class SqlValidationResult:
    ok: bool
    message: str
    rewritten_sql: Optional[str] = None


@dataclass
class QueryExecutionResult:
    ok: bool
    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    message: str = ""


@dataclass
class FinalAnswer:
    ok: bool
    status: str
    user_message: str
    route: str
    source_tables: List[str] = field(default_factory=list)
    sql: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    data: List[Dict[str, Any]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
