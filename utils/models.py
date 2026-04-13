from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
