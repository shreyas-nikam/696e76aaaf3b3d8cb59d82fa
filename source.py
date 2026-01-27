# source.py
"""
Function-driven LangGraph Due Diligence Workflow (Streamlit-ready)

Changes vs your notebook-style source:
1) OpenAI key is provided by the caller (e.g., Streamlit user input) — no os.getenv / dotenv.
2) Uses OpenAI (ChatOpenAI) everywhere — no mock LLM/agents.
3) Replaces mock tools with real tools:
   - SEC filings retrieval via SEC EDGAR public endpoints (requires a proper User-Agent header).
   - Dimension scoring via OpenAI structured output.
   - Financial impact projection via real math.
   - Job postings analysis: pluggable provider (SerpAPI optional) + fallback HTML scrape of a provided careers URL.
     (There is no universal public “job postings API” for arbitrary companies; this is the most practical “actual tool”
      pattern for production: provide a provider or a URL.)

This module is designed to be imported by Streamlit app.py:
- You create a graph with create_due_diligence_graph(...)
- You run it with run_due_diligence(...)
- You can handle HITL with approve_workflow(...)

NOTE:
- SEC endpoints require a real User-Agent string with contact info per SEC guidance.
  Example: "SynergyCapital/1.0 (sarah@synergycapital.com)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator
import functools
import json
import re
import math
import time

import structlog
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------
# Agent Memory (simple implementation + verification hook for app.py)
# ---------------------------------------------------------------------


class AgentMemory:
    """
    Lightweight semantic memory abstraction.
    You can later swap this with Mem0 or another vector DB.
    """

    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata or {}
        memory_id = f"mem-{int(datetime.utcnow().timestamp())}"
        self._store.setdefault(user_id, []).append({
            "id": memory_id,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        })
        return memory_id

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        items = self._store.get(user_id, [])
        q = query.lower()
        results = [
            m for m in items
            if q in m["content"].lower()
            or any(q in str(v).lower() for v in m["metadata"].values())
        ]
        return results[:limit]

    async def get_company_context(
        self,
        company_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        memories = await self.search_memories(company_id, user_id=user_id, limit=20)
        return {
            "company_id": company_id,
            "memory_count": len(memories),
            "memories": memories,
        }

    async def store_assessment_outcome(
        self,
        company_id: str,
        assessment_type: str,
        final_score: float,
        key_findings: List[str],
        user_id: str = "system",
    ) -> None:
        content = f"""Company {company_id} {assessment_type} assessment completed.
Final Org-AI-R score: {final_score:.1f}
Key findings:
{chr(10).join(f'- {f}' for f in key_findings)}
"""
        metadata = {
            "company_id": company_id,
            "assessment_type": assessment_type,
            "final_score": final_score,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.add_memory(content=content, user_id=user_id, metadata=metadata)


# Global instance (used by app.py)
agent_memory = AgentMemory()


# ---------------------------------------------------------------------
# verify_agent_memory (required by app.py)
# ---------------------------------------------------------------------
async def verify_agent_memory():
    """
    Sanity-check that agent_memory is usable.
    Called once by app.py at startup.
    """
    test_user_id = "system_test"
    test_company_id = "TestCorp Inc."

    await agent_memory.add_memory(
        content="Test memory for TestCorp Inc.",
        user_id=test_user_id,
        metadata={"company_id": test_company_id, "purpose": "startup_check"},
    )

    results = await agent_memory.search_memories(
        query="TestCorp",
        user_id=test_user_id,
        limit=5,
    )

    logger.info(
        "agent_memory_verified",
        test_company_id=test_company_id,
        results_count=len(results),
    )

    return True

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


class AgentMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str]
    timestamp: datetime


class DueDiligenceState(TypedDict):
    # Input
    company_id: str
    assessment_type: Literal["screening", "limited", "full"]
    requested_by: str

    # Messages (append-only)
    messages: Annotated[List[AgentMessage], operator.add]

    # Agent outputs
    sec_analysis: Optional[Dict[str, Any]]
    talent_analysis: Optional[Dict[str, Any]]
    scoring_result: Optional[Dict[str, Any]]
    value_creation_plan: Optional[Dict[str, Any]]

    # Workflow control
    next_agent: Optional[str]
    requires_approval: bool
    approval_reason: Optional[str]
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    approved_by: Optional[str]

    # Tracing
    trace_id: Optional[str]

    # Metadata
    started_at: datetime
    completed_at: Optional[datetime]
    total_tokens: int
    total_cost_usd: float
    error: Optional[str]


# -----------------------------------------------------------------------------
# Settings (Streamlit can pass these into create_due_diligence_graph)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowSettings:
    hitl_ebitda_projection_threshold: float = 0.07  # 7%
    score_min: float = 40.0
    score_max: float = 85.0
    openai_model: str = "gpt-4o-2024-08-06"


# -----------------------------------------------------------------------------
# Trace utilities
# -----------------------------------------------------------------------------
@dataclass
class TraceStep:
    node: str
    started_at: datetime
    completed_at: datetime = field(default_factory=datetime.utcnow)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at).total_seconds() * 1000.0


@dataclass
class AgentTrace:
    trace_id: str
    workflow_name: str
    started_at: datetime
    steps: List[TraceStep] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    final_status: str = "running"

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def complete(self, status: str = "completed") -> None:
        self.completed_at = datetime.utcnow()
        self.final_status = status

    def to_mermaid(self) -> str:
        lines = [
            "```mermaid",
            "stateDiagram-v2",
            "    direction LR",
        ]
        for i, step in enumerate(self.steps):
            dur = f"{step.duration_ms:.0f}ms"
            if i == 0:
                lines.append(f"    [*] --> {step.node}: start")
            else:
                prev = self.steps[i - 1]
                lines.append(f"    {prev.node} --> {step.node}: {dur}")
        if self.steps:
            last = self.steps[-1].node
            if self.final_status == "completed":
                lines.append(f"    {last} --> [*]: done")
            elif self.final_status == "awaiting_approval":
                lines.append(f"    {last} --> HITL: approval needed")
            elif self.final_status == "rejected":
                lines.append(f"    {last} --> [*]: rejected")
        lines.append("```")
        return "\n".join(lines)


class TraceManager:
    def __init__(self) -> None:
        self._active: Dict[str, AgentTrace] = {}
        self._completed: List[AgentTrace] = []

    def start_trace(self, trace_id: str, workflow_name: str) -> AgentTrace:
        trace = AgentTrace(
            trace_id=trace_id, workflow_name=workflow_name, started_at=datetime.utcnow())
        self._active[trace_id] = trace
        logger.info("trace_started", trace_id=trace_id, workflow=workflow_name)
        return trace

    def get_trace(self, trace_id: str) -> Optional[AgentTrace]:
        return self._active.get(trace_id) or next((t for t in self._completed if t.trace_id == trace_id), None)

    def complete_trace(self, trace_id: str, status: str = "completed") -> Optional[AgentTrace]:
        trace = self._active.pop(trace_id, None)
        if trace:
            trace.complete(status=status)
            self._completed.append(trace)
            logger.info("trace_completed", trace_id=trace_id, status=status)
        return trace


# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------
def create_openai_llm(openai_api_key: str, model: str, temperature: float) -> ChatOpenAI:
    if not openai_api_key or not openai_api_key.strip():
        raise ValueError(
            "openai_api_key is required (pass from Streamlit user input).")
    return ChatOpenAI(model=model, temperature=temperature, api_key=openai_api_key)


# -----------------------------------------------------------------------------
# SEC EDGAR utilities (public endpoints)
# -----------------------------------------------------------------------------
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _requests_session(user_agent: str) -> requests.Session:
    if not user_agent or "@" not in user_agent:
        # SEC strongly prefers UA with contact; enforce a basic guard.
        raise ValueError(
            "SEC requests require a proper User-Agent with contact info "
            '(e.g., "SynergyCapital/1.0 (sarah@synergycapital.com)").'
        )
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
    )
    return s


def _safe_get_json(sess: requests.Session, url: str, timeout: int = 30) -> Dict[str, Any]:
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def resolve_company_to_cik(company: str, user_agent: str) -> Optional[str]:
    """
    Resolve a company name (or ticker) to a CIK using SEC's company_tickers.json.
    Best-effort: matches by ticker exact or name substring.
    Returns CIK as string without leading zeros (e.g., "320193").
    """
    sess = _requests_session(user_agent)
    data = _safe_get_json(sess, SEC_TICKER_CIK_URL)

    q = (company or "").strip().lower()
    if not q:
        return None

    # Data is a dict of numeric keys -> {cik_str, ticker, title}
    # Try ticker exact first
    for _, row in data.items():
        ticker = str(row.get("ticker", "")).lower()
        if ticker and ticker == q:
            return str(row.get("cik_str"))

    # Try name substring match
    candidates = []
    for _, row in data.items():
        title = str(row.get("title", "")).lower()
        if q in title:
            candidates.append(row)

    if not candidates:
        return None

    # Pick the shortest title match (often the most direct)
    best = sorted(candidates, key=lambda r: len(str(r.get("title", ""))))[0]
    return str(best.get("cik_str"))


def fetch_recent_filings(cik: str, user_agent: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch recent filings metadata from submissions endpoint.
    Returns list of dicts: {form, filingDate, accessionNumber, primaryDocument}
    """
    sess = _requests_session(user_agent)
    cik10 = str(cik).zfill(10)
    sub = _safe_get_json(sess, SEC_SUBMISSIONS_URL.format(cik10=cik10))

    recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accessions = recent.get("accessionNumber", []) or []
    primdocs = recent.get("primaryDocument", []) or []

    out: List[Dict[str, Any]] = []
    for i in range(min(limit, len(forms), len(dates), len(accessions), len(primdocs))):
        out.append(
            {
                "form": forms[i],
                "filingDate": dates[i],
                "accessionNumber": accessions[i],
                "primaryDocument": primdocs[i],
            }
        )
    return out


def fetch_filing_text(cik: str, accession_number: str, primary_document: str, user_agent: str, max_chars: int = 200_000) -> str:
    """
    Download the primary document content from SEC Archives and return text (best-effort).
    Many filings are HTML; we return raw text with tags stripped lightly.
    """
    sess = _requests_session(user_agent)
    cik_nolead = str(int(cik))
    acc_nodash = accession_number.replace("-", "")
    url = f"{SEC_ARCHIVES_BASE}/{cik_nolead}/{acc_nodash}/{primary_document}"
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    content = r.text

    # Very light HTML tag stripping (avoid adding heavy deps)
    content = re.sub(r"(?is)<script.*?>.*?</script>", " ", content)
    content = re.sub(r"(?is)<style.*?>.*?</style>", " ", content)
    content = re.sub(r"(?s)<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content).strip()

    return content[:max_chars]


# -----------------------------------------------------------------------------
# "Actual tools" (LangChain tools)
# -----------------------------------------------------------------------------
def make_sec_search_tool(user_agent: str):
    @tool
    def search_sec_filings(company_id: str, query: str, limit: int = 3) -> str:
        """
        Search SEC filings for evidence by:
        1) resolving company -> CIK
        2) pulling recent filings metadata
        3) downloading primary docs and extracting snippets around query terms

        Returns a compact JSON string with matched snippets and filing references.
        """
        cik = resolve_company_to_cik(company_id, user_agent=user_agent)
        if not cik:
            return json.dumps(
                {"company_id": company_id,
                    "error": "CIK not found (try ticker or a more exact legal name)."},
                ensure_ascii=False,
            )

        filings = fetch_recent_filings(
            cik=cik, user_agent=user_agent, limit=max(limit, 1))
        if not filings:
            return json.dumps({"company_id": company_id, "cik": cik, "matches": []}, ensure_ascii=False)

        q = (query or "").strip()
        q_low = q.lower()

        matches = []
        for f in filings:
            try:
                text = fetch_filing_text(
                    cik=cik,
                    accession_number=f["accessionNumber"],
                    primary_document=f["primaryDocument"],
                    user_agent=user_agent,
                    max_chars=180_000,
                )
                text_low = text.lower()
                idx = text_low.find(q_low) if q_low else -1
                if idx != -1:
                    start = max(0, idx - 400)
                    end = min(len(text), idx + 600)
                    snippet = text[start:end]
                    matches.append(
                        {
                            "form": f["form"],
                            "filingDate": f["filingDate"],
                            "accessionNumber": f["accessionNumber"],
                            "primaryDocument": f["primaryDocument"],
                            "snippet": snippet,
                        }
                    )
            except Exception as e:
                matches.append(
                    {
                        "form": f.get("form"),
                        "filingDate": f.get("filingDate"),
                        "accessionNumber": f.get("accessionNumber"),
                        "primaryDocument": f.get("primaryDocument"),
                        "error": str(e),
                    }
                )

        return json.dumps(
            {"company_id": company_id, "cik": cik,
                "query": query, "matches": matches[:limit]},
            ensure_ascii=False,
        )

    return search_sec_filings


def make_job_postings_tool(
    serpapi_api_key: Optional[str] = None,
    default_careers_url: Optional[str] = None,
):
    @tool
    def analyze_job_postings(company_id: str, careers_url: Optional[str] = None, limit: int = 20) -> str:
        """
        Analyze job postings using one of:
        - SerpAPI (if serpapi_api_key provided): query Google Jobs results (best for arbitrary companies).
        - Fallback: scrape a provided careers_url (or default_careers_url) and extract role titles.

        Returns JSON string with role titles and lightweight AI-signal counts.
        """
        url = careers_url or default_careers_url
        roles: List[str] = []

        # Option A: SerpAPI (recommended for production)
        if serpapi_api_key:
            try:
                import requests as _rq  # keep local alias

                params = {
                    "engine": "google_jobs",
                    "q": f"{company_id} careers AI machine learning",
                    "api_key": serpapi_api_key,
                }
                r = _rq.get("https://serpapi.com/search.json",
                            params=params, timeout=45)
                r.raise_for_status()
                data = r.json()
                jobs = data.get("jobs_results", []) or []
                for j in jobs[:limit]:
                    title = j.get("title")
                    if title:
                        roles.append(title)
            except Exception as e:
                return json.dumps({"company_id": company_id, "error": f"SerpAPI failed: {e}"}, ensure_ascii=False)

        # Option B: scrape careers URL
        elif url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                html = r.text
                # naive title extraction
                # (in production, use BeautifulSoup; keeping deps minimal here)
                candidates = re.findall(r"(?i)>([^<]{3,80})<", html)
                # keep likely job-title-ish strings
                for c in candidates:
                    t = re.sub(r"\s+", " ", c).strip()
                    if any(k in t.lower() for k in ["engineer", "scientist", "analyst", "ml", "ai", "data", "product", "platform"]):
                        roles.append(t)
                roles = roles[:limit]
            except Exception as e:
                return json.dumps(
                    {"company_id": company_id, "careers_url": url,
                        "error": f"Scrape failed: {e}"},
                    ensure_ascii=False,
                )
        else:
            return json.dumps(
                {
                    "company_id": company_id,
                    "error": "No job postings provider configured. Provide serpapi_api_key or careers_url.",
                },
                ensure_ascii=False,
            )

        ai_keywords = ["ai", "machine learning", "ml", "llm",
                       "pytorch", "tensorflow", "mlops", "data scientist"]
        ai_hits = 0
        senior_hits = 0
        for t in roles:
            tl = t.lower()
            if any(k in tl for k in ai_keywords):
                ai_hits += 1
            if any(k in tl for k in ["senior", "staff", "principal", "lead", "director", "vp", "head"]):
                senior_hits += 1

        talent_concentration = (ai_hits / max(len(roles), 1)) if roles else 0.0
        seniority_index = (senior_hits / max(len(roles), 1)) * \
            5.0 if roles else 0.0  # scaled 0..5

        return json.dumps(
            {
                "company_id": company_id,
                "careers_url": url,
                "role_count": len(roles),
                "roles_sample": roles[:10],
                "ai_role_count": ai_hits,
                "talent_concentration": round(talent_concentration, 3),
                "seniority_index": round(seniority_index, 2),
            },
            ensure_ascii=False,
        )

    return analyze_job_postings


@tool
def project_financial_impact(entry_score: float, target_score: float, h_r_score: float) -> str:
    """
    Project EBITDA impact from AI improvements based on current and target scores.
    Simplified linear model (deterministic).
    """
    delta = float(target_score) - float(entry_score)
    impact = 0.0025 + 0.0005 * delta + 0.00025 * \
        delta * float(h_r_score) / 100.0
    return json.dumps({"projected_ebitda_impact_pct": impact, "model": "linear_v2_demo"}, ensure_ascii=False)


# -----------------------------------------------------------------------------
# OpenAI-backed scoring helpers (replaces mock dimension scoring + mock OrgAIR)
# -----------------------------------------------------------------------------
DIMENSIONS = [
    "data_infrastructure",
    "ai_governance",
    "technology_stack",
    "talent_and_org",
    "delivery_and_mlops",
    "security_and_risk",
    "business_value_realization",
]


def _openai_json(llm: ChatOpenAI, system: str, user: str) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", user)])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({})


async def _openai_json_async(llm: ChatOpenAI, system: str, user: str) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", user)])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return await chain.ainvoke({})


def compute_org_air_score(dimension_scores: Dict[str, float], talent_concentration: float, hr_baseline: float, evidence_count: int) -> Dict[str, Any]:
    """
    Deterministic, transparent scoring function (no proprietary “mock calculator”).
    You can swap this with your PE Org-AI-R formula later.
    """
    dims = [float(dimension_scores.get(d, 0.0)) for d in DIMENSIONS]
    avg_dim = sum(dims) / max(len(dims), 1)

    # Example weighting (adjust freely)
    score = (
        0.65 * avg_dim
        + 0.20 * (float(talent_concentration) * 100.0)
        + 0.10 * float(hr_baseline)
        + 0.05 * min(float(evidence_count), 50.0) * 2.0  # caps at +5
    )
    score = max(0.0, min(100.0, score))
    return {
        "final_score": round(score, 1),
        "avg_dimension_score": round(avg_dim, 1),
        "dimension_scores": {k: round(float(v), 1) for k, v in dimension_scores.items()},
        "inputs": {
            "talent_concentration": talent_concentration,
            "hr_baseline": hr_baseline,
            "evidence_count": evidence_count,
        },
        "model": "transparent_org_air_demo_v1",
    }


# -----------------------------------------------------------------------------
# Agents (function-driven, constructed with factories)
# -----------------------------------------------------------------------------
class SECAnalysisAgent:
    SYSTEM_PROMPT = """You are an expert at analyzing SEC filings (10-K, 10-Q, 8-K, DEF-14A) for AI-readiness indicators for private equity due diligence.
Return STRICT JSON with:
{
  "findings": "concise narrative",
  "evidence": [{"dimension": "...", "quote_or_snippet": "...", "filing_ref": "..."}],
  "risks": ["..."],
  "opportunities": ["..."],
  "confidence": "low|medium|high"
}
"""

    def __init__(self, llm: ChatOpenAI, sec_tool):
        self.llm = llm
        self.sec_tool = sec_tool

    async def analyze(self, company_id: str, assessment_type: str) -> Dict[str, Any]:
        # pull raw snippets first (tool)
        raw = self.sec_tool.invoke(
            {"company_id": company_id, "query": "artificial intelligence OR machine learning OR data platform", "limit": 3})
        system = self.SYSTEM_PROMPT
        user = f"""
Company: {company_id}
Assessment type: {assessment_type}

SEC tool output (JSON string):
{raw}

Synthesize SEC evidence into the required JSON.
Focus on concrete indicators for the 7 dimensions: {DIMENSIONS}.
"""
        out = await _openai_json_async(self.llm, system=system, user=user)
        # basic normalization
        evidence = out.get("evidence") or []
        evidence_count = len(evidence)
        dims_covered = sorted({(e.get("dimension") or "").strip()
                              for e in evidence if e.get("dimension")})
        return {
            "company_id": company_id,
            "findings": out.get("findings", ""),
            "evidence": evidence,
            "evidence_count": evidence_count,
            "dimensions_covered": dims_covered,
            "risks": out.get("risks") or [],
            "opportunities": out.get("opportunities") or [],
            "confidence": out.get("confidence", "medium"),
            "raw_tool": raw,
        }


class TalentAnalysisAgent:
    SYSTEM_PROMPT = """You are an expert at analyzing hiring patterns and talent pools for AI capability.
Return STRICT JSON:
{
  "ai_role_count": number,
  "talent_concentration": number (0..1),
  "seniority_index": number (0..5),
  "key_skills": ["..."],
  "hiring_trend": "decreasing|stable|increasing",
  "notes": "short"
}
"""

    def __init__(self, llm: ChatOpenAI, jobs_tool):
        self.llm = llm
        self.jobs_tool = jobs_tool

    async def analyze(self, company_id: str, careers_url: Optional[str] = None) -> Dict[str, Any]:
        raw = self.jobs_tool.invoke(
            {"company_id": company_id, "careers_url": careers_url, "limit": 20})
        system = self.SYSTEM_PROMPT
        user = f"""
Company: {company_id}

Job postings tool output (JSON string):
{raw}

Infer hiring trend + key skills. If tool has only titles, extract likely skills from titles conservatively.
"""
        out = await _openai_json_async(self.llm, system=system, user=user)
        # merge with tool metrics when present
        try:
            tool_json = json.loads(raw)
        except Exception:
            tool_json = {}

        return {
            "company_id": company_id,
            "ai_role_count": int(out.get("ai_role_count") or tool_json.get("ai_role_count") or 0),
            "talent_concentration": float(out.get("talent_concentration") or tool_json.get("talent_concentration") or 0.0),
            "seniority_index": float(out.get("seniority_index") or tool_json.get("seniority_index") or 0.0),
            "key_skills": out.get("key_skills") or [],
            "hiring_trend": out.get("hiring_trend") or "stable",
            "notes": out.get("notes") or "",
            "raw_tool": raw,
        }


class ScoringAgent:
    SYSTEM_PROMPT = """You are scoring AI-readiness for PE due diligence.
Return STRICT JSON:
{
  "dimension_scores": {
    "data_infrastructure": 0-100,
    "ai_governance": 0-100,
    "technology_stack": 0-100,
    "talent_and_org": 0-100,
    "delivery_and_mlops": 0-100,
    "security_and_risk": 0-100,
    "business_value_realization": 0-100
  },
  "rationale": { "<dimension>": "1-2 sentences", ... },
  "confidence": "low|medium|high"
}
"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def score(self, sec_analysis: Dict[str, Any], talent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        system = self.SYSTEM_PROMPT
        user = f"""
SEC analysis (JSON):
{json.dumps(sec_analysis, ensure_ascii=False)}

Talent analysis (JSON):
{json.dumps(talent_analysis, ensure_ascii=False)}

Score all 7 dimensions. Use evidence-based scoring; if uncertain, pick conservative midpoints.
"""
        out = await _openai_json_async(self.llm, system=system, user=user)
        dim_scores = out.get("dimension_scores") or {}
        # ensure all dims exist
        for d in DIMENSIONS:
            if d not in dim_scores:
                dim_scores[d] = 60.0
        # clamp
        dim_scores = {k: max(0.0, min(100.0, float(v)))
                      for k, v in dim_scores.items()}
        return {
            "dimension_scores": dim_scores,
            "rationale": out.get("rationale") or {},
            "confidence": out.get("confidence") or "medium",
        }


class ValueCreationAgent:
    SYSTEM_PROMPT = """You are an AI value creation specialist for private equity.
Return STRICT JSON:
{
  "initiatives": [{"name":"...", "why":"...", "impact_points":0-15, "cost_mm": number, "timeline_months": number}],
  "target_score": 0-100,
  "hr_score": 0-100,
  "notes": "short"
}
"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def plan(self, company_id: str, current_score: float, talent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        system = self.SYSTEM_PROMPT
        user = f"""
Company: {company_id}
Current Org-AI-R score: {current_score}
Talent analysis:
{json.dumps(talent_analysis, ensure_ascii=False)}

Create a practical value creation plan. Keep it PE-executable.
"""
        out = await _openai_json_async(self.llm, system=system, user=user)
        initiatives = out.get("initiatives") or []
        target_score = float(out.get("target_score")
                             or min(current_score + 20.0, 95.0))
        hr_score = float(out.get("hr_score") or 70.0)

        # deterministic projection
        proj = json.loads(project_financial_impact.invoke(
            {"entry_score": current_score, "target_score": target_score, "h_r_score": hr_score}))
        projected = float(proj.get("projected_ebitda_impact_pct") or 0.0)

        return {
            "company_id": company_id,
            "current_score": current_score,
            "target_score": target_score,
            "hr_score": hr_score,
            "initiatives": initiatives,
            "projected_ebitda_impact_pct": projected,
            "timeline_months": int(max([int(i.get("timeline_months", 0)) for i in initiatives] + [0])),
            "notes": out.get("notes") or "",
        }


# -----------------------------------------------------------------------------
# Graph node factories (so everything is function-driven, no globals)
# -----------------------------------------------------------------------------
async def supervisor_node(state: DueDiligenceState) -> Dict[str, Any]:
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        return {"next_agent": "wait_for_approval"}

    if not state.get("sec_analysis"):
        return {"next_agent": "sec_analyst"}
    if not state.get("talent_analysis"):
        return {"next_agent": "talent_analyst"}
    if not state.get("scoring_result"):
        return {"next_agent": "scorer"}
    if not state.get("value_creation_plan") and state["assessment_type"] != "screening":
        return {"next_agent": "value_creator"}
    return {"next_agent": "complete"}


def route_from_supervisor(state: DueDiligenceState) -> str:
    nxt = state.get("next_agent") or "complete"
    if nxt == "wait_for_approval":
        return END
    return nxt


def make_sec_analyst_node(sec_agent: SECAnalysisAgent, trace_manager: TraceManager):
    async def node(state: DueDiligenceState) -> Dict[str, Any]:
        node_name = "sec_analyst"
        trace_id = state.get("trace_id")
        started_at = datetime.utcnow()
        err = None
        out: Dict[str, Any] = {}
        try:
            analysis = await sec_agent.analyze(state["company_id"], state["assessment_type"])
            out = {"sec_analysis": analysis}
        except Exception as e:
            err = str(e)
            out = {"sec_analysis": {
                "company_id": state["company_id"], "error": err, "evidence_count": 0, "dimensions_covered": []}}

        if trace_id and (tr := trace_manager.get_trace(trace_id)):
            tr.add_step(
                TraceStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    inputs={
                        "company_id": state["company_id"], "assessment_type": state["assessment_type"]},
                    outputs=out,
                    error=err,
                )
            )

        msg = "SEC analysis complete." if not err else f"SEC analysis failed: {err}"
        return {
            **out,
            "messages": [
                AgentMessage(role="assistant", content=msg,
                             name="sec_analyst", timestamp=datetime.utcnow())
            ],
        }

    return node


def make_talent_analyst_node(talent_agent: TalentAnalysisAgent, trace_manager: TraceManager, careers_url: Optional[str]):
    async def node(state: DueDiligenceState) -> Dict[str, Any]:
        node_name = "talent_analyst"
        trace_id = state.get("trace_id")
        started_at = datetime.utcnow()
        err = None
        out: Dict[str, Any] = {}
        try:
            analysis = await talent_agent.analyze(state["company_id"], careers_url=careers_url)
            out = {"talent_analysis": analysis}
        except Exception as e:
            err = str(e)
            out = {"talent_analysis": {
                "company_id": state["company_id"], "error": err, "talent_concentration": 0.0, "seniority_index": 0.0}}

        if trace_id and (tr := trace_manager.get_trace(trace_id)):
            tr.add_step(
                TraceStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    inputs={
                        "company_id": state["company_id"], "careers_url": careers_url},
                    outputs=out,
                    error=err,
                )
            )

        tc = float(out.get("talent_analysis", {}).get(
            "talent_concentration", 0.0) or 0.0)
        msg = f"Talent analysis complete. Talent concentration: {tc:.0%}." if not err else f"Talent analysis failed: {err}"
        return {
            **out,
            "messages": [
                AgentMessage(role="assistant", content=msg,
                             name="talent_analyst", timestamp=datetime.utcnow())
            ],
        }

    return node


def make_scorer_node(scoring_agent: ScoringAgent, trace_manager: TraceManager, settings: WorkflowSettings):
    async def node(state: DueDiligenceState) -> Dict[str, Any]:
        node_name = "scorer"
        trace_id = state.get("trace_id")
        started_at = datetime.utcnow()
        err = None

        requires_approval = False
        approval_reason = None

        out: Dict[str, Any] = {}
        try:
            sec = state.get("sec_analysis") or {}
            tal = state.get("talent_analysis") or {}
            scored = await scoring_agent.score(sec, tal)

            # compute final score deterministically
            org = compute_org_air_score(
                dimension_scores=scored["dimension_scores"],
                talent_concentration=float(
                    tal.get("talent_concentration") or 0.0),
                hr_baseline=85.0,
                evidence_count=int(sec.get("evidence_count") or 0),
            )
            out = {"scoring_result": {**org, **scored}}

            score = float(out["scoring_result"]["final_score"])
            if score < settings.score_min or score > settings.score_max:
                requires_approval = True
                approval_reason = f"Org-AI-R score {score:.1f} outside normal range [{settings.score_min:.0f}, {settings.score_max:.0f}]"

        except Exception as e:
            err = str(e)
            out = {"scoring_result": {"final_score": 0.0, "error": err}}

        if trace_id and (tr := trace_manager.get_trace(trace_id)):
            tr.add_step(
                TraceStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    inputs={"sec_analysis_exists": bool(state.get(
                        "sec_analysis")), "talent_analysis_exists": bool(state.get("talent_analysis"))},
                    outputs={**out, "requires_approval": requires_approval,
                             "approval_reason": approval_reason},
                    error=err,
                )
            )

        msg = f"Scoring complete. Org-AI-R: {float(out.get('scoring_result', {}).get('final_score', 0.0)):.1f}"
        if requires_approval:
            msg += f" — Requires approval: {approval_reason}"
        if err:
            msg = f"Scoring failed: {err}"

        return {
            **out,
            "requires_approval": requires_approval,
            "approval_reason": approval_reason,
            "approval_status": "pending" if requires_approval else None,
            "messages": [AgentMessage(role="assistant", content=msg, name="scorer", timestamp=datetime.utcnow())],
        }

    return node


def make_value_creator_node(value_agent: ValueCreationAgent, trace_manager: TraceManager, settings: WorkflowSettings):
    async def node(state: DueDiligenceState) -> Dict[str, Any]:
        node_name = "value_creator"
        trace_id = state.get("trace_id")
        started_at = datetime.utcnow()
        err = None

        requires_approval = False
        approval_reason = None

        out: Dict[str, Any] = {}
        try:
            company_id = state["company_id"]
            current_score = float(
                (state.get("scoring_result") or {}).get("final_score") or 60.0)
            tal = state.get("talent_analysis") or {}
            plan = await value_agent.plan(company_id=company_id, current_score=current_score, talent_analysis=tal)
            out = {"value_creation_plan": plan}

            ebitda = float(plan.get("projected_ebitda_impact_pct") or 0.0)
            if ebitda > settings.hitl_ebitda_projection_threshold:
                requires_approval = True
                approval_reason = (
                    f"EBITDA projection {ebitda:.1%} exceeds threshold "
                    f"({settings.hitl_ebitda_projection_threshold:.1%})"
                )

        except Exception as e:
            err = str(e)
            out = {"value_creation_plan": {"error": err,
                                           "projected_ebitda_impact_pct": 0.0, "initiatives": []}}

        if trace_id and (tr := trace_manager.get_trace(trace_id)):
            tr.add_step(
                TraceStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    inputs={"current_score": (
                        state.get("scoring_result") or {}).get("final_score")},
                    outputs={**out, "requires_approval": requires_approval,
                             "approval_reason": approval_reason},
                    error=err,
                )
            )

        ebitda = float(out.get("value_creation_plan", {}).get(
            "projected_ebitda_impact_pct", 0.0) or 0.0)
        msg = f"Value creation plan complete. Projected EBITDA impact: {ebitda:.1%}"
        if requires_approval:
            msg += " — Requires approval"
        if err:
            msg = f"Value creation failed: {err}"

        existing_approval = state.get("approval_status")
        final_approval_status = "pending" if requires_approval else existing_approval

        return {
            **out,
            "requires_approval": requires_approval,
            "approval_reason": approval_reason,
            "approval_status": final_approval_status,
            "messages": [AgentMessage(role="assistant", content=msg, name="value_creator", timestamp=datetime.utcnow())],
        }

    return node


def make_complete_node(trace_manager: TraceManager):
    async def node(state: DueDiligenceState) -> Dict[str, Any]:
        node_name = "complete"
        trace_id = state.get("trace_id")
        started_at = datetime.utcnow()

        if trace_id and (tr := trace_manager.get_trace(trace_id)):
            tr.add_step(
                TraceStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    inputs={"final_score": (
                        state.get("scoring_result") or {}).get("final_score")},
                    outputs={"status": "completed"},
                )
            )
            trace_manager.complete_trace(trace_id, status="completed")

        return {
            "completed_at": datetime.utcnow(),
            "messages": [AgentMessage(role="assistant", content="Due diligence assessment complete.", name="supervisor", timestamp=datetime.utcnow())],
        }

    return node


# -----------------------------------------------------------------------------
# Public API: build graph + run + approve
# -----------------------------------------------------------------------------
def create_due_diligence_graph(
    *,
    openai_api_key: str,
    sec_user_agent: str,
    settings: Optional[WorkflowSettings] = None,
    trace_manager: Optional[TraceManager] = None,
    serpapi_api_key: Optional[str] = None,
    careers_url: Optional[str] = None,
):
    """
    Build and compile the LangGraph workflow.

    Streamlit usage:
      graph, trace_mgr = create_due_diligence_graph(openai_api_key=key, sec_user_agent=ua, ...)
    """
    settings = settings or WorkflowSettings()
    trace_manager = trace_manager or TraceManager()

    # tools
    sec_tool = make_sec_search_tool(user_agent=sec_user_agent)
    jobs_tool = make_job_postings_tool(
        serpapi_api_key=serpapi_api_key, default_careers_url=careers_url)

    # llms
    llm_sec = create_openai_llm(
        openai_api_key, model=settings.openai_model, temperature=0.3)
    llm_talent = create_openai_llm(
        openai_api_key, model=settings.openai_model, temperature=0.2)
    llm_score = create_openai_llm(
        openai_api_key, model=settings.openai_model, temperature=0.1)
    llm_value = create_openai_llm(
        openai_api_key, model=settings.openai_model, temperature=0.3)

    # agents
    sec_agent = SECAnalysisAgent(llm=llm_sec, sec_tool=sec_tool)
    talent_agent = TalentAnalysisAgent(llm=llm_talent, jobs_tool=jobs_tool)
    scoring_agent = ScoringAgent(llm=llm_score)
    value_agent = ValueCreationAgent(llm=llm_value)

    # graph
    workflow = StateGraph(DueDiligenceState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("sec_analyst", make_sec_analyst_node(
        sec_agent, trace_manager))
    workflow.add_node("talent_analyst", make_talent_analyst_node(
        talent_agent, trace_manager, careers_url))
    workflow.add_node("scorer", make_scorer_node(
        scoring_agent, trace_manager, settings))
    workflow.add_node("value_creator", make_value_creator_node(
        value_agent, trace_manager, settings))
    workflow.add_node("complete", make_complete_node(trace_manager))

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sec_analyst": "sec_analyst",
            "talent_analyst": "talent_analyst",
            "scorer": "scorer",
            "value_creator": "value_creator",
            "complete": "complete",
            END: END,
        },
    )

    workflow.add_edge("sec_analyst", "supervisor")
    workflow.add_edge("talent_analyst", "supervisor")
    workflow.add_edge("scorer", "supervisor")
    workflow.add_edge("value_creator", "supervisor")
    workflow.add_edge("complete", END)

    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)
    return compiled, trace_manager


async def run_due_diligence(
    *,
    graph,
    trace_manager: TraceManager,
    company_id: str,
    assessment_type: Literal["screening", "limited", "full"] = "limited",
    requested_by: str = "system",
    thread_id: Optional[str] = None,
) -> DueDiligenceState:
    """
    Execute workflow until completion OR HITL pause (approval_status == "pending").
    """
    if not thread_id:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", company_id).strip("_").lower()
        thread_id = f"dd-{safe}-{datetime.utcnow().isoformat(timespec='seconds')}"

    initial_state: DueDiligenceState = {
        "company_id": company_id,
        "assessment_type": assessment_type,
        "requested_by": requested_by,
        "messages": [],
        "sec_analysis": None,
        "talent_analysis": None,
        "scoring_result": None,
        "value_creation_plan": None,
        "next_agent": None,
        "requires_approval": False,
        "approval_reason": None,
        "approval_status": None,
        "approved_by": None,
        "trace_id": thread_id,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    trace_manager.start_trace(thread_id, f"Due Diligence for {company_id}")
    final_state = await graph.ainvoke(initial_state, config)

    # if not pending, mark complete
    if final_state.get("approval_status") != "pending":
        trace_manager.complete_trace(
            thread_id, status=final_state.get("approval_status") or "completed")

    return final_state


async def approve_workflow(
    *,
    graph,
    trace_manager: TraceManager,
    thread_id: str,
    approved_by: str,
    decision: Literal["approved", "rejected"],
    notes: Optional[str] = None,
) -> DueDiligenceState:
    """
    Resume a paused workflow after HITL decision.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if not state:
        raise ValueError(
            f"No active workflow found for thread_id: {thread_id}")

    update = {
        "approval_status": decision,
        "approved_by": approved_by,
        "requires_approval": False,
        "messages": [
            AgentMessage(
                role="system",
                content=f"Workflow {decision} by {approved_by}" +
                (f": {notes}" if notes else ""),
                name="hitl",
                timestamp=datetime.utcnow(),
            )
        ],
    }

    if decision == "approved":
        final_state = await graph.ainvoke({**state.values, **update}, config)
        trace_manager.complete_trace(thread_id, status="completed")
        return final_state

    # rejected
    trace_manager.complete_trace(thread_id, status="rejected")
    return {**state.values, **update, "error": f"Workflow rejected by {approved_by}. Reason: {notes or 'No reason provided'}."}


# -----------------------------------------------------------------------------
# Convenience: initial state factory (optional)
# -----------------------------------------------------------------------------
def make_initial_state(
    company_id: str,
    assessment_type: Literal["screening", "limited", "full"],
    requested_by: str,
    thread_id: str,
) -> DueDiligenceState:
    return {
        "company_id": company_id,
        "assessment_type": assessment_type,
        "requested_by": requested_by,
        "messages": [],
        "sec_analysis": None,
        "talent_analysis": None,
        "scoring_result": None,
        "value_creation_plan": None,
        "next_agent": None,
        "requires_approval": False,
        "approval_reason": None,
        "approval_status": None,
        "approved_by": None,
        "trace_id": thread_id,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }
