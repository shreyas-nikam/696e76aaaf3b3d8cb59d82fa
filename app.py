# app.py
# Streamlit multi-page lab app: Week 10 — LangGraph Multi-Agent Orchestration
# Covers: Supervisor pattern, specialist agents, HITL gates, parallel execution, semantic memory, debug traces + Mermaid
# Bloom's taxonomy alignment based on lab preamble. :contentReference[oaicite:1]{index=1}

from __future__ import annotations

import os
import time
import json
import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Tuple

import streamlit as st
import streamlit.components.v1 as components


# =============================================================================
# Page config + session bootstrap
# =============================================================================

st.set_page_config(
    page_title="QuLab: LangGraph Multi-Agent Orchestration",
    layout="wide",
)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""
if "demo_threads" not in st.session_state:
    # thread_id -> state dict
    st.session_state.demo_threads = {}
if "demo_traces" not in st.session_state:
    # trace_id -> AgentTrace object (serialized dict)
    st.session_state.demo_traces = {}
if "demo_memory" not in st.session_state:
    # user_id -> list[memdict]
    st.session_state.demo_memory = {}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = ""


# =============================================================================
# Mermaid rendering helper (Streamlit doesn't natively render Mermaid everywhere)
# =============================================================================

def render_mermaid(diagram: str, height: int = 520) -> None:
    """
    Render a Mermaid diagram using mermaid.js via components.html.
    Provide diagram WITHOUT ```mermaid fences.
    """
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        securityLevel: "loose"
      }});
    </script>
    <div class="mermaid">
{diagram}
    </div>
    """
    components.html(html, height=height, scrolling=True)


def now_utc() -> datetime:
    return datetime.utcnow()


def fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# =============================================================================
# Lab concepts (mirroring the PDF)
# =============================================================================

BloomLevel = Literal["Remember", "Understand",
                     "Apply", "Analyze", "Evaluate", "Create"]

BLOOM_OBJECTIVES: List[Dict[str, str]] = [
    {
        "Bloom Level": "Remember",
        "Objective": "List LangGraph node types and edge conditions",
    },
    {
        "Bloom Level": "Understand",
        "Objective": "Explain the supervisor pattern for agent coordination",
    },
    {
        "Bloom Level": "Apply",
        "Objective": "Implement a multi-agent due diligence workflow",
    },
    {
        "Bloom Level": "Analyze",
        "Objective": "Compare sequential vs parallel agent execution",
    },
    {
        "Bloom Level": "Evaluate",
        "Objective": "Assess HITL breakpoints for risk mitigation",
    },
    {
        "Bloom Level": "Create",
        "Objective": "Design a production agent system with memory",
    },
]

KEY_CONCEPTS = [
    "",
    "Supervisor pattern for agent coordination",
    "Specialist agents (SEC, Talent, Scoring, Value)",
    "Human-in-the-loop (HITL) approval gates",
    "Parallel agent execution",
    "Semantic memory (Mem0-style) with fallback",
    "Agent debug traces + Mermaid visualization",
]


# =============================================================================
# Demo Data Model (in the spirit of the lab state.py)
# =============================================================================

AssessmentType = Literal["screening", "limited", "full"]
ApprovalStatus = Optional[Literal["pending", "approved", "rejected"]]


def make_initial_state(
    company_id: str,
    assessment_type: AssessmentType,
    requested_by: str,
) -> Dict[str, Any]:
    # Mirrors the lab's DueDiligenceState fields conceptually. :contentReference[oaicite:2]{index=2}
    return {
        "company_id": company_id,
        "assessment_type": assessment_type,
        "requested_by": requested_by,
        "messages": [],  # append-only
        "sec_analysis": None,
        "talent_analysis": None,
        "scoring_result": None,
        "value_creation_plan": None,
        "next_agent": None,
        "requires_approval": False,
        "approval_reason": None,
        "approval_status": None,
        "approved_by": None,
        "started_at": now_utc(),
        "completed_at": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }


def add_message(state: Dict[str, Any], role: str, content: str, name: Optional[str] = None) -> None:
    state["messages"].append(
        {
            "role": role,
            "content": content,
            "name": name,
            "timestamp": fmt_dt(now_utc()),
        }
    )


# =============================================================================
# Demo Trace System (inspired by Task 10.5) :contentReference[oaicite:3]{index=3}
# =============================================================================

@dataclass
class TraceStep:
    node: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000.0
        return 0.0


@dataclass
class AgentTrace:
    trace_id: str
    workflow_name: str
    started_at: datetime
    steps: List[TraceStep] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    final_status: str = "running"  # running | completed | awaiting_approval | failed

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def complete(self, status: str = "completed") -> None:
        self.completed_at = now_utc()
        self.final_status = status

    def to_mermaid_state_diagram(self) -> str:
        # Mirrors the idea in traces.py: stateDiagram-v2 :contentReference[oaicite:4]{index=4}
        lines = [
            "stateDiagram-v2",
            "  direction LR",
        ]
        if not self.steps:
            lines.append("  [*] --> Idle")
            lines.append("  Idle --> [*]")
            return "\n".join(lines)

        lines.append(f"  [*] --> {self.steps[0].node}: start")
        for i in range(1, len(self.steps)):
            prev = self.steps[i - 1]
            cur = self.steps[i]
            duration = f"{cur.duration_ms:.0f}ms" if cur.completed_at else "..."
            lines.append(f"  {prev.node} --> {cur.node}: {duration}")

        last = self.steps[-1]
        if self.final_status == "completed":
            lines.append(f"  {last.node} --> [*]: done")
        elif self.final_status == "awaiting_approval":
            lines.append(f"  {last.node} --> HITL: approval needed")
            lines.append("  HITL --> [*]: resume/stop")
        else:
            lines.append(f"  {last.node} --> [*]: {self.final_status}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_status": self.final_status,
            "steps": [
                {
                    "node": s.node,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "total_duration_ms": sum(s.duration_ms for s in self.steps),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentTrace":
        t = AgentTrace(
            trace_id=d["trace_id"],
            workflow_name=d["workflow_name"],
            started_at=datetime.fromisoformat(d["started_at"]),
            completed_at=datetime.fromisoformat(
                d["completed_at"]) if d["completed_at"] else None,
            final_status=d.get("final_status", "running"),
        )
        for s in d.get("steps", []):
            t.steps.append(
                TraceStep(
                    node=s["node"],
                    started_at=datetime.fromisoformat(s["started_at"]),
                    completed_at=datetime.fromisoformat(
                        s["completed_at"]) if s["completed_at"] else None,
                    error=s.get("error"),
                )
            )
        return t


# =============================================================================
# Semantic memory (Mem0-style) with fallback store :contentReference[oaicite:5]{index=5}
# =============================================================================

def memory_add(user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    mem_id = f"mem-{uuid.uuid4().hex[:10]}"
    item = {
        "id": mem_id,
        "content": content,
        "metadata": metadata or {},
        "created_at": now_utc().isoformat(),
    }
    st.session_state.demo_memory.setdefault(user_id, []).append(item)
    return mem_id


def memory_search(user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    query_l = query.lower()
    items = st.session_state.demo_memory.get(user_id, [])
    matches = [m for m in items if query_l in m["content"].lower()]
    return matches[:limit]


# =============================================================================
# Supervisor routing (inspired by Task 10.3) :contentReference[oaicite:6]{index=6}
# =============================================================================

def supervisor_route(state: Dict[str, Any]) -> str:
    # HITL pause check
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        return "wait_for_approval"

    if not state.get("sec_analysis"):
        return "sec_analyst"
    if not state.get("talent_analysis"):
        return "talent_analyst"
    if not state.get("scoring_result"):
        return "scorer"
    if not state.get("value_creation_plan") and state.get("assessment_type") != "screening":
        return "value_creator"
    return "complete"


# =============================================================================
# Specialist simulations (LLM optional; safe stub by default)
# =============================================================================

def have_openai_key() -> bool:
    return bool(st.session_state.OPENAI_API_KEY.strip())


def set_openai_env() -> None:
    if have_openai_key():
        os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY.strip()


async def sim_sec_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    # Simulate async work
    await asyncio.sleep(0.6)
    company_id = state["company_id"]
    return {
        "company_id": company_id,
        "findings": f"Simulated SEC evidence for {company_id} across filings (10-K/10-Q/8-K).",
        "evidence_count": 15,
        "dimensions_covered": ["data_infrastructure", "ai_governance", "technology_stack"],
        "confidence": "medium",
    }


async def sim_talent_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.5)
    company_id = state["company_id"]
    return {
        "company_id": company_id,
        "ai_role_count": 45,
        "talent_concentration": 0.23,
        "seniority_index": 3.2,
        "key_skills": ["pytorch", "mlops", "llm"],
        "hiring_trend": "increasing",
    }


async def sim_scorer(state: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.4)

    # Deterministic-ish score to make the lab feel interactive
    base = 60
    if state.get("talent_analysis"):
        base += int(20 *
                    state["talent_analysis"].get("talent_concentration", 0.2))
    if state.get("sec_analysis"):
        base += min(10,
                    int(state["sec_analysis"].get("evidence_count", 10) / 2))
    # Nudge by assessment depth
    if state["assessment_type"] == "full":
        base += 6
    elif state["assessment_type"] == "screening":
        base -= 4

    score = max(0, min(100, float(base)))
    return {
        "final_score": score,
        "dimension_scores": {
            "data": min(100, score + 2),
            "governance": max(0, score - 3),
            "stack": min(100, score + 1),
            "talent": min(100, score + 4),
        },
        "confidence_interval": [max(0, score - 6), min(100, score + 6)],
    }


async def sim_value_creator(state: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.5)
    current_score = float(state["scoring_result"]["final_score"])
    target_score = min(current_score + 20, 95.0)
    delta = target_score - current_score
    # Simple impact function (demo)
    projected_ebitda_impact_pct = round(2.5 + 0.25 * delta, 2)
    return {
        "current_score": current_score,
        "target_score": target_score,
        "initiatives": [
            {"name": "Data Platform Modernization",
                "impact_pts": 8, "cost_mm": 2.5},
            {"name": "AI Center of Excellence", "impact_pts": 6, "cost_mm": 1.5},
            {"name": "MLOps Implementation", "impact_pts": 5, "cost_mm": 1.0},
        ],
        "projected_ebitda_impact_pct": projected_ebitda_impact_pct,
        "timeline_months": 24,
    }


# =============================================================================
# HITL policy (mirrors lab thresholds conceptually) :contentReference[oaicite:7]{index=7}
# =============================================================================

def needs_hitl_for_score(score: float) -> Tuple[bool, Optional[str]]:
    if score > 85 or score < 40:
        return True, f"Score {score:.1f} outside normal range [40, 85]"
    return False, None


def needs_hitl_for_ebitda_projection(ebitda_impact_pct: float, threshold: float) -> Tuple[bool, Optional[str]]:
    if ebitda_impact_pct > threshold:
        return True, f"EBITDA projection {ebitda_impact_pct:.1f}% exceeds threshold {threshold:.1f}%"
    return False, None


# =============================================================================
# Orchestration engine (demo, but maps directly to the lab concepts)
# =============================================================================

async def run_demo_workflow(
    state: Dict[str, Any],
    *,
    parallel_sec_talent: bool,
    hitl_ebitda_threshold: float,
    trace: AgentTrace,
) -> Dict[str, Any]:
    """
    Runs until completion OR until it hits a HITL pause.
    Uses supervisor_route() repeatedly, similar to a LangGraph StateGraph loop.
    """
    while True:
        next_node = supervisor_route(state)
        state["next_agent"] = next_node

        if next_node == "wait_for_approval":
            trace.complete("awaiting_approval")
            return state

        if next_node == "complete":
            state["completed_at"] = now_utc()
            add_message(state, "assistant",
                        "Due diligence assessment complete.", name="supervisor")
            trace.add_step(TraceStep(node="complete",
                           started_at=now_utc(), completed_at=now_utc()))
            trace.complete("completed")
            return state

        # Parallel execution option: SEC + Talent concurrently
        if next_node == "sec_analyst" and parallel_sec_talent and not state.get("talent_analysis"):
            step = TraceStep(
                node="sec_analyst + talent_analyst (parallel)", started_at=now_utc())
            try:
                sec_res, talent_res = await asyncio.gather(
                    sim_sec_analyst(state),
                    sim_talent_analyst(state),
                )
                state["sec_analysis"] = sec_res
                state["talent_analysis"] = talent_res
                add_message(
                    state, "assistant", "SEC analysis complete (parallel run).", name="sec_analyst")
                add_message(
                    state, "assistant", "Talent analysis complete (parallel run).", name="talent_analyst")
                step.outputs = {"sec_analysis": sec_res,
                                "talent_analysis": talent_res}
            except Exception as e:
                state["error"] = str(e)
                step.error = str(e)
            finally:
                step.completed_at = now_utc()
                trace.add_step(step)

            continue

        # Otherwise, execute node-by-node
        if next_node == "sec_analyst":
            step = TraceStep(node="sec_analyst", started_at=now_utc(), inputs={
                             "company_id": state["company_id"]})
            try:
                sec_res = await sim_sec_analyst(state)
                state["sec_analysis"] = sec_res
                add_message(
                    state,
                    "assistant",
                    f"SEC analysis complete. Evidence count: {sec_res.get('evidence_count')}.",
                    name="sec_analyst",
                )
                step.outputs = {"sec_analysis": sec_res}
            except Exception as e:
                state["error"] = str(e)
                step.error = str(e)
            finally:
                step.completed_at = now_utc()
                trace.add_step(step)
            continue

        if next_node == "talent_analyst":
            step = TraceStep(node="talent_analyst", started_at=now_utc(), inputs={
                             "company_id": state["company_id"]})
            try:
                talent_res = await sim_talent_analyst(state)
                state["talent_analysis"] = talent_res
                add_message(
                    state,
                    "assistant",
                    f"Talent analysis complete. Talent concentration: {talent_res.get('talent_concentration', 0):.0%}.",
                    name="talent_analyst",
                )
                step.outputs = {"talent_analysis": talent_res}
            except Exception as e:
                state["error"] = str(e)
                step.error = str(e)
            finally:
                step.completed_at = now_utc()
                trace.add_step(step)
            continue

        if next_node == "scorer":
            step = TraceStep(node="scorer", started_at=now_utc())
            try:
                scoring = await sim_scorer(state)
                state["scoring_result"] = scoring
                score = float(scoring["final_score"])
                # :contentReference[oaicite:8]{index=8}
                req, reason = needs_hitl_for_score(score)
                state["requires_approval"] = req
                state["approval_reason"] = reason
                state["approval_status"] = "pending" if req else None
                add_message(
                    state,
                    "assistant",
                    f"Scoring complete. Org-AI-R: {score:.1f}" +
                    (f" ⚠️ Requires approval: {reason}" if req else ""),
                    name="scorer",
                )
                step.outputs = {"scoring_result": scoring,
                                "requires_approval": req, "approval_reason": reason}
            except Exception as e:
                state["error"] = str(e)
                step.error = str(e)
            finally:
                step.completed_at = now_utc()
                trace.add_step(step)
            continue

        if next_node == "value_creator":
            step = TraceStep(node="value_creator", started_at=now_utc())
            try:
                plan = await sim_value_creator(state)
                state["value_creation_plan"] = plan
                ebitda = float(plan["projected_ebitda_impact_pct"])
                req, reason = needs_hitl_for_ebitda_projection(
                    ebitda, hitl_ebitda_threshold)
                if req:
                    state["requires_approval"] = True
                    state["approval_reason"] = reason
                    state["approval_status"] = "pending"
                add_message(
                    state,
                    "assistant",
                    f"Value creation plan complete. Projected EBITDA impact: {ebitda:.1f}%"
                    + (" ⚠️ Requires approval" if req else ""),
                    name="value_creator",
                )
                step.outputs = {"value_creation_plan": plan,
                                "requires_approval": req, "approval_reason": reason}
            except Exception as e:
                state["error"] = str(e)
                step.error = str(e)
            finally:
                step.completed_at = now_utc()
                trace.add_step(step)
            continue


def persist_trace(trace: AgentTrace) -> None:
    st.session_state.demo_traces[trace.trace_id] = trace.to_dict()


def load_trace(trace_id: str) -> Optional[AgentTrace]:
    d = st.session_state.demo_traces.get(trace_id)
    if not d:
        return None
    return AgentTrace.from_dict(d)


# =============================================================================
# Sidebar (navigation + OpenAI key input)
# =============================================================================


st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.markdown("**Navigation**")
PAGES = [
    "01 — Lab Overview (Bloom + Concepts)",
    "02 — LangGraph Basics (Nodes + Edges)",
    "03 — State Model (DueDiligenceState)",
    "04 — Supervisor Pattern + Routing",
    "05 — Specialists + Tools",
    "06 — Parallel Execution (Seq vs Parallel)",
    "07 — HITL Approval Gates",
    "08 — Semantic Memory (Mem0-style)",
    "09 — Debug Traces + Mermaid",
    "10 — Run the Demo Workflow",
]
page = st.sidebar.selectbox("Select a page", PAGES, index=0)

st.sidebar.divider()
st.sidebar.markdown("**OpenAI API Key** (stored only in session)")
api_key = st.sidebar.text_input(
    "OPENAI_API_KEY", type="password", value=st.session_state.OPENAI_API_KEY)
st.session_state.OPENAI_API_KEY = api_key
if have_openai_key():
    set_openai_env()
    st.sidebar.success("OpenAI key loaded into session.")
else:
    st.sidebar.info(
        "Add a key to enable any optional LLM calls (demo works without it).")

st.sidebar.divider()
st.sidebar.markdown("**Demo Controls**")
company_id = st.sidebar.text_input("Company ID", value="ACME-TECH")
assessment_type = st.sidebar.selectbox(
    "Assessment Type", ["screening", "limited", "full"], index=1)
requested_by = st.sidebar.text_input("Requested By", value="user")

parallel_mode = st.sidebar.toggle(
    "Parallel SEC + Talent", value=True, help="Demonstrates concurrent specialist execution.")
hitl_ebitda_threshold = st.sidebar.slider(
    "HITL EBITDA Threshold (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.5)

st.sidebar.divider()
st.sidebar.markdown("**Threads**")
if st.sidebar.button("➕ New Thread"):
    tid = f"dd-{company_id}-{uuid.uuid4().hex[:8]}"
    st.session_state.demo_threads[tid] = make_initial_state(
        company_id, assessment_type, requested_by)
    st.session_state.active_thread_id = tid

thread_ids = list(st.session_state.demo_threads.keys())
active_thread = st.sidebar.selectbox("Active thread", [
                                     ""] + thread_ids, index=(1 if st.session_state.active_thread_id in thread_ids else 0))
if active_thread:
    st.session_state.active_thread_id = active_thread

if st.session_state.active_thread_id:
    st.sidebar.caption(f"Active: `{st.session_state.active_thread_id}`")


# =============================================================================
# Page content
# =============================================================================

def page_overview() -> None:

    st.markdown(
        """
This app walks through the Week 10 lab in a **Bloom-ordered** sequence:

**Remember → Understand → Apply → Analyze → Evaluate → Create**

It covers the key concepts: supervisor routing, specialist agents, HITL gates, parallel execution,
semantic memory, and debug traces.
"""
    )

    st.subheader("Bloom’s Taxonomy Objectives (Fixed Progression)")
    st.dataframe(BLOOM_OBJECTIVES, use_container_width=True)

    st.subheader("Key Concepts Covered")
    concepts = "\n- ".join(KEY_CONCEPTS)
    st.markdown(f"{concepts}")

    st.subheader("Core Workflow (Conceptual)")
    render_mermaid(
        """
flowchart LR
  U[User Request] --> S[Supervisor]
  S --> SEC[SEC Specialist]
  S --> TAL[Talent Specialist]
  SEC --> S
  TAL --> S
  S --> SC[Scoring Specialist]
  SC -->|if screening| DONE[Complete]
  SC -->|else| VC[Value Creation Specialist]
  VC --> DONE
  SC -->|HITL Gate| HITL[Human Approval]
  VC -->|HITL Gate| HITL
  HITL --> S
""",
        height=520,
    )

    st.markdown(
        """
**What you’ll build (in the demo):**
- A supervisor loop that chooses the next specialist  
- Specialists producing structured outputs  
- Approval gates that pause execution  
- An optional parallel run for SEC + Talent  
- A trace that renders into Mermaid  
"""
    )


def page_langgraph_basics() -> None:
    st.title("LangGraph Basics: Nodes + Edges (Remember)")
    st.markdown(
        """
In LangGraph-style orchestration, you typically model a workflow as:
- **Nodes**: functions/agents that transform state
- **Edges**: transitions between nodes
- **Conditional edges**: routing based on state (supervisor pattern)

Below is a simplified view of node/edge semantics in Mermaid.
"""
    )

    render_mermaid(
        """
stateDiagram-v2
  direction LR
  [*] --> supervisor
  supervisor --> sec_analyst: if sec_analysis missing
  supervisor --> talent_analyst: if talent_analysis missing
  supervisor --> scorer: if scoring_result missing
  scorer --> HITL: if requires_approval
  HITL --> supervisor: if approved
  scorer --> value_creator: if full/limited
  value_creator --> supervisor
  supervisor --> complete: else
  complete --> [*]
""",
        height=520,
    )

    st.subheader("Demonstration (Code Block)")
    st.code(
        """
def supervisor_route(state):
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        return "wait_for_approval"
    if not state.get("sec_analysis"):
        return "sec_analyst"
    if not state.get("talent_analysis"):
        return "talent_analyst"
    if not state.get("scoring_result"):
        return "scorer"
    if not state.get("value_creation_plan") and state.get("assessment_type") != "screening":
        return "value_creator"
    return "complete"
""".strip(),
        language="python",
    )


def page_state_model() -> None:
    st.title("State Model: DueDiligenceState (Remember → Understand)")
    st.markdown(
        """
A robust orchestration depends on a **well-defined state**: inputs, messages, specialist outputs,
workflow controls, and metadata.

The lab defines a `DueDiligenceState` with fields like `sec_analysis`, `talent_analysis`,
`scoring_result`, `requires_approval`, etc. :contentReference[oaicite:12]{index=12}
"""
    )

    st.subheader("State as a Data Contract")
    render_mermaid(
        """
classDiagram
  class DueDiligenceState {
    +str company_id
    +str assessment_type
    +str requested_by
    +list messages (append-only)
    +dict sec_analysis
    +dict talent_analysis
    +dict scoring_result
    +dict value_creation_plan
    +str next_agent
    +bool requires_approval
    +str approval_reason
    +str approval_status
    +str approved_by
    +datetime started_at
    +datetime completed_at
    +float total_cost_usd
    +str error
  }
""",
        height=520,
    )

    st.subheader("Demonstration (Code Block)")
    st.code(
        """
def make_initial_state(company_id, assessment_type, requested_by):
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
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }
""".strip(),
        language="python",
    )


def page_supervisor() -> None:
    st.title("Supervisor Pattern + Routing (Understand → Apply)")
    st.markdown(
        """
The **supervisor** is responsible for:
- Inspecting the current state
- Selecting the next specialist to run
- Pausing when HITL approval is pending

This is directly aligned with the lab supervisor node logic. :contentReference[oaicite:13]{index=13}
"""
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Supervisor Routing Mermaid")
        render_mermaid(
            """
flowchart TD
  A[Supervisor] --> B{Approval Pending?}
  B -->|Yes| P[Pause: wait_for_approval]
  B -->|No| C{sec_analysis exists?}
  C -->|No| SEC[Run SEC Specialist]
  C -->|Yes| D{talent_analysis exists?}
  D -->|No| TAL[Run Talent Specialist]
  D -->|Yes| E{scoring_result exists?}
  E -->|No| SC[Run Scoring Specialist]
  E -->|Yes| F{value plan needed?}
  F -->|Yes| VC[Run Value Creation Specialist]
  F -->|No| DONE[Complete]
""",
            height=560,
        )

    with col2:
        st.subheader("Why this matters")
        st.markdown(
            """
- Makes orchestration **deterministic and debuggable**
- Encodes business logic (“what comes next?”) in one place
- Enables **resumability** (especially with a checkpointer in real LangGraph)

**Production note:** in the lab, missing a checkpointer breaks HITL pause/resume. :contentReference[oaicite:14]{index=14}
"""
        )

        st.subheader("Demonstration (Routing Function)")
        st.code(
            """
def supervisor_route(state):
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        return "wait_for_approval"
    if not state.get("sec_analysis"):
        return "sec_analyst"
    elif not state.get("talent_analysis"):
        return "talent_analyst"
    elif not state.get("scoring_result"):
        return "scorer"
    elif not state.get("value_creation_plan") and state["assessment_type"] != "screening":
        return "value_creator"
    else:
        return "complete"
""".strip(),
            language="python",
        )


def page_specialists() -> None:
    st.title("Specialist Agents + Tools (Apply)")
    st.markdown(
        """
The lab uses **4 specialists**:
- SEC Analysis
- Talent Analysis
- Scoring
- Value Creation

Agents call tools (search filings, analyze job postings, compute scores, project impact). :contentReference[oaicite:15]{index=15}
"""
    )

    st.subheader("Mermaid: Specialist Decomposition")
    render_mermaid(
        """
flowchart LR
  SEC[SEC Agent] -->|Tool| T1[search_sec_filings]
  TAL[Talent Agent] -->|Tool| T2[analyze_job_postings]
  SC[Scoring Agent] -->|Tool| T3[calculate_dimension_score]
  VC[Value Agent] -->|Tool| T4[project_financial_impact]
""",
        height=420,
    )

    st.subheader("Demonstration (Tool-like Functions)")
    st.code(
        """
def search_sec_filings(company_id: str, query: str, limit: int = 10) -> str:
    return f"Found {limit} SEC filing excerpts for {company_id} matching '{query}'"

def analyze_job_postings(company_id: str) -> str:
    return f"Analyzed job postings for {company_id}: 45 AI-related roles, seniority index 3.2"

def calculate_dimension_score(dimension: str, evidence_summary: str) -> str:
    return f"Dimension {dimension} score: 72 (based on evidence analysis)"
""".strip(),
        language="python",
    )

    st.info("In production, these would wrap real retrievers, scoring calculators, and audited evaluators.")


def page_parallel() -> None:
    st.title("Parallel Execution (Analyze)")
    st.markdown(
        """
The lab explicitly targets **parallel agent execution** and expects a speedup vs sequential runs. :contentReference[oaicite:16]{index=16}

In many due diligence workflows, SEC evidence gathering and talent scanning are independent,
so you can run them concurrently (then merge into scoring).
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sequential")
        render_mermaid(
            """
sequenceDiagram
  participant S as Supervisor
  participant SEC as SEC Agent
  participant TAL as Talent Agent
  participant SC as Scorer
  S->>SEC: run()
  SEC-->>S: sec_analysis
  S->>TAL: run()
  TAL-->>S: talent_analysis
  S->>SC: run()
  SC-->>S: scoring_result
""",
            height=520,
        )

    with col2:
        st.subheader("Parallel")
        render_mermaid(
            """
sequenceDiagram
  participant S as Supervisor
  participant SEC as SEC Agent
  participant TAL as Talent Agent
  participant SC as Scorer
  par SEC + Talent in parallel
    S->>SEC: run()
    S->>TAL: run()
    SEC-->>S: sec_analysis
    TAL-->>S: talent_analysis
  end
  S->>SC: run()
  SC-->>S: scoring_result
""",
            height=520,
        )

    st.subheader("Demonstration (Code Block: asyncio.gather)")
    st.code(
        """
sec_res, talent_res = await asyncio.gather(
    sim_sec_analyst(state),
    sim_talent_analyst(state),
)
state["sec_analysis"] = sec_res
state["talent_analysis"] = talent_res
""".strip(),
        language="python",
    )


def page_hitl() -> None:
    st.title("HITL Approval Gates (Evaluate)")
    st.markdown(
        """
Human-in-the-loop (HITL) gates are used for **high-stakes** decisions.

In the lab supervisor flow:
- If **score > 85 or score < 40**, it requires approval before continuing. :contentReference[oaicite:17]{index=17}
- Value creation can also require approval if projections exceed a threshold (policy-driven).
"""
    )

    render_mermaid(
        """
flowchart TD
  SC[Scorer Output] --> G{Score outside <40, 85>?}
  G -->|Yes| HITL[Pause for Human Approval]
  HITL -->|Approve| CONT[Continue Workflow]
  HITL -->|Reject| STOP[Stop + Record Error]
  G -->|No| CONT
""",
    )

    st.subheader("Policy Snippet")
    st.code(
        """
def needs_hitl_for_score(score: float):
    if score > 85 or score < 40:
        return True, f"Score {score:.1f} outside normal range [40, 85]"
    return False, None
""".strip(),
        language="python",
    )

    st.subheader("Quick Scenario Check")
    s = st.slider("Assessed Score", 0.0, 100.0, 86.0, 0.5)
    req, reason = needs_hitl_for_score(s)
    if req:
        st.warning(f"HITL required: {reason}")
    else:
        st.success("No HITL required; continue automatically.")


def page_memory() -> None:
    st.title("Semantic Memory (Mem0-style) (Create)")
    st.markdown(
        """
The lab adds **semantic memory** so agents can retain long-term context across assessments:
- store outcomes (scores, findings)
- retrieve company context later

When Mem0 isn't available, the lab uses a fallback store. :contentReference[oaicite:18]{index=18}
"""
    )

    render_mermaid(
        """
flowchart LR
  A[New Assessment] --> B[Store Outcome]
  B --> M[(Semantic Memory)]
  Q[Future Query] --> R[Retrieve Context]
  R --> M
  M --> S[Supervisor / Specialists]
""",
        height=170,
    )

    st.subheader("Try the Memory Store (Demo)")
    user_id = st.text_input("User ID", value="system")
    col1, col2 = st.columns([1, 1])
    with col1:
        content = st.text_area(
            "Memory content", value="Company ACME-TECH limited assessment completed. Final score: 78. Key findings: strong talent, moderate governance.")
        if st.button("Add memory"):
            mem_id = memory_add(user_id, content, metadata={
                                "company_id": "ACME-TECH", "assessment_type": "limited"})
            st.success(f"Added memory: {mem_id}")

    with col2:
        q = st.text_input("Search query", value="ACME-TECH")
        if st.button("Search memories"):
            results = memory_search(user_id, q, limit=5)
            st.write(results if results else "No matches.")

    st.subheader("Demonstration (Code Block)")
    st.code(
        """
def memory_add(user_id, content, metadata=None):
    mem_id = f"mem-{uuid.uuid4().hex[:10]}"
    st.session_state.demo_memory.setdefault(user_id, []).append({
        "id": mem_id,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(),
    })
    return mem_id

def memory_search(user_id, query, limit=5):
    q = query.lower()
    items = st.session_state.demo_memory.get(user_id, [])
    matches = [m for m in items if q in m["content"].lower()]
    return matches[:limit]
""".strip(),
        language="python",
    )


def page_traces() -> None:
    st.title("Debug Traces + Mermaid Visualization (Apply → Analyze)")
    st.markdown(
        """
The lab emphasizes **debug traces** with Mermaid diagrams so workflows are debuggable end-to-end. :contentReference[oaicite:19]{index=19}

Below you can view any traces produced by the demo runner.
"""
    )

    trace_ids = list(st.session_state.demo_traces.keys())
    if not trace_ids:
        st.info("No traces yet. Run the demo workflow to generate one.")
        return

    trace_id = st.selectbox("Select trace", trace_ids,
                            index=len(trace_ids) - 1)
    trace = load_trace(trace_id)
    if not trace:
        st.error("Could not load trace.")
        return

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.subheader("Mermaid Trace Diagram")
        render_mermaid(trace.to_mermaid_state_diagram(), height=560)

    with col2:
        st.subheader("Trace Metadata")
        st.markdown(f"- **Trace ID:** `{trace.trace_id}`")
        st.markdown(f"- **Workflow:** `{trace.workflow_name}`")
        st.markdown(f"- **Started:** {fmt_dt(trace.started_at)}")
        st.markdown(f"- **Completed:** {fmt_dt(trace.completed_at)}")
        st.markdown(f"- **Status:** `{trace.final_status}`")

        st.subheader("Steps")
        step_rows = []
        for s in trace.steps:
            step_rows.append(
                {
                    "node": s.node,
                    "duration_ms": round(s.duration_ms, 1),
                    "error": s.error or "",
                }
            )
        st.dataframe(step_rows, use_container_width=True)

        st.subheader("Raw JSON")
        st.code(json.dumps(trace.to_dict(), indent=2), language="json")

    st.divider()
    st.subheader("Common Mistakes (From Lab)")
    st.markdown(
        """
- Not using a checkpointer for HITL pauses (state gets lost). :contentReference[oaicite:20]{index=20}  
- Blocking calls inside async nodes (freezes event loop). :contentReference[oaicite:21]{index=21}  
- Missing approval timeouts (workflow stuck). :contentReference[oaicite:22]{index=22}  
- No try/except around external calls (crashes workflow). :contentReference[oaicite:23]{index=23}  
"""
    )


def page_run_demo() -> None:
    st.title("Run the Demo Workflow (Apply)")
    st.markdown(
        """
This page runs a **LangGraph-like supervisor loop** with:
- Specialist agents
- Optional SEC+Talent parallelism
- HITL pause/resume
- Trace → Mermaid

It’s a faithful *teaching* implementation (no external dependencies required).
"""
    )

    if not st.session_state.active_thread_id:
        st.warning("Create/select a thread in the sidebar first.")
        return

    tid = st.session_state.active_thread_id
    state = st.session_state.demo_threads[tid]

    # Show state summary
    st.subheader("Current State Snapshot")
    snap = {
        "company_id": state["company_id"],
        "assessment_type": state["assessment_type"],
        "next_agent": state.get("next_agent"),
        "requires_approval": state.get("requires_approval"),
        "approval_status": state.get("approval_status"),
        "approval_reason": state.get("approval_reason"),
        "started_at": fmt_dt(state.get("started_at")),
        "completed_at": fmt_dt(state.get("completed_at")),
        "error": state.get("error"),
    }
    st.json(snap)

    # Run / Resume controls
    col_run, col_reset, col_trace = st.columns([0.25, 0.25, 0.5])

    with col_run:
        run_label = "▶️ Run / Continue"
        if st.button(run_label):
            # New trace per run/continue for clarity
            trace = AgentTrace(
                trace_id=f"trace-{uuid.uuid4().hex[:10]}",
                workflow_name="due_diligence_demo",
                started_at=now_utc(),
            )
            # Execute
            updated = asyncio.run(
                run_demo_workflow(
                    state,
                    parallel_sec_talent=parallel_mode,
                    hitl_ebitda_threshold=hitl_ebitda_threshold,
                    trace=trace,
                )
            )
            st.session_state.demo_threads[tid] = updated
            persist_trace(trace)
            st.success(f"Run complete with status: {trace.final_status}")
            st.rerun()

    with col_reset:
        if st.button("♻️ Reset Thread"):
            st.session_state.demo_threads[tid] = make_initial_state(
                company_id=company_id,
                assessment_type=assessment_type,
                requested_by=requested_by,
            )
            st.success("Thread reset.")
            st.rerun()

    with col_trace:
        st.caption(
            "Tip: Visit **Debug Traces + Mermaid** to visualize the execution path.")

    # HITL controls if paused
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        st.warning("Workflow paused for HITL approval.")
        st.markdown(f"**Reason:** {state.get('approval_reason')}")

        approver = st.text_input("Approved by", value=requested_by)
        notes = st.text_input("Notes (optional)", value="")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve and Resume"):
                state["approval_status"] = "approved"
                state["approved_by"] = approver
                state["requires_approval"] = False
                add_message(state, "system", f"Workflow approved by {approver}" + (
                    f": {notes}" if notes else ""), name="hitl")
                st.session_state.demo_threads[tid] = state
                st.success("Approved. Click Run / Continue to resume.")
                st.rerun()

        with c2:
            if st.button("⛔ Reject"):
                state["approval_status"] = "rejected"
                state["approved_by"] = approver
                state["requires_approval"] = False
                state["error"] = f"Rejected by {approver}"
                add_message(state, "system", f"Workflow rejected by {approver}" + (
                    f": {notes}" if notes else ""), name="hitl")
                st.session_state.demo_threads[tid] = state
                st.error("Rejected. Workflow stopped.")
                st.rerun()

    st.divider()
    st.subheader("Messages (Append-only Log)")
    if state["messages"]:
        for m in state["messages"][-20:]:
            st.markdown(
                f"- **[{m['timestamp']}] {m.get('name') or m['role']}**: {m['content']}")
    else:
        st.info("No messages yet. Run the workflow to populate messages.")

    st.divider()
    st.subheader("Outputs")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SEC Analysis**")
        st.json(state.get("sec_analysis") or {})
        st.markdown("**Talent Analysis**")
        st.json(state.get("talent_analysis") or {})
    with c2:
        st.markdown("**Scoring Result**")
        st.json(state.get("scoring_result") or {})
        st.markdown("**Value Creation Plan**")
        st.json(state.get("value_creation_plan") or {})


# =============================================================================
# Router
# =============================================================================


st.title("QuLab: LangGraph Multi-Agent Orchestration")
st.divider()
if page.startswith("01"):
    page_overview()
elif page.startswith("02"):
    page_langgraph_basics()
elif page.startswith("03"):
    page_state_model()
elif page.startswith("04"):
    page_supervisor()
elif page.startswith("05"):
    page_specialists()
elif page.startswith("06"):
    page_parallel()
elif page.startswith("07"):
    page_hitl()
elif page.startswith("08"):
    page_memory()
elif page.startswith("09"):
    page_traces()
elif page.startswith("10"):
    page_run_demo()
else:
    st.error("Unknown page selection.")
st.caption('''
    ---
    ## QuantUniversity License

    © QuantUniversity 2025  
    This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

    - You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
    - You **may not delete or modify this license cell** without authorization.  
    - This notebook was generated using **QuCreate**, an AI-powered assistant.  
    - Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

    All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
    ''')
