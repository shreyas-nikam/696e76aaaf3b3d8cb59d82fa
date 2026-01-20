
# Building an Automated Due Diligence Workflow with LangGraph for Private Equity

## Introduction: Empowering PE Analysts with AI-Driven Insights

**Persona:** Sarah, a Software Developer at "Synergy Capital," a forward-thinking Private Equity (PE) firm.  
**Organization:** Synergy Capital, specializing in acquiring and growing technology companies.

Sarah's team is tasked with modernizing Synergy Capital's initial due diligence process. Currently, PE analysts spend significant time manually sifting through financial filings, talent reports, and market data to assess potential target companies. This is time-consuming and prone to inconsistencies, delaying critical investment decisions.

The goal is to develop a proof-of-concept for an AI-powered multi-agent system that automates the initial assessment of a target company's AI-readiness. This system will leverage LangGraph to orchestrate specialized AI agents (e.g., SEC Analysis, Talent Analysis, Scoring, Value Creation), providing a structured, consolidated report to analysts. This allows analysts to quickly grasp a company's profile and focus their expertise on high-value strategic considerations, significantly accelerating the deal pipeline.

This notebook will guide Sarah through the creation of this foundational LangGraph application, demonstrating how to define the workflow state, build specialist agents, orchestrate them with a supervisor agent, integrate human-in-the-loop (HITL) approvals, add semantic memory with Mem0, and visualize agent traces for debugging.

## 1. Setup and Dependencies

Before we dive into building the multi-agent system, we need to install the necessary libraries and import them. This ensures our environment has all the tools required for LangGraph orchestration, LLM interactions, memory management, and tracing.

### Install Required Libraries

```python
!pip install -qU langgraph langchain-openai langchain-anthropic mem0 structlog python-dotenv typer
```

### Import Required Dependencies

```python
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, Callable
from datetime import datetime
import operator
import functools
import json
import os
import structlog
from dataclasses import dataclass, field
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables (e.g., API keys)
load_dotenv()

# Configure structlog for consistent logging
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()
```

## 2. Defining the Due Diligence Workflow State

Sarah begins by defining the shared state that will govern the entire due diligence workflow. In LangGraph, the `TypedDict` acts as the single source of truth, holding all the information that agents need to access and update. This centralized state ensures consistency and enables complex conditional routing.

The `DueDiligenceState` includes inputs like `company_id` and `assessment_type`, append-only `messages` for conversation history, fields to store outputs from each specialist agent, workflow control flags (`next_agent`, `requires_approval`, `approval_status`), and metadata for tracking execution.

```python
class AgentMessage(TypedDict):
    """Message in agent conversation."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str]
    timestamp: datetime

class DueDiligenceState(TypedDict):
    """State for due diligence workflow."""
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

# Mock configuration settings, typically loaded from a config file or environment
class MockSettings:
    HITL_EBITDA_PROJECTION_THRESHOLD: float = 0.07 # 7% EBITDA projection requires HITL approval

settings = MockSettings()

# Example initial state
initial_company_state: DueDiligenceState = {
    "company_id": "TechInnovate Inc.",
    "assessment_type": "full",
    "requested_by": "Sarah (Software Developer)",
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
    "trace_id": None, # Will be set at runtime
    "started_at": datetime.utcnow(),
    "completed_at": None,
    "total_tokens": 0,
    "total_cost_usd": 0.0,
    "error": None,
}
print(f"Initial state defined for company: {initial_company_state['company_id']}")
```

## 3. Crafting Specialist Agents and Their Tools

Sarah now defines the specialist agents that perform discrete analytical tasks. Each agent is designed with a specific expertise, represented by its `SYSTEM_PROMPT`, and equipped with a set of tools to interact with simulated external systems (e.g., searching SEC filings, analyzing job postings). This modular design allows for clear separation of concerns and easy extensibility.

### Mock External Tools

These tools simulate interactions with external data sources or calculation services. For a real-world scenario, these would connect to actual databases, APIs, or data analysis pipelines.

```python
# MOCK TOOLS FOR AGENTS
@tool
def search_sec_filings(company_id: str, query: str, limit: int = 10) -> str:
    """Search SEC filings for AI-readiness evidence related to a company.
    Simplified - in production, this would use a robust retriever."""
    if "AI" in query.lower() or "artificial intelligence" in query.lower():
        return f"Found {limit} SEC filing excerpts for {company_id} matching '{query}': Mentions of 'AI strategy' in 10-K section 1A, 'machine learning investments' in 10-Q exhibit 99.1."
    return f"Found {limit} general SEC filing excerpts for {company_id} matching '{query}'."

@tool
def analyze_job_postings(company_id: str) -> str:
    """Analyze job postings for AI talent signals for a specific company.
    Simplified - in production, this would scrape and analyze actual job boards."""
    if "techinnovate" in company_id.lower():
        return f"Analyzed job postings for {company_id}: 45 AI-related roles, seniority index 3.2 (high), key skills: ['PyTorch', 'MLOps', 'LLM'], hiring trend: 'increasing'."
    return f"Analyzed job postings for {company_id}: 10 AI-related roles, seniority index 1.5 (medium), key skills: ['Python', 'SQL'], hiring trend: 'stable'."

@tool
def calculate_dimension_score(dimension: str, evidence_summary: str) -> str:
    """Calculate a score (0-100) for a specific AI-readiness dimension based on provided evidence."""
    # Mock scoring logic
    if "data_infrastructure" in dimension.lower() and "AI strategy" in evidence_summary:
        return f"Dimension {dimension} score: 85 (based on strong evidence of data infrastructure supporting AI strategy)"
    return f"Dimension {dimension} score: 72 (based on general evidence analysis)"

@tool
def project_financial_impact(
    entry_score: float,
    target_score: float,
    h_r_score: float
) -> str:
    """Project EBITDA impact from AI improvements based on current and target scores.
    This uses a simplified linear model for demonstration."""
    delta = target_score - entry_score
    # The impact calculation uses coefficients derived from a hypothetical PE value creation model.
    # The formula is: Base Impact + (Impact per score point * Delta) + (HR Factor * Delta * HR Score)
    impact = 0.0025 + 0.0005 * delta + 0.00025 * delta * h_r_score / 100
    return f"Projected EBITDA impact: {impact:.1%} (base case)"

# Mock Org-AI-R Calculator (simulating a proprietary scoring model)
class MockOrgAIRCalculator:
    def calculate(self, company_id: str, sector_id: str, dimension_scores: List[int], talent_concentration: float, hr_baseline: int, position_factor: float, evidence_count: int) -> Any:
        # Simplified mock calculation, designed to sometimes trigger HITL for demonstration
        avg_score = sum(dimension_scores) / len(dimension_scores)
        
        # Adjust score based on company_id for HITL demonstration
        if company_id == "RiskyBet Corp.":
            final_score = (avg_score * 0.4) + (talent_concentration * 100 * 0.2) + (hr_baseline * 0.1) # Lower score
        elif company_id == "StarPerformer Inc.":
            final_score = (avg_score * 0.6) + (talent_concentration * 100 * 0.4) + (hr_baseline * 0.3) # Higher score
        else:
            final_score = (avg_score * 0.5) + (talent_concentration * 100 * 0.3) + (hr_baseline * 0.2)
            
        final_score = min(max(final_score, 0), 100) # Ensure score is 0-100

        class MockResult:
            def __init__(self, score):
                self.final_score = round(score, 1) # Round for consistent output
            def to_dict(self):
                return {"final_score": self.final_score, "details": "mock Org-AI-R calculation based on simplified model", "dimensions_scores": dimension_scores}
        return MockResult(final_score)

org_air_calculator = MockOrgAIRCalculator()
```

### Specialist Agent Definitions

Each agent is a Python class containing its LLM instance (e.g., `ChatOpenAI`, `ChatAnthropic`) and a list of tools it can use. The `analyze` or `calculate` method defines its core logic.

```python
# SPECIALIST AGENT DEFINITIONS
class SECAnalysisAgent:
    """Agent specialized in SEC filing analysis for AI-readiness indicators."""
    SYSTEM_PROMPT = """You are an expert at analyzing SEC filings (10-K, 10-Q, 8-K, DEF-14A) for AI-readiness indicators for private equity due diligence.
    Your task is to:
    1. Search relevant SEC filings.
    2. Extract evidence for AI-readiness dimensions (e.g., data infrastructure, AI governance, technology stack).
    3. Identify risks and opportunities mentioned in filings regarding AI.
    4. Summarize findings with confidence levels, citing specific filing sections.
    Be thorough but concise, focusing on quantifiable and actionable insights for a PE firm."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.3)
        self.tools = [search_sec_filings]
        self.llm_with_tools = self.llm.bind_tools(self.tools) # Bind tools once

    async def analyze(self, state: DueDiligenceState) -> Dict[str, Any]:
        """Perform SEC analysis based on company ID and assessment type."""
        company_id = state["company_id"]
        assessment_type = state["assessment_type"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", f"Analyze SEC filings for company {company_id}. Assessment type: {assessment_type}. Focus on AI-related strategic mentions, investments, and risks."),
        ])
        
        response = await self.llm_with_tools.ainvoke(prompt.format_messages())
        
        # Mocking the content extraction for the purpose of this notebook
        findings_content = f"Initial SEC findings for {company_id} based on {assessment_type} assessment. Identified key evidence regarding 'AI strategy' in annual reports, 'R&D investment in ML' in quarterly reports, and 'data governance initiatives' in proxy statements. Found 15 distinct pieces of evidence."
        
        logger.info("sec_analysis_complete", company_id=company_id)
        
        return {
            "sec_analysis": {
                "company_id": company_id,
                "findings": findings_content,
                "evidence_count": 15,
                "dimensions_covered": ["data_infrastructure", "ai_governance", "technology_stack"],
                "confidence": "medium",
            }
        }

class TalentAnalysisAgent:
    """Agent specialized in talent and hiring analysis for AI capability."""
    SYSTEM_PROMPT = """You are an expert at analyzing hiring patterns and talent pools for AI capability, providing insights for private equity due diligence.
    Your task is to:
    1. Analyze job postings for AI-related roles.
    2. Calculate talent concentration and seniority metrics specific to AI/ML.
    3. Identify skill gaps and hiring trends in AI.
    4. Assess leadership AI commitment based on talent data.
    Focus on quantifiable metrics and provide a concise summary."""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-3.5-20240620", temperature=0.3) # Using Claude for diversity
        self.tools = [analyze_job_postings]
        self.llm_with_tools = self.llm.bind_tools(self.tools) # Bind tools once

    async def analyze(self, state: DueDiligenceState) -> Dict[str, Any]:
        """Perform talent analysis based on company ID."""
        company_id = state["company_id"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", f"Analyze job postings for AI talent signals for company {company_id}."),
        ])
        
        response = await self.llm_with_tools.ainvoke(prompt.format_messages())
        
        # Simulate tool use response to get specific numbers
        tool_output = analyze_job_postings(company_id)
        ai_roles_count = 45 if "TechInnovate" in company_id else 10 # Mock based on company
        talent_concentration = 0.23 if "TechInnovate" in company_id else 0.10
        seniority_index = 3.2 if "TechInnovate" in company_id else 1.5

        logger.info("talent_analysis_complete", company_id=company_id)

        return {
            "talent_analysis": {
                "company_id": company_id,
                "ai_role_count": ai_roles_count,
                "talent_concentration": talent_concentration,
                "seniority_index": seniority_index,
                "key_skills": ["PyTorch", "MLOps", "LLM"],
                "hiring_trend": "increasing",
            }
        }

class ScoringAgent:
    """Agent for calculating Org-AI-R scores using a proprietary model."""
    SYSTEM_PROMPT = """You are an expert at calculating AI-readiness scores (Org-AI-R).
    Your task is to:
    1. Synthesize evidence from SEC and talent analyses provided.
    2. Score each of the seven dimensions (0-100) based on the synthesized evidence.
    3. Calculate the full Org-AI-R score using the v2.0 formula.
    4. Generate confidence intervals and show your reasoning for the overall score."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.2)
        self.tools = [calculate_dimension_score]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def calculate(self, state: DueDiligenceState) -> Dict[str, Any]:
        """Calculate Org-AI-R score, synthesizing SEC and talent analysis."""
        company_id = state["company_id"]
        
        # Extract data from state for scoring
        sec_analysis_data = state.get("sec_analysis", {})
        talent_analysis_data = state.get("talent_analysis", {})

        # Mock dimension scores based on provided analysis (in a real scenario, LLM would help derive these)
        dimension_scores = [70, 65, 75, 68, 72, 60, 70] # Example scores
        talent_concentration = talent_analysis_data.get("talent_concentration", 0.2)
        
        # Use the actual (mock) calculator
        result = org_air_calculator.calculate(
            company_id=company_id,
            sector_id="technology",
            dimension_scores=dimension_scores,
            talent_concentration=talent_concentration,
            hr_baseline=85,
            position_factor=0.1,
            evidence_count=sec_analysis_data.get("evidence_count", 0),
        )
        
        logger.info("scoring_complete", company_id=company_id, score=result.final_score)
        
        return {
            "scoring_result": result.to_dict()
        }

class ValueCreationAgent:
    """Agent for creating AI value creation plans for private equity."""
    SYSTEM_PROMPT = """You are an expert at AI value creation for private equity, focusing on actionable, measurable recommendations.
    Your task is to:
    1. Identify highest-impact AI improvement areas based on the Org-AI-R score and previous analyses.
    2. Design specific AI initiatives (e.g., Data Platform Modernization, AI Center of Excellence).
    3. Project EBITDA impact using v2.0 parameters.
    4. Create an implementation roadmap with estimated costs and timelines."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.4)
        self.tools = [project_financial_impact]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def plan(self, state: DueDiligenceState) -> Dict[str, Any]:
        """Create a value creation plan based on scoring results."""
        company_id = state["company_id"]
        current_score = state.get("scoring_result", {}).get("final_score", 60)
        
        # Mocking target score and HR score for financial impact projection
        target_score = min(current_score + 20, 95)
        hr_score = state.get("talent_analysis", {}).get("seniority_index", 2.5) * 20 # Convert index to score

        # Use the project_financial_impact tool
        impact_projection_str = project_financial_impact(
            entry_score=current_score,
            target_score=target_score,
            h_r_score=hr_score
        )
        
        # Extract the percentage value from the mock tool's output string
        # Example: "Projected EBITDA impact: 7.5% (base case)"
        try:
            projected_ebitda_impact_pct = float(impact_projection_str.split(":")[1].split("%")[0].strip()) / 100
        except:
            projected_ebitda_impact_pct = 0.05 # Default if parsing fails

        logger.info("value_creation_complete", company_id=company_id)

        return {
            "value_creation_plan": {
                "company_id": company_id,
                "current_score": current_score,
                "target_score": target_score,
                "initiatives": [
                    {"name": "Data Platform Modernization", "impact": 8, "cost_mm": 2.5},
                    {"name": "AI Center of Excellence", "impact": 6, "cost_mm": 1.5},
                    {"name": "MLOps Implementation", "impact": 5, "cost_mm": 1.0},
                ],
                "projected_ebitda_impact_pct": projected_ebitda_impact_pct,
                "timeline_months": 24,
            }
        }

# Initialize specialist agents
sec_agent = SECAnalysisAgent()
talent_agent = TalentAnalysisAgent()
scoring_agent = ScoringAgent()
value_agent = ValueCreationAgent()
```

## 4. Building the Supervisor and Graph with HITL

This is the core orchestration layer. Sarah uses LangGraph's `StateGraph` to define the workflow, where a central `supervisor_node` directs traffic between the specialist agents. Critically, Human-in-the-Loop (HITL) approval gates are integrated to pause the workflow for manual review, especially for high-stakes decisions like unusual Org-AI-R scores or significant EBITDA projections.

### Understanding HITL Conditions

The workflow will pause for human approval if:
1.  **Org-AI-R Score Deviation:** The calculated `Org-AI-R score` $S$ falls outside a predefined normal range $[S_{min}, S_{max}]$. For this lab, we define $S_{min}=40$ and $S_{max}=85$. The condition for triggering HITL is:
    $$(S < S_{min}) \lor (S > S_{max})$$
    where $\lor$ denotes the logical OR operator. This ensures that exceptionally low or high scores (which might indicate an outlier or a miscalculation) are flagged for review by a human analyst.
2.  **High Projected EBITDA Impact:** The `projected_ebitda_impact_pct` $I$ from the value creation plan exceeds a specified threshold. For this lab, the threshold is dynamically loaded from `settings.HITL_EBITDA_PROJECTION_THRESHOLD`. The condition for triggering HITL is:
    $$I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD}$$
    This allows analysts to scrutinize plans with very optimistic (or pessimistic) financial projections.

### Agent Nodes and Routing Logic

```python
# MOCK TRACE MANAGER (to be fully defined in section 6)
# Temporarily define it here so the graph can be built, then fully implement it later.
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
        return (self.completed_at - self.started_at).total_seconds() * 1000 if self.completed_at and self.started_at else 0

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
            status_indicator = "✓" if not step.error else "x"
            duration_str = f"{step.duration_ms:.0f}ms" if step.completed_at else "..."
            
            if i == 0:
                lines.append(f"    [*] --> {step.node}: start")
            else:
                prev = self.steps[i-1]
                lines.append(f"    {prev.node} --> {step.node}: {duration_str}")
        
        if self.steps:
            last_node = self.steps[-1].node
            if self.final_status == "completed":
                lines.append(f"    {last_node} --> [*]: done")
            elif self.final_status == "awaiting_approval":
                lines.append(f"    {last_node} --> HITL: approval needed")
            elif self.final_status == "rejected":
                lines.append(f"    {last_node} --> [*]: rejected")

        lines.append("```")
        return "\n".join(lines)

class TraceManager:
    def __init__(self):
        self._active_traces: Dict[str, AgentTrace] = {}
        self._completed_traces: List[AgentTrace] = []

    def start_trace(self, trace_id: str, workflow_name: str) -> AgentTrace:
        trace = AgentTrace(trace_id=trace_id, workflow_name=workflow_name, started_at=datetime.utcnow())
        self._active_traces[trace_id] = trace
        logger.info("trace_started", trace_id=trace_id, workflow=workflow_name)
        return trace

    def record_step(self, trace_id: str, step: TraceStep) -> None:
        trace = self._active_traces.get(trace_id)
        if trace:
            trace.add_step(step)
        else:
            logger.warning("record_step_no_active_trace", trace_id=trace_id, step_node=step.node)

    def complete_trace(self, trace_id: str, status: str = "completed") -> Optional[AgentTrace]:
        trace = self._active_traces.pop(trace_id, None)
        if trace:
            trace.complete(status)
            self._completed_traces.append(trace)
            logger.info("trace_completed", trace_id=trace_id, status=status)
        return trace

    def get_trace(self, trace_id: str) -> Optional[AgentTrace]:
        return self._active_traces.get(trace_id) or next(
            (t for t in self._completed_traces if t.trace_id == trace_id),
            None
        )

global_trace_manager = TraceManager() # Global instance for convenience in notebook

# NODE FUNCTIONS - each represents a step in the workflow
async def supervisor_node(state: DueDiligenceState) -> Dict[str, Any]:
    """Supervisor decides which agent to run next based on assessment type and completed analyses."""
    company_id = state["company_id"]
    logger.info("supervisor_routing", company_id=company_id)

    # Check if approval is pending (HITL pause)
    if state.get("requires_approval") and state.get("approval_status") == "pending":
        return {"next_agent": "wait_for_approval"}

    # Route to next agent based on completion status
    if not state.get("sec_analysis"):
        return {"next_agent": "sec_analyst"}
    elif not state.get("talent_analysis"):
        return {"next_agent": "talent_analyst"}
    elif not state.get("scoring_result"):
        return {"next_agent": "scorer"}
    elif not state.get("value_creation_plan") and state["assessment_type"] != "screening":
        # Only run value creation for 'limited' or 'full' assessments
        return {"next_agent": "value_creator"}
    else:
        return {"next_agent": "complete"}

async def sec_analyst_node(state: DueDiligenceState, trace_manager_instance: TraceManager) -> Dict[str, Any]:
    """Run SEC analysis agent and record trace."""
    node_name = "sec_analyst"
    trace_id = state.get("trace_id")
    started_at = datetime.utcnow()
    inputs_for_trace = {"company_id": state["company_id"], "assessment_type": state["assessment_type"]}
    result = {}
    error_for_trace = None

    try:
        result = await sec_agent.analyze(state)
        outputs_for_trace = result
    except Exception as e:
        error_for_trace = str(e)
        logger.error("SEC analysis failed", company_id=state["company_id"], error=error_for_trace)
        # Ensure result has a structure even on error for subsequent nodes
        result["sec_analysis"] = {"company_id": state["company_id"], "findings": f"Error: {error_for_trace}", "evidence_count": 0, "dimensions_covered": [], "confidence": "low"}
        outputs_for_trace = result

    completed_at = datetime.utcnow()
    if trace_id:
        active_trace = trace_manager_instance.get_trace(trace_id) # Retrieve active trace
        if active_trace:
            step = TraceStep(
                node=node_name,
                started_at=started_at,
                completed_at=completed_at,
                inputs=inputs_for_trace,
                outputs=outputs_for_trace,
                error=error_for_trace
            )
            active_trace.add_step(step)
    
    return {
        **result,
        "messages": [AgentMessage(
            role="assistant",
            content=f"SEC analysis complete. Found evidence for {len(result['sec_analysis']['dimensions_covered'])} dimensions." if not error_for_trace else f"SEC analysis failed: {error_for_trace}",
            name="sec_analyst",
            timestamp=datetime.utcnow(),
        )],
    }

async def talent_analyst_node(state: DueDiligenceState, trace_manager_instance: TraceManager) -> Dict[str, Any]:
    """Run talent analysis agent and record trace."""
    node_name = "talent_analyst"
    trace_id = state.get("trace_id")
    started_at = datetime.utcnow()
    inputs_for_trace = {"company_id": state["company_id"]}
    result = {}
    error_for_trace = None

    try:
        result = await talent_agent.analyze(state)
        outputs_for_trace = result
    except Exception as e:
        error_for_trace = str(e)
        logger.error("Talent analysis failed", company_id=state["company_id"], error=error_for_trace)
        result["talent_analysis"] = {"company_id": state["company_id"], "ai_role_count": 0, "talent_concentration": 0.0, "seniority_index": 0.0, "key_skills": [], "hiring_trend": "error"}
        outputs_for_trace = result

    completed_at = datetime.utcnow()
    if trace_id:
        active_trace = trace_manager_instance.get_trace(trace_id)
        if active_trace:
            step = TraceStep(
                node=node_name,
                started_at=started_at,
                completed_at=completed_at,
                inputs=inputs_for_trace,
                outputs=outputs_for_trace,
                error=error_for_trace
            )
            active_trace.add_step(step)

    talent_concentration = result.get("talent_analysis", {}).get("talent_concentration", 0.0)
    return {
        **result,
        "messages": [AgentMessage(
            role="assistant",
            content=f"Talent analysis complete. Talent concentration: {talent_concentration:.0%}." if not error_for_trace else f"Talent analysis failed: {error_for_trace}",
            name="talent_analyst",
            timestamp=datetime.utcnow(),
        )],
    }

async def scorer_node(state: DueDiligenceState, trace_manager_instance: TraceManager) -> Dict[str, Any]:
    """Run scoring agent with HITL check and record trace."""
    node_name = "scorer"
    trace_id = state.get("trace_id")
    started_at = datetime.utcnow()
    inputs_for_trace = {"sec_analysis_exists": bool(state.get("sec_analysis")), "talent_analysis_exists": bool(state.get("talent_analysis"))}
    result = {}
    error_for_trace = None

    requires_approval = False
    approval_reason = None

    try:
        result = await scoring_agent.calculate(state)
        score = result["scoring_result"]["final_score"]
        outputs_for_trace = result
        
        # HITL check based on Org-AI-R score deviation
        S_min = 40.0
        S_max = 85.0
        if score < S_min or score > S_max:
            requires_approval = True
            approval_reason = f"Org-AI-R Score {score:.1f} outside normal range [{S_min:.0f}, {S_max:.0f}]"
            outputs_for_trace["requires_approval"] = True
            outputs_for_trace["approval_reason"] = approval_reason

    except Exception as e:
        error_for_trace = str(e)
        logger.error("Scoring failed", company_id=state["company_id"], error=error_for_trace)
        result["scoring_result"] = {"final_score": 0.0, "details": f"Error: {error_for_trace}"}
        outputs_for_trace = result

    completed_at = datetime.utcnow()
    if trace_id:
        active_trace = trace_manager_instance.get_trace(trace_id)
        if active_trace:
            step = TraceStep(
                node=node_name,
                started_at=started_at,
                completed_at=completed_at,
                inputs=inputs_for_trace,
                outputs=outputs_for_trace,
                error=error_for_trace
            )
            active_trace.add_step(step)

    content_msg = f"Scoring complete. Org-AI-R: {result.get('scoring_result',{}).get('final_score', 0.0):.1f}"
    if requires_approval:
        content_msg += f" ! Requires approval: {approval_reason}"

    return {
        **result,
        "requires_approval": requires_approval,
        "approval_reason": approval_reason,
        "approval_status": "pending" if requires_approval else None,
        "messages": [AgentMessage(
            role="assistant",
            content=content_msg if not error_for_trace else f"Scoring failed: {error_for_trace}",
            name="scorer",
            timestamp=datetime.utcnow(),
        )],
    }

async def value_creator_node(state: DueDiligenceState, trace_manager_instance: TraceManager) -> Dict[str, Any]:
    """Run value creation agent with HITL check for large projections and record trace."""
    node_name = "value_creator"
    trace_id = state.get("trace_id")
    started_at = datetime.utcnow()
    inputs_for_trace = {"current_score": state.get("scoring_result", {}).get("final_score", 0.0)}
    result = {}
    error_for_trace = None

    requires_approval = False
    approval_reason = None
    ebitda_impact_pct = 0.0

    try:
        result = await value_agent.plan(state)
        ebitda_impact_pct = result["value_creation_plan"]["projected_ebitda_impact_pct"]
        outputs_for_trace = result
        
        # HITL check for large EBITDA projections
        if ebitda_impact_pct > settings.HITL_EBITDA_PROJECTION_THRESHOLD:
            requires_approval = True
            approval_reason = f"EBITDA projection {ebitda_impact_pct:.1%} exceeds threshold ({settings.HITL_EBITDA_PROJECTION_THRESHOLD:.1%})"
            outputs_for_trace["requires_approval"] = True
            outputs_for_trace["approval_reason"] = approval_reason

    except Exception as e:
        error_for_trace = str(e)
        logger.error("Value creation plan failed", company_id=state["company_id"], error=error_for_trace)
        result["value_creation_plan"] = {"projected_ebitda_impact_pct": 0.0, "initiatives": [], "timeline_months": 0}
        outputs_for_trace = result

    completed_at = datetime.utcnow()
    if trace_id:
        active_trace = trace_manager_instance.get_trace(trace_id)
        if active_trace:
            step = TraceStep(
                node=node_name,
                started_at=started_at,
                completed_at=completed_at,
                inputs=inputs_for_trace,
                outputs=outputs_for_trace,
                error=error_for_trace
            )
            active_trace.add_step(step)

    content_msg = f"Value creation plan complete. Projected EBITDA impact: {ebitda_impact_pct:.1%}"
    if requires_approval:
        content_msg += f" ! Requires approval"

    # Preserve existing approval status if it's already pending from a previous step
    existing_approval_status = state.get("approval_status")
    final_approval_status = "pending" if requires_approval else existing_approval_status

    return {
        **result,
        "requires_approval": requires_approval,
        "approval_reason": approval_reason,
        "approval_status": final_approval_status,
        "messages": [AgentMessage(
            role="assistant",
            content=content_msg if not error_for_trace else f"Value creation failed: {error_for_trace}",
            name="value_creator",
            timestamp=datetime.utcnow(),
        )],
    }

async def complete_node(state: DueDiligenceState, trace_manager_instance: TraceManager) -> Dict[str, Any]:
    """Finalize workflow and record trace."""
    node_name = "complete"
    trace_id = state.get("trace_id")
    started_at = datetime.utcnow()
    inputs_for_trace = {"final_score": state.get("scoring_result", {}).get("final_score")}
    
    completed_at = datetime.utcnow()
    if trace_id:
        active_trace = trace_manager_instance.get_trace(trace_id)
        if active_trace:
            step = TraceStep(
                node=node_name,
                started_at=started_at,
                completed_at=completed_at,
                inputs=inputs_for_trace,
                outputs={"status": "completed"},
                error=None
            )
            active_trace.add_step(step)
            trace_manager_instance.complete_trace(trace_id, status="completed")

    return {
        "completed_at": datetime.utcnow(),
        "messages": [AgentMessage(
            role="assistant",
            content="Due diligence assessment complete.",
            name="supervisor",
            timestamp=datetime.utcnow(),
        )],
    }


# ROUTING LOGIC
def route_from_supervisor(state: DueDiligenceState) -> str:
    """Route from supervisor to next node based on 'next_agent' flag."""
    next_agent = state.get("next_agent", "complete")
    if next_agent == "wait_for_approval":
        return END # Pause for human approval
    return next_agent

# GRAPH CONSTRUCTION
def create_due_diligence_graph(trace_manager_instance: TraceManager) -> StateGraph:
    """Create the due diligence workflow graph."""
    workflow = StateGraph(DueDiligenceState)

    # Add nodes, using functools.partial to inject the trace_manager_instance
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("sec_analyst", functools.partial(sec_analyst_node, trace_manager_instance=trace_manager_instance))
    workflow.add_node("talent_analyst", functools.partial(talent_analyst_node, trace_manager_instance=trace_manager_instance))
    workflow.add_node("scorer", functools.partial(scorer_node, trace_manager_instance=trace_manager_instance))
    workflow.add_node("value_creator", functools.partial(value_creator_node, trace_manager_instance=trace_manager_instance))
    workflow.add_node("complete", functools.partial(complete_node, trace_manager_instance=trace_manager_instance))

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sec_analyst": "sec_analyst",
            "talent_analyst": "talent_analyst",
            "scorer": "scorer",
            "value_creator": "value_creator",
            "complete": "complete",
        },
    )

    # All specialist agents route back to supervisor
    workflow.add_edge("sec_analyst", "supervisor")
    workflow.add_edge("talent_analyst", "supervisor")
    workflow.add_edge("scorer", "supervisor")
    workflow.add_edge("value_creator", "supervisor")

    # Final edge to END
    workflow.add_edge("complete", END)

    # Create graph with memory checkpoint for persistence and HITL
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    return compiled_graph

# Compile the graph
due_diligence_graph = create_due_diligence_graph(global_trace_manager)
print("LangGraph due diligence workflow compiled successfully.")
```

## 5. Integrating Semantic Memory (Mem0)

Sarah now integrates Mem0 for semantic memory. This allows agents to store and retrieve contextual information about past assessments, company profiles, and user interactions. This capability is crucial for building intelligent agents that can learn from previous workflows, avoid redundant analysis, and provide more personalized insights over time, making the system more efficient and powerful.

If Mem0 is not available, a simple dictionary-based fallback is used.

```python
# SEMANTIC MEMORY WITH MEM0
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
    # Ensure MEM0_API_KEY is set in .env if using remote Mem0 service
    # Alternatively, Mem0 can run locally: pip install "mem0[qdrant]" or "mem0[chroma]"
except ImportError:
    MEM0_AVAILABLE = False
    print("Mem0 not available. Using fallback dict storage for AgentMemory.")
    Memory = None

class AgentMemory:
    """Semantic memory for PE Org-AI-R agents.
    Stores and retrieves contextual information about:
    - Companies (past assessments, key findings)
    - Users (preferences, past interactions)
    - Workflows (decisions, outcomes)
    """
    def __init__(self):
        if MEM0_AVAILABLE and os.getenv("MEM0_API_KEY"):
            try:
                self.memory = Memory(api_key=os.getenv("MEM0_API_KEY"))
                logger.info("Mem0 initialized with API key.")
            except Exception as e:
                logger.warning("Mem0 initialization failed, falling back to dict storage", error=str(e))
                self.memory = None
                MEM0_AVAILABLE = False
        else:
            self.memory = None
            logger.warning("Mem0 not available or MEM0_API_KEY not set. Using fallback dict storage.")
        self._fallback_store: Dict[str, List[Dict[str, Any]]] = {}

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a memory."""
        metadata = metadata or {}
        if self.memory:
            try:
                result = await self.memory.add(
                    content,
                    user_id=user_id,
                    metadata=metadata,
                )
                memory_id = result.get("id", "unknown")
                logger.debug("mem0_memory_added", memory_id=memory_id, user_id=user_id)
                return memory_id
            except Exception as e:
                logger.error("Mem0 add_memory failed, falling back", error=str(e))
                # Fallback to local storage
                pass # Continue to fallback logic below
        
        # Fallback
        memory_id = f"mem-{datetime.utcnow().timestamp():.0f}"
        if user_id not in self._fallback_store:
            self._fallback_store[user_id] = []
        self._fallback_store[user_id].append({
            "id": memory_id,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        })
        logger.debug("fallback_memory_added", memory_id=memory_id, user_id=user_id)
        return memory_id

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search relevant memories."""
        if self.memory:
            try:
                results = await self.memory.search(
                    query,
                    user_id=user_id,
                    limit=limit,
                )
                logger.debug("mem0_search_complete", query=query, user_id=user_id, results_count=len(results))
                return results
            except Exception as e:
                logger.error("Mem0 search_memories failed, falling back", error=str(e))
                # Fallback to local storage
                pass # Continue to fallback logic below
        
        # Fallback: Simple keyword search
        user_memories = self._fallback_store.get(user_id, [])
        query_lower = query.lower()
        matches = [
            m for m in user_memories
            if query_lower in m["content"].lower() or any(query_lower in str(v).lower() for v in m["metadata"].values())
        ]
        logger.debug("fallback_search_complete", query=query, user_id=user_id, results_count=len(matches))
        return matches[:limit]

    async def get_company_context(
        self,
        company_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Get all stored context about a company."""
        memories = await self.search_memories(
            f"company {company_id}", # Search for mentions of the company
            user_id=user_id,
            limit=20,
        )
        # Filter memories specifically tied to this company_id in metadata
        company_specific_memories = [m for m in memories if m.get("metadata", {}).get("company_id") == company_id]
        logger.debug("get_company_context_complete", company_id=company_id, memory_count=len(company_specific_memories))
        return {
            "company_id": company_id,
            "memory_count": len(company_specific_memories),
            "memories": company_specific_memories,
        }

    async def store_assessment_outcome(
        self,
        company_id: str,
        assessment_type: str,
        final_score: float,
        key_findings: List[str],
        user_id: str = "system",
    ) -> None:
        """Store assessment outcome for future reference."""
        content = f"""Company {company_id} {assessment_type} assessment completed.
Final Org-AI-R score: {final_score:.1f}
Key findings:
{chr(10).join(f'- {f}' for f in key_findings)}"""
        
        metadata = {
            "company_id": company_id,
            "assessment_type": assessment_type,
            "final_score": final_score,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.add_memory(content=content, user_id=user_id, metadata=metadata)
        logger.info("assessment_outcome_stored", company_id=company_id, final_score=final_score)

agent_memory = AgentMemory()
```

## 6. Executing a Due Diligence Workflow (with HITL Pause)

Sarah will now run the entire due diligence workflow for a sample company. This execution demonstrates how the supervisor orchestrates specialist agents and, critically, how the system pauses for human approval when predefined HITL conditions are met.

We'll run the workflow for "RiskyBet Corp." with a `full` assessment, which is specifically designed to produce an Org-AI-R score that falls outside the normal range, triggering a HITL review.

```python
# EXECUTION API
async def run_due_diligence(
    company_id: str,
    assessment_type: Literal["screening", "limited", "full"] = "limited",
    requested_by: str = "system",
    thread_id: Optional[str] = None,
) -> DueDiligenceState:
    """Run due diligence workflow.
    Args:
        company_id: Company to assess.
        assessment_type: Depth of assessment ("screening", "limited", "full").
        requested_by: User requesting assessment.
        thread_id: Optional thread ID for resuming a paused workflow.
    Returns:
        Final workflow state.
    """
    if not thread_id:
        # Generate a unique thread ID for a new workflow run
        thread_id = f"dd-{company_id.replace(' ', '_').lower()}-{datetime.utcnow().isoformat(timespec='seconds')}"

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
        "trace_id": thread_id, # Link the trace to the thread ID
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    
    # Start tracing for this workflow run
    global_trace_manager.start_trace(thread_id, f"Due Diligence for {company_id}")

    logger.info("due_diligence_started", company_id=company_id, assessment_type=assessment_type, thread_id=thread_id)

    # Run until completion or HITL pause
    final_state = await due_diligence_graph.ainvoke(initial_state, config)
    
    # If the workflow completed without HITL, or after HITL approval
    if final_state.get("approval_status") != "pending":
        global_trace_manager.complete_trace(thread_id, status=final_state.get("approval_status", "completed"))

    return final_state

# --- Run the workflow for a company that triggers HITL ---
company_to_assess_hitl = "RiskyBet Corp."
print(f"Initiating full due diligence for {company_to_assess_hitl} (expected to trigger HITL)...")
hitl_triggered_state = await run_due_diligence(
    company_id=company_to_assess_hitl,
    assessment_type="full",
    requested_by="Sarah"
)

print(f"\nWorkflow for '{company_to_assess_hitl}' current status:")
print(f"Approval Status: {hitl_triggered_state.get('approval_status')}")
print(f"Requires Approval: {hitl_triggered_state.get('requires_approval')}")
print(f"Approval Reason: {hitl_triggered_state.get('approval_reason')}")
print(f"Final Score: {hitl_triggered_state.get('scoring_result', {}).get('final_score')}")

# Display the Mermaid trace up to the HITL pause
if hitl_triggered_state.get("trace_id"):
    current_trace = global_trace_manager.get_trace(hitl_triggered_state["trace_id"])
    if current_trace:
        print("\n--- Agent Trace (up to HITL pause) ---")
        print(current_trace.to_mermaid())
    else:
        print(f"Trace for ID {hitl_triggered_state['trace_id']} not found.")
```

### Explanation of Execution
As a Software Developer, Sarah sees that the workflow for "RiskyBet Corp." has paused (`approval_status: pending`). This demonstrates the HITL mechanism in action, as the `Org-AI-R Score` of `RiskyBet Corp.` (which is likely below 40 due to our mock `OrgAIRCalculator` logic for this company) triggered the conditional edge from the `scorer` node. The trace diagram provides a clear visual path of how the agents executed until the HITL breakpoint, indicating where human intervention is needed. This ensures that any unusual or critical findings are reviewed by an analyst before proceeding.

## 7. Handling Human-in-the-Loop Approval and Resuming

After the workflow pauses for HITL, a human analyst (or Sarah, simulating an analyst) reviews the findings. Based on their assessment, they can either approve or reject the continuation of the workflow. Sarah implements the `approve_workflow` function to submit this decision and resume the LangGraph execution.

```python
async def approve_workflow(
    thread_id: str,
    approved_by: str,
    decision: Literal["approved", "rejected"],
    notes: Optional[str] = None,
) -> DueDiligenceState:
    """Submit approval decision and resume workflow."""
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state before update
    state = await due_diligence_graph.aget_state(config)
    if not state:
        raise ValueError(f"No active workflow found for thread_id: {thread_id}")

    # Update approval status
    update = {
        "approval_status": decision,
        "approved_by": approved_by,
        "requires_approval": False, # Reset approval flag
        "messages": [AgentMessage(
            role="system",
            content=f"Workflow {decision} by {approved_by}" + (f": {notes}" if notes else ""),
            name="hitl",
            timestamp=datetime.utcnow(),
        )],
    }
    
    logger.info("hitl_decision_made", thread_id=thread_id, decision=decision, approved_by=approved_by)

    # Resume if approved
    if decision == "approved":
        final_state = await due_diligence_graph.ainvoke({**state.values, **update}, config)
        global_trace_manager.complete_trace(thread_id, status="completed")
        return final_state
    else:
        # If rejected, update trace and return current state with error
        current_trace = global_trace_manager.get_trace(thread_id)
        if current_trace:
            current_trace.complete(status="rejected")
            # Move from active to completed traces
            global_trace_manager._completed_traces.append(global_trace_manager._active_traces.pop(thread_id))

        return {**state.values, **update, "error": f"Workflow rejected by {approved_by}. Reason: {notes or 'No reason provided'}."}

# --- Simulate human approval and resume ---
workflow_thread_id = hitl_triggered_state["trace_id"] # Use the trace_id as thread_id
print(f"\nSimulating human approval for workflow ID: {workflow_thread_id}")

# Approve the workflow
final_state_after_approval = await approve_workflow(
    thread_id=workflow_thread_id,
    approved_by="Lead Analyst John",
    decision="approved",
    notes="Acknowledged low score, proceed for further detailed analysis."
)

print(f"\nWorkflow for '{company_to_assess_hitl}' final status after approval:")
print(f"Approval Status: {final_state_after_approval.get('approval_status')}")
print(f"Final Score: {final_state_after_approval.get('scoring_result', {}).get('final_score')}")
print(f"Value Creation Plan exists: {bool(final_state_after_approval.get('value_creation_plan'))}")

# Display the complete Mermaid trace
if final_state_after_approval.get("trace_id"):
    complete_trace = global_trace_manager.get_trace(final_state_after_approval["trace_id"])
    if complete_trace:
        print("\n--- Complete Agent Trace ---")
        print(complete_trace.to_mermaid())
    else:
        print(f"Trace for ID {final_state_after_approval['trace_id']} not found (might be already moved to completed).")
        # Try to retrieve from _completed_traces if it's already finished
        complete_trace = next(
            (t for t in global_trace_manager._completed_traces if t.trace_id == final_state_after_approval["trace_id"]),
            None
        )
        if complete_trace:
            print(complete_trace.to_mermaid())

```

### Explanation of Execution
After the `Lead Analyst John` approves the workflow, Sarah observes that the `approval_status` changes to "approved", and the workflow successfully continues and completes, including generating a `value_creation_plan`. The updated Mermaid trace now shows the entire path from start, through the `scorer` node, the HITL pause (represented by `HITL: approval needed`), and then the continuation to `value_creator` and `complete`. This visual confirms that the human intervention point was successfully handled and the system resumed its automated tasks.

## 8. Storing Outcomes and Retrieving Context with Mem0

Now that the due diligence for "RiskyBet Corp." is complete, Sarah will store the key findings and the final Org-AI-R score into Mem0. This allows the system to build a rich historical context for each company. Afterwards, she demonstrates how to retrieve this stored context, showcasing how future analyses can benefit from past insights without re-running all initial steps.

```python
# --- Store the assessment outcome ---
company_id_for_memory = final_state_after_approval["company_id"]
assessment_type_for_memory = final_state_after_approval["assessment_type"]
final_score_for_memory = final_state_after_approval.get("scoring_result", {}).get("final_score", 0.0)
key_findings_for_memory = [
    "Identified low Org-AI-R score, requiring HITL review.",
    f"Projected EBITDA impact: {final_state_after_approval.get('value_creation_plan', {}).get('projected_ebitda_impact_pct', 0.0):.1%}",
    "Key value creation initiatives proposed: Data Platform Modernization, AI Center of Excellence, MLOps Implementation."
]

print(f"\nStoring assessment outcome for {company_id_for_memory} in Mem0...")
await agent_memory.store_assessment_outcome(
    company_id=company_id_for_memory,
    assessment_type=assessment_type_for_memory,
    final_score=final_score_for_memory,
    key_findings=key_findings_for_memory,
    user_id="Sarah"
)
print("Assessment outcome stored.")

# --- Retrieve context for the company from Mem0 ---
print(f"\nRetrieving historical context for {company_id_for_memory} from Mem0...")
company_context = await agent_memory.get_company_context(company_id=company_id_for_memory, user_id="Sarah")

print(f"\nRetrieved {company_context['memory_count']} memories for {company_context['company_id']}:")
for i, mem in enumerate(company_context["memories"]):
    print(f"--- Memory {i+1} ---")
    print(f"ID: {mem.get('id', 'N/A')}")
    print(f"Content: {mem.get('content', 'N/A')[:200]}...") # Print first 200 chars
    print(f"Metadata: {mem.get('metadata', 'N/A')}")
    print("-" * 20)

# --- Demonstrate another run without HITL for a "StarPerformer" company ---
company_to_assess_no_hitl = "StarPerformer Inc."
print(f"\nInitiating full due diligence for {company_to_assess_no_hitl} (expected to complete without HITL)...")
normal_run_state = await run_due_diligence(
    company_id=company_to_assess_no_hitl,
    assessment_type="full",
    requested_by="Sarah"
)

print(f"\nWorkflow for '{company_to_assess_no_hitl}' current status:")
print(f"Approval Status: {normal_run_state.get('approval_status')}")
print(f"Final Score: {normal_run_state.get('scoring_result', {}).get('final_score')}")

if normal_run_state.get("trace_id"):
    normal_trace = global_trace_manager.get_trace(normal_run_state["trace_id"])
    if normal_trace:
        print("\n--- Agent Trace (Normal Run) ---")
        print(normal_trace.to_mermaid())
    else:
        print(f"Trace for ID {normal_run_state['trace_id']} not found.")

```

### Explanation of Execution
Sarah successfully stores the detailed outcome of "RiskyBet Corp."'s assessment in Mem0. When she retrieves the company's context, the previously stored assessment is visible, demonstrating the long-term memory capability. This is valuable for PE analysts who can instantly pull up past assessments, track changes, and avoid repeating foundational research.

Furthermore, she runs a second assessment for "StarPerformer Inc." This company is designed to have an Org-AI-R score and projected EBITDA within the normal ranges, demonstrating a smooth, fully automated workflow *without* triggering any HITL pauses. The trace for "StarPerformer Inc." visually confirms this direct path, highlighting the flexibility of the multi-agent system to adapt based on dynamic conditions.

