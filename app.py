
import streamlit as st
from source import *
import asyncio
import uuid
from datetime import datetime
import nest_asyncio

# Apply nest_asyncio at the top level to allow nested asyncio.run calls.
# This is a common workaround for Streamlit apps that need to run async functions
# synchronously, especially when dealing with libraries that might also manage
# their own event loops or call asyncio.run internally.
nest_asyncio.apply()

# --- Page Configuration ---
st.set_page_config(page_title="QuLab: Lab 10: LangGraph Multi-Agent Orchestration", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 10: LangGraph Multi-Agent Orchestration")
st.divider()

# --- Helper Functions ---
def run_async_function(func, *args, **kwargs):
    """
    Helper function to run an async function synchronously.
    With nest_asyncio.apply() called globally, asyncio.run can be called
    even if an event loop is already running, simplifying this function.
    """
    return asyncio.run(func(*args, **kwargs))

# --- Session State Initialization ---
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "company_id_input" not in st.session_state:
    st.session_state["company_id_input"] = "TechInnovate Inc."
if "assessment_type_input" not in st.session_state:
    st.session_state["assessment_type_input"] = "full"
if "requested_by_input" not in st.session_state:
    st.session_state["requested_by_input"] = "Sarah (Streamlit App)"
if "latest_workflow_state" not in st.session_state:
    st.session_state["latest_workflow_state"] = None
if "latest_thread_id" not in st.session_state:
    st.session_state["latest_thread_id"] = None
if "hitl_approval_by" not in st.session_state:
    st.session_state["hitl_approval_by"] = "Lead Analyst"
if "hitl_decision" not in st.session_state:
    st.session_state["hitl_decision"] = "approved"
if "hitl_notes" not in st.session_state:
    st.session_state["hitl_notes"] = ""
if "company_history_id_input" not in st.session_state:
    st.session_state["company_history_id_input"] = "RiskyBet Corp."
if "company_memories" not in st.session_state:
    st.session_state["company_memories"] = None

# --- Source Module Initialization ---
# The error message "SyntaxError: 'await' outside function (source.py, line 981)"
# indicates that `source.py` had a top-level `await verify_agent_memory()`.
# To fix this, `source.py` must define `verify_agent_memory` as an `async def` function
# without calling `await` at the top level.
# This block then correctly calls `verify_agent_memory()` using `run_async_function`
# once at application startup, assuming `source.py` has been adjusted to be syntactically valid.
# The current `app.py` file is already structured correctly to interact with a syntactically
# valid `source.py`. The `SyntaxError` in `source.py` is a parsing error that occurs
# during the `from source import *` statement and cannot be directly fixed by modifying `app.py`
# itself without making the application non-functional. The `app.py` file, as provided,
# assumes `source.py` has been corrected to remove the top-level `await` statement.
if "source_module_initialized" not in st.session_state:
    try:
        # Check if 'verify_agent_memory' is imported, is callable, and is an async function
        if "verify_agent_memory" in globals() and callable(verify_agent_memory) and asyncio.iscoroutinefunction(verify_agent_memory):
            run_async_function(verify_agent_memory)
            st.session_state["source_module_initialized"] = True
            # Optional: st.success("Agent memory verified successfully.")
        else:
            st.warning("Warning: 'verify_agent_memory' function not found or not an async function after importing 'source'. Skipping initialization.")
            st.session_state["source_module_initialized"] = True # Mark as initialized to prevent repeated warnings
    except NameError:
        st.error("Initialization error: 'verify_agent_memory' was not found during source module setup. Ensure it's defined in source.py.")
        st.session_state["source_module_initialized"] = True # Prevent repeated errors on rerun
    except Exception as e:
        st.error(f"Error during source module initialization: {e}")
        st.session_state["source_module_initialized"] = True # Prevent repeated errors on rerun

# --- Callbacks ---
def start_assessment_callback():
    st.session_state["current_page"] = "assessment_details"
    thread_id = f"dd-{st.session_state['company_id_input'].replace(' ', '_').lower()}-{datetime.utcnow().isoformat(timespec='seconds')}"
    st.session_state["latest_thread_id"] = thread_id
    with st.spinner("Starting multi-agent due diligence workflow..."):
        workflow_state = run_async_function(
            run_due_diligence,
            company_id=st.session_state["company_id_input"],
            assessment_type=st.session_state["assessment_type_input"],
            requested_by=st.session_state["requested_by_input"],
            thread_id=st.session_state["latest_thread_id"]
        )
        st.session_state["latest_workflow_state"] = workflow_state
        if workflow_state.get("approval_status") == "pending":
            st.warning("Workflow paused for HITL approval!")
        elif workflow_state.get("error"):
            st.error("Workflow completed with errors.")
        else:
            st.success("Workflow completed successfully.")

def submit_approval_callback():
    if not st.session_state["latest_thread_id"]:
        st.error("No active workflow to approve.")
        return
    
    with st.spinner(f"Submitting HITL decision '{st.session_state['hitl_decision']}'..."):
        updated_state = run_async_function(
            approve_workflow,
            thread_id=st.session_state["latest_thread_id"],
            approved_by=st.session_state["hitl_approval_by"],
            decision=st.session_state["hitl_decision"],
            notes=st.session_state["hitl_notes"]
        )
        st.session_state["latest_workflow_state"] = updated_state
        if updated_state.get("approval_status") == "approved":
            st.success("Workflow approved and resumed/completed!")
        elif updated_state.get("approval_status") == "rejected":
            st.error("Workflow rejected and terminated.")
        else:
            st.info("HITL decision processed, workflow state updated.")

def store_outcome_callback():
    if not st.session_state["latest_workflow_state"]:
        st.error("No completed workflow state to store.")
        return

    state = st.session_state["latest_workflow_state"]
    company_id = state.get("company_id")
    assessment_type = state.get("assessment_type")
    final_score = state.get("scoring_result", {}).get("final_score", 0.0)
    
    # Summarize key findings for memory
    key_findings = []
    if state.get("sec_analysis", {}).get("findings"):
        key_findings.append(f"SEC Analysis: {state['sec_analysis']['findings'][:100]}...")
    if state.get("talent_analysis", {}).get("ai_role_count"):
        key_findings.append(f"Talent Analysis: {state['talent_analysis']['ai_role_count']} AI roles, {state['talent_analysis']['talent_concentration']:.1%} talent concentration.")
    if state.get("value_creation_plan", {}).get("initiatives"):
        initiatives = [f"{i['name']} (Impact: {i['impact']})" for i in state['value_creation_plan']['initiatives']]
        key_findings.append(f"Value Creation Initiatives: {'; '.join(initiatives)}.")
    if state.get("approval_reason"):
        key_findings.append(f"HITL triggered due to: {state['approval_reason']}.")

    if not key_findings:
        key_findings.append("No specific key findings generated during assessment.")

    with st.spinner("Storing assessment outcome in Mem0..."):
        run_async_function(
            agent_memory.store_assessment_outcome,
            company_id=company_id,
            assessment_type=assessment_type,
            final_score=final_score,
            key_findings=key_findings,
            user_id=st.session_state["requested_by_input"]
        )
    st.success(f"Assessment outcome for '{company_id}' stored in Mem0!")

def view_history_callback():
    if st.session_state["latest_workflow_state"]:
        st.session_state["company_history_id_input"] = st.session_state["latest_workflow_state"]["company_id"]
    st.session_state["current_page"] = "company_history"
    load_company_history_callback() 

def load_company_history_callback():
    if not st.session_state["company_history_id_input"]:
        st.error("Please enter a Company ID to load history.")
        return

    with st.spinner(f"Loading historical context for '{st.session_state['company_history_id_input']}' from Mem0..."):
        memories = run_async_function(
            agent_memory.get_company_context,
            company_id=st.session_state["company_history_id_input"],
            user_id=st.session_state["requested_by_input"] 
        )
        st.session_state["company_memories"] = memories
    st.success("Company history loaded!")

# --- Streamlit Application Layout ---
st.sidebar.title("Navigation")
page_options = ["Home", "New Assessment", "Assessment Details", "Company History"]
current_selection = "Home"

if st.session_state["current_page"] == "new_assessment":
    current_selection = "New Assessment"
elif st.session_state["current_page"] == "assessment_details":
    current_selection = "Assessment Details"
elif st.session_state["current_page"] == "company_history":
    current_selection = "Company History"

page_selection = st.sidebar.radio(
    "Go to",
    page_options,
    index=page_options.index(current_selection)
)

# Update current_page based on sidebar selection
if page_selection == "Home": st.session_state["current_page"] = "home"
elif page_selection == "New Assessment": st.session_state["current_page"] = "new_assessment"
elif page_selection == "Assessment Details": st.session_state["current_page"] = "assessment_details"
elif page_selection == "Company History": st.session_state["current_page"] = "company_history"

# --- Main Content Area ---
if st.session_state["current_page"] == "home":
    st.title("AI-Powered Due Diligence for Private Equity")
    st.markdown(f"")

    st.markdown(f"## Introduction: Empowering PE Analysts with AI-Driven Insights")
    st.markdown(f"")
    st.markdown(f"**Persona:** Sarah, a Software Developer at \"Synergy Capital,\" a forward-thinking Private Equity (PE) firm.")
    st.markdown(f"")
    st.markdown(f"**Organization:** Synergy Capital, specializing in acquiring and growing technology companies.")
    st.markdown(f"")
    st.markdown(f"Sarah's team is tasked with modernizing Synergy Capital's initial due diligence process. Currently, PE analysts spend significant time manually sifting through financial filings, talent reports, and market data to assess potential target companies. This is time-consuming and prone to inconsistencies, delaying critical investment decisions.")
    st.markdown(f"")
    st.markdown(f"The goal is to develop a proof-of-concept for an AI-powered multi-agent system that automates the initial assessment of a target company's AI-readiness. This system will leverage LangGraph to orchestrate specialized AI agents (e.g., SEC Analysis, Talent Analysis, Scoring, Value Creation), providing a structured, consolidated report to analysts. This allows analysts to quickly grasp a company's profile and focus their expertise on high-value strategic considerations, significantly accelerating the deal pipeline.")
    st.markdown(f"")
    st.markdown(f"This application demonstrates how to define the workflow state, build specialist agents, orchestrate them with a supervisor agent, integrate human-in-the-loop (HITL) approvals, add semantic memory with Mem0, and visualize agent traces for debugging.")
    st.markdown(f"")

    st.markdown(f"---")
    st.markdown(f"### Key Concepts Addressed in this Application:")
    st.markdown(f"")
    st.markdown(f"- Supervisor pattern for agent coordination")
    st.markdown(f"- Specialist agents (SEC, Talent, Scoring, Value)")
    st.markdown(f"- Human-in-the-loop (HITL) approval gates")
    st.markdown(f"- Semantic memory with Mem0")
    st.markdown(f"- Agent debug traces")
    st.markdown(f"")
    st.markdown(f"Navigate using the sidebar to start a new assessment or view company history.")

elif st.session_state["current_page"] == "new_assessment":
    st.title("Initiate New Due Diligence Assessment")
    st.markdown(f"")
    st.markdown(f"Enter the details for the company you wish to assess. The multi-agent workflow will analyze the company's AI-readiness and generate a comprehensive report.")
    st.markdown(f"")

    st.text_input("Company ID", value=st.session_state["company_id_input"], key="company_id_input")
    
    assessment_options = ["screening", "limited", "full"]
    st.selectbox(
        "Assessment Type", 
        assessment_options, 
        index=assessment_options.index(st.session_state["assessment_type_input"]),
        key="assessment_type_input"
    )
    
    st.text_input("Requested By", value=st.session_state["requested_by_input"], key="requested_by_input")

    st.button("Start Assessment", on_click=start_assessment_callback, type="primary")

elif st.session_state["current_page"] == "assessment_details":
    st.title("Due Diligence Assessment Details")
    st.markdown(f"")

    if not st.session_state["latest_workflow_state"]:
        st.info("No assessment has been run yet. Please start a new assessment.")
        if st.button("Go to New Assessment"):
            st.session_state["current_page"] = "new_assessment"
            st.rerun()
    else:
        current_state = st.session_state["latest_workflow_state"]

        st.markdown(f"### Workflow Status for {current_state['company_id']}")
        st.markdown(f"")
        if current_state.get("approval_status") == "pending" and current_state.get("requires_approval"):
            st.warning(f"**Workflow Paused for Human-in-the-Loop (HITL) Approval!**")
            st.markdown(f"**Reason:** {current_state.get('approval_reason', 'N/A')}")
        elif current_state.get("error"):
            st.error(f"**Workflow Ended with Error:** {current_state['error']}")
        elif current_state.get("completed_at"):
            st.success(f"**Workflow Completed Successfully!**")
        else:
            st.info(f"**Workflow is currently running or pending.** (Last update: {current_state['started_at'].strftime('%Y-%m-%d %H:%M:%S')})")
        st.markdown(f"")

        # Display SEC Analysis
        if current_state.get("sec_analysis"):
            st.markdown(f"### 📄 SEC Analysis Findings")
            st.markdown(f"")
            sec_data = current_state["sec_analysis"]
            st.markdown(f"- **Company ID:** {sec_data.get('company_id')}")
            st.markdown(f"- **Findings Summary:** {sec_data.get('findings')}")
            st.markdown(f"- **Evidence Count:** {sec_data.get('evidence_count')}")
            st.markdown(f"- **Dimensions Covered:** {', '.join(sec_data.get('dimensions_covered', []))}")
            st.markdown(f"- **Confidence:** {sec_data.get('confidence')}")
            st.markdown(f"")

        # Display Talent Analysis
        if current_state.get("talent_analysis"):
            st.markdown(f"### 🧑‍💻 Talent Analysis Findings")
            st.markdown(f"")
            talent_data = current_state["talent_analysis"]
            st.markdown(f"- **Company ID:** {talent_data.get('company_id')}")
            st.markdown(f"- **AI Role Count:** {talent_data.get('ai_role_count')}")
            st.markdown(f"- **Talent Concentration:** {talent_data.get('talent_concentration', 0.0):.2%}")
            st.markdown(f"- **Seniority Index:** {talent_data.get('seniority_index', 0.0):.1f}")
            st.markdown(f"- **Key Skills:** {', '.join(talent_data.get('key_skills', []))}")
            st.markdown(f"- **Hiring Trend:** {talent_data.get('hiring_trend')}")
            st.markdown(f"")

        # Display Scoring Result
        if current_state.get("scoring_result"):
            st.markdown(f"### 📊 Org-AI-R Scoring Result")
            st.markdown(f"")
            scoring_data = current_state["scoring_result"]
            final_score = scoring_data.get('final_score', 0.0)
            st.markdown(f"- **Final Org-AI-R Score:** **{final_score:.1f}**")
            st.markdown(f"- **Details:** {scoring_data.get('details', 'N/A')}")
            st.markdown(f"")
            st.markdown(r"$$ (S < S_{\text{min}}) \lor (S > S_{\text{max}}) $$")
            st.markdown(r"where $S$ is the Org-AI-R score, $S_{\text{min}}=40$, and $S_{\text{max}}=85$.")
            st.markdown(f"")

        # Display Value Creation Plan
        if current_state.get("value_creation_plan"):
            st.markdown(f"### 💰 Value Creation Plan")
            st.markdown(f"")
            value_data = current_state["value_creation_plan"]
            st.markdown(f"- **Current Org-AI-R Score:** {value_data.get('current_score', 0.0):.1f}")
            st.markdown(f"- **Target Org-AI-R Score:** {value_data.get('target_score', 0.0):.1f}")
            st.markdown(f"- **Projected EBITDA Impact:** **{value_data.get('projected_ebitda_impact_pct', 0.0):.1%}**")
            st.markdown(f"- **Timeline (Months):** {value_data.get('timeline_months')}")
            st.markdown(f"- **Key Initiatives:**")
            st.markdown(f"")
            for initiative in value_data.get('initiatives', []):
                st.markdown(f"  - **{initiative.get('name')}**: Impact: {initiative.get('impact')}, Cost: ${initiative.get('cost_mm', 0.0):.1f}M")
            st.markdown(f"")
            st.markdown(r"$$ I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD} $$")
            st.markdown(r"where $I$ is the projected EBITDA impact and $\text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD}$ is a predefined threshold (e.g., 7%).")
            st.markdown(f"")

        # HITL Approval Section
        if current_state.get("requires_approval") and current_state.get("approval_status") == "pending":
            st.markdown(f"---")
            st.subheader("Human-in-the-Loop (HITL) Approval Required")
            st.markdown(f"")
            st.warning(f"The workflow for '{current_state['company_id']}' requires your review before proceeding.")
            st.markdown(f"**Reason for HITL:** {current_state.get('approval_reason', 'N/A')}")
            st.markdown(f"")

            st.text_input("Approved By", value=st.session_state["hitl_approval_by"], key="hitl_approval_by")
            
            decision_options = ["approved", "rejected"]
            st.radio(
                "Decision", 
                decision_options, 
                index=0 if st.session_state["hitl_decision"] == "approved" else 1, 
                key="hitl_decision"
            )
            
            st.text_area("Notes", value=st.session_state["hitl_notes"], key="hitl_notes")

            st.button("Submit Approval", on_click=submit_approval_callback, type="secondary")
            st.markdown(f"")

        # Workflow Trace Visualization
        st.markdown(f"---")
        st.subheader("Agent Workflow Trace")
        st.markdown(f"")
        st.markdown(f"This diagram visualizes the execution path of the multi-agent system, including any HITL pauses.")
        st.markdown(f"")
        
        trace_id_for_mermaid = current_state.get("trace_id")
        if trace_id_for_mermaid:
            # Check if global_trace_manager is available (imported from source)
            if "global_trace_manager" in globals():
                current_trace = global_trace_manager.get_trace(trace_id_for_mermaid)
                if current_trace:
                    st.markdown(current_trace.to_mermaid())
                else:
                    st.info(f"Trace for ID {trace_id_for_mermaid} not found in manager. It might have already been moved to completed traces.")
            else:
                st.warning("global_trace_manager not found. Cannot display workflow trace.")
        else:
            st.info("No trace ID available for this workflow.")
        st.markdown(f"")

        # Action buttons (Store Outcome, View Company History)
        if not current_state.get("requires_approval") and current_state.get("completed_at") and not current_state.get("error"):
            col1, col2 = st.columns(2)
            with col1:
                st.button("Store Outcome in Mem0", on_click=store_outcome_callback, type="secondary")
            with col2:
                st.button("View Company History", on_click=view_history_callback, type="secondary")
            st.markdown(f"")

elif st.session_state["current_page"] == "company_history":
    st.title("Company Assessment History")
    st.markdown(f"")
    st.markdown(f"Retrieve past assessment outcomes and contextual information for a specific company from Mem0.")
    st.markdown(f"")

    st.text_input("Company ID for History", value=st.session_state["company_history_id_input"], key="company_history_id_input")
    st.button("Load Company History", on_click=load_company_history_callback, type="primary")
    st.markdown(f"")

    if st.session_state["company_memories"]:
        st.subheader(f"Memories for {st.session_state['company_memories']['company_id']}")
        st.markdown(f"")
        if st.session_state["company_memories"]['memory_count'] > 0:
            for i, mem in enumerate(st.session_state["company_memories"]["memories"]):
                st.markdown(f"---")
                st.markdown(f"**Memory {i+1} (ID: {mem.get('id', 'N/A')})**")
                st.markdown(f"")
                st.markdown(f"**Content:** {mem.get('content', 'N/A')}")
                st.markdown(f"")
                st.markdown(f"**Metadata:**")
                st.markdown(f"")
                for k, v in mem.get('metadata', {}).items():
                    st.markdown(f"  - {k}: {v}")
                st.markdown(f"---")
                st.markdown(f"")
        else:
            st.info(f"No historical memories found for '{st.session_state['company_history_id_input']}'.")
    else:
        st.info("No company history loaded yet.")

