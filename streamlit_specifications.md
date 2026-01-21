
# Streamlit Application Specification: AI-Powered Due Diligence for Private Equity

## 1. Application Overview

### Purpose
The "AI-Powered Due Diligence for Private Equity" application aims to revolutionize the initial assessment process for target companies by leveraging a LangGraph-orchestrated multi-agent AI system. This proof-of-concept Streamlit application allows Private Equity (PE) analysts, like Sarah (a Software Developer at Synergy Capital), to automate the initial screening for a company's AI-readiness. The goal is to accelerate investment decisions by providing structured, consolidated reports, flagging critical findings for human review, and building a semantic memory of past assessments. This reduces manual effort, improves consistency, and enables analysts to focus on high-value strategic analysis.

### High-Level Story Flow
1.  **Introduction**: Sarah launches the application and is greeted with an overview of its capabilities and the problem it solves. She learns about the multi-agent architecture and its benefits for due diligence.
2.  **New Assessment Initiation**: Sarah navigates to the "New Assessment" page. She inputs a target `Company ID`, selects an `Assessment Type` (screening, limited, or full), and provides her name.
3.  **Workflow Execution**: Upon clicking "Start Assessment", the application triggers a LangGraph multi-agent workflow in the backend. This workflow involves specialized AI agents (SEC Analysis, Talent Analysis, Scoring, Value Creation) working in a coordinated manner. The application displays a real-time (mock) progress indicator.
4.  **Human-in-the-Loop (HITL) Intervention**: If the workflow identifies critical conditions (e.g., an Org-AI-R score outside a predefined range, or an unusually high EBITDA projection), the workflow pauses. The application presents the reason for the pause and prompts Sarah (or a designated analyst) for approval or rejection.
5.  **Approval/Rejection and Resumption**: Sarah reviews the findings and makes a decision, providing optional notes. If approved, the workflow resumes from where it left off, continuing subsequent agent tasks. If rejected, the workflow terminates with an error.
6.  **Consolidated Report & Trace Visualization**: Once the workflow is complete (either fully automated or after HITL approval), the application displays a comprehensive report detailing findings from each specialist agent, the final Org-AI-R score, and any generated value creation plans. A visual trace of the agent's execution path (Mermaid diagram) is also presented, clearly showing all steps, including any HITL pauses.
7.  **Semantic Memory Integration**: Sarah can choose to store the completed assessment's key outcomes in Mem0, the application's semantic long-term memory. This builds a rich historical context for each company.
8.  **Company History Review**: Sarah can then navigate to the "Company History" page, input a `Company ID`, and retrieve all past assessment outcomes and contextual information stored in Mem0 for that company, demonstrating the system's ability to learn from and leverage previous insights.

## 2. Code Requirements

### Import Statement
```python
from source import * # Imports all functions, classes, and global instances (like agent_memory, global_trace_manager, due_diligence_graph) from source.py
import streamlit as st
import asyncio
import uuid
from datetime import datetime
```

### `st.session_state` Design
The application will heavily rely on `st.session_state` to manage navigation, preserve user inputs, and store the workflow's intermediate and final states across Streamlit reruns and "page" interactions.

*   `st.session_state["current_page"]`:
    *   **Initialization**: `st.session_state.setdefault("current_page", "home")`
    *   **Purpose**: Controls which main content block is displayed in the application, simulating a multi-page experience.
    *   **Values**: `"home"`, `"new_assessment"`, `"assessment_details"`, `"company_history"`
    *   **Update**: Updated by sidebar navigation radio buttons or by action buttons that lead to a different "page".
    *   **Read**: Used at the top level to conditionally render content.

*   `st.session_state["company_id_input"]`:
    *   **Initialization**: `st.session_state.setdefault("company_id_input", "TechInnovate Inc.")`
    *   **Purpose**: Stores the user-entered company ID for a new assessment.
    *   **Update**: Updated when the user types in the `st.text_input` widget on the "New Assessment" page.
    *   **Read**: Used by `run_due_diligence` function call.

*   `st.session_state["assessment_type_input"]`:
    *   **Initialization**: `st.session_state.setdefault("assessment_type_input", "full")`
    *   **Purpose**: Stores the user-selected assessment type.
    *   **Update**: Updated when the user selects an option in the `st.selectbox` widget on the "New Assessment" page.
    *   **Read**: Used by `run_due_diligence` function call.

*   `st.session_state["requested_by_input"]`:
    *   **Initialization**: `st.session_state.setdefault("requested_by_input", "Sarah (Streamlit App)")`
    *   **Purpose**: Stores the name of the user requesting the assessment.
    *   **Update**: Updated when the user types in the `st.text_input` widget on the "New Assessment" page.
    *   **Read**: Used by `run_due_diligence` function call.

*   `st.session_state["latest_workflow_state"]`:
    *   **Initialization**: `st.session_state.setdefault("latest_workflow_state", None)`
    *   **Purpose**: Stores the entire `DueDiligenceState` object from the latest workflow run or resumed workflow. This is critical for displaying results, managing HITL, and resuming the graph.
    *   **Update**: Updated by the return value of `run_due_diligence` and `approve_workflow`.
    *   **Read**: Used extensively on the "Assessment Details" page to display findings, check HITL status, and provide context for subsequent actions.

*   `st.session_state["latest_thread_id"]`:
    *   **Initialization**: `st.session_state.setdefault("latest_thread_id", None)`
    *   **Purpose**: Stores the unique `thread_id` associated with the current active LangGraph workflow. This allows for workflow persistence and resumption.
    *   **Update**: Set when `run_due_diligence` is first called (a new `uuid` is generated if it's None) and passed into subsequent `approve_workflow` calls.
    *   **Read**: Used to retrieve the current workflow state, especially during HITL approval, and to fetch the `AgentTrace` from `global_trace_manager`.

*   `st.session_state["hitl_approval_by"]`:
    *   **Initialization**: `st.session_state.setdefault("hitl_approval_by", "Lead Analyst")`
    *   **Purpose**: Stores the name of the analyst making the HITL decision.
    *   **Update**: Updated when the user types in the `st.text_input` widget on the "Assessment Details" page.
    *   **Read**: Used by `approve_workflow` function call.

*   `st.session_state["hitl_decision"]`:
    *   **Initialization**: `st.session_state.setdefault("hitl_decision", "approved")`
    *   **Purpose**: Stores the "approved" or "rejected" decision for HITL.
    *   **Update**: Updated when the user selects an option in the `st.radio` widget on the "Assessment Details" page.
    *   **Read**: Used by `approve_workflow` function call.

*   `st.session_state["hitl_notes"]`:
    *   **Initialization**: `st.session_state.setdefault("hitl_notes", "")`
    *   **Purpose**: Stores any notes provided during the HITL decision.
    *   **Update**: Updated when the user types in the `st.text_area` widget on the "Assessment Details" page.
    *   **Read**: Used by `approve_workflow` function call.

*   `st.session_state["company_history_id_input"]`:
    *   **Initialization**: `st.session_state.setdefault("company_history_id_input", "RiskyBet Corp.")`
    *   **Purpose**: Stores the company ID for which to retrieve historical memories.
    *   **Update**: Updated when the user types in the `st.text_input` widget on the "Company History" page. Also updated when clicking "View Company History" from "Assessment Details".
    *   **Read**: Used by `agent_memory.get_company_context` function call.

*   `st.session_state["company_memories"]`:
    *   **Initialization**: `st.session_state.setdefault("company_memories", None)`
    *   **Purpose**: Stores the results returned by `agent_memory.get_company_context`.
    *   **Update**: Updated by the return value of `agent_memory.get_company_context`.
    *   **Read**: Used on the "Company History" page to display past assessment information.

### UI Interactions and Function Calls

```python
# Helper to run async functions in Streamlit
def run_async_function(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If event loop is already running, run the coroutine in a new task
        task = loop.create_task(func(*args, **kwargs))
        return asyncio.run(task)
    else:
        # If no event loop is running, run the coroutine directly
        return asyncio.run(func(*args, **kwargs))

# --- Session State Initialization ---
if "current_page" not in st.session_state: st.session_state["current_page"] = "home"
if "company_id_input" not in st.session_state: st.session_state["company_id_input"] = "TechInnovate Inc."
if "assessment_type_input" not in st.session_state: st.session_state["assessment_type_input"] = "full"
if "requested_by_input" not in st.session_state: st.session_state["requested_by_input"] = "Sarah (Streamlit App)"
if "latest_workflow_state" not in st.session_state: st.session_state["latest_workflow_state"] = None
if "latest_thread_id" not in st.session_state: st.session_state["latest_thread_id"] = None
if "hitl_approval_by" not in st.session_state: st.session_state["hitl_approval_by"] = "Lead Analyst"
if "hitl_decision" not in st.session_state: st.session_state["hitl_decision"] = "approved"
if "hitl_notes" not in st.session_state: st.session_state["hitl_notes"] = ""
if "company_history_id_input" not in st.session_state: st.session_state["company_history_id_input"] = "RiskyBet Corp."
if "company_memories" not in st.session_state: st.session_state["company_memories"] = None

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
    load_company_history_callback() # Automatically load history for the selected company

def load_company_history_callback():
    if not st.session_state["company_history_id_input"]:
        st.error("Please enter a Company ID to load history.")
        return

    with st.spinner(f"Loading historical context for '{st.session_state['company_history_id_input']}' from Mem0..."):
        memories = run_async_function(
            agent_memory.get_company_context,
            company_id=st.session_state["company_history_id_input"],
            user_id=st.session_state["requested_by_input"] # Use the current user as context
        )
        st.session_state["company_memories"] = memories
    st.success("Company history loaded!")

# --- Streamlit Application Layout ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Home", "New Assessment", "Assessment Details", "Company History"],
    index=["Home", "New Assessment", "Assessment Details", "Company History"].index(
        "Home" if st.session_state.get("current_page") == "home" else
        "New Assessment" if st.session_state.get("current_page") == "new_assessment" else
        "Assessment Details" if st.session_state.get("current_page") == "assessment_details" else
        "Company History" if st.session_state.get("current_page") == "company_history" else "Home"
    )
)

# Update current_page based on sidebar selection
if page_selection == "Home": st.session_state["current_page"] = "home"
elif page_selection == "New Assessment": st.session_state["current_page"] = "new_assessment"
elif page_selection == "Assessment Details": st.session_state["current_page"] = "assessment_details"
elif page_selection == "Company History": st.session_state["current_page"] = "company_history"

# --- Main Content Area (Conditional Rendering) ---
if st.session_state["current_page"] == "home":
    st.title("AI-Powered Due Diligence for Private Equity")
    st.markdown(f"") # Separator for strict markdown rendering

    st.markdown(f"## Introduction: Empowering PE Analysts with AI-Driven Insights")
    st.markdown(f"") # Separator
    st.markdown(f"**Persona:** Sarah, a Software Developer at \"Synergy Capital,\" a forward-thinking Private Equity (PE) firm.")
    st.markdown(f"") # Separator
    st.markdown(f"**Organization:** Synergy Capital, specializing in acquiring and growing technology companies.")
    st.markdown(f"") # Separator
    st.markdown(f"Sarah's team is tasked with modernizing Synergy Capital's initial due diligence process. Currently, PE analysts spend significant time manually sifting through financial filings, talent reports, and market data to assess potential target companies. This is time-consuming and prone to inconsistencies, delaying critical investment decisions.")
    st.markdown(f"") # Separator
    st.markdown(f"The goal is to develop a proof-of-concept for an AI-powered multi-agent system that automates the initial assessment of a target company's AI-readiness. This system will leverage LangGraph to orchestrate specialized AI agents (e.g., SEC Analysis, Talent Analysis, Scoring, Value Creation), providing a structured, consolidated report to analysts. This allows analysts to quickly grasp a company's profile and focus their expertise on high-value strategic considerations, significantly accelerating the deal pipeline.")
    st.markdown(f"") # Separator
    st.markdown(f"This application demonstrates how to define the workflow state, build specialist agents, orchestrate them with a supervisor agent, integrate human-in-the-loop (HITL) approvals, add semantic memory with Mem0, and visualize agent traces for debugging.")
    st.markdown(f"") # Separator

    st.markdown(f"---")
    st.markdown(f"### Key Concepts Addressed in this Application:")
    st.markdown(f"") # Separator
    st.markdown(f"- Supervisor pattern for agent coordination")
    st.markdown(f"- Specialist agents (SEC, Talent, Scoring, Value)")
    st.markdown(f"- Human-in-the-loop (HITL) approval gates")
    st.markdown(f"- Semantic memory with Mem0")
    st.markdown(f"- Agent debug traces")
    st.markdown(f"") # Separator
    st.markdown(f"Navigate using the sidebar to start a new assessment or view company history.")

elif st.session_state["current_page"] == "new_assessment":
    st.title("Initiate New Due Diligence Assessment")
    st.markdown(f"") # Separator
    st.markdown(f"Enter the details for the company you wish to assess. The multi-agent workflow will analyze the company's AI-readiness and generate a comprehensive report.")
    st.markdown(f"") # Separator

    st.text_input("Company ID", value=st.session_state["company_id_input"], key="company_id_input")
    st.selectbox("Assessment Type", ["screening", "limited", "full"], index=["screening", "limited", "full"].index(st.session_state["assessment_type_input"]), key="assessment_type_input")
    st.text_input("Requested By", value=st.session_state["requested_by_input"], key="requested_by_input")

    st.button("Start Assessment", on_click=start_assessment_callback, type="primary")

elif st.session_state["current_page"] == "assessment_details":
    st.title("Due Diligence Assessment Details")
    st.markdown(f"") # Separator

    if not st.session_state["latest_workflow_state"]:
        st.info("No assessment has been run yet. Please start a new assessment.")
        if st.button("Go to New Assessment"):
            st.session_state["current_page"] = "new_assessment"
            st.rerun()
    else:
        current_state = st.session_state["latest_workflow_state"]

        st.markdown(f"### Workflow Status for {current_state['company_id']}")
        st.markdown(f"") # Separator
        if current_state.get("approval_status") == "pending" and current_state.get("requires_approval"):
            st.warning(f"**Workflow Paused for Human-in-the-Loop (HITL) Approval!**")
            st.markdown(f"**Reason:** {current_state.get('approval_reason', 'N/A')}")
        elif current_state.get("error"):
            st.error(f"**Workflow Ended with Error:** {current_state['error']}")
        elif current_state.get("completed_at"):
            st.success(f"**Workflow Completed Successfully!**")
        else:
            st.info(f"**Workflow is currently running or pending.** (Last update: {current_state['started_at'].strftime('%Y-%m-%d %H:%M:%S')})")
        st.markdown(f"") # Separator

        # Display SEC Analysis
        if current_state.get("sec_analysis"):
            st.markdown(f"### 📄 SEC Analysis Findings")
            st.markdown(f"") # Separator
            sec_data = current_state["sec_analysis"]
            st.markdown(f"- **Company ID:** {sec_data.get('company_id')}")
            st.markdown(f"- **Findings Summary:** {sec_data.get('findings')}")
            st.markdown(f"- **Evidence Count:** {sec_data.get('evidence_count')}")
            st.markdown(f"- **Dimensions Covered:** {', '.join(sec_data.get('dimensions_covered', []))}")
            st.markdown(f"- **Confidence:** {sec_data.get('confidence')}")
            st.markdown(f"") # Separator

        # Display Talent Analysis
        if current_state.get("talent_analysis"):
            st.markdown(f"### 🧑‍💻 Talent Analysis Findings")
            st.markdown(f"") # Separator
            talent_data = current_state["talent_analysis"]
            st.markdown(f"- **Company ID:** {talent_data.get('company_id')}")
            st.markdown(f"- **AI Role Count:** {talent_data.get('ai_role_count')}")
            st.markdown(f"- **Talent Concentration:** {talent_data.get('talent_concentration', 0.0):.2%}")
            st.markdown(f"- **Seniority Index:** {talent_data.get('seniority_index', 0.0):.1f}")
            st.markdown(f"- **Key Skills:** {', '.join(talent_data.get('key_skills', []))}")
            st.markdown(f"- **Hiring Trend:** {talent_data.get('hiring_trend')}")
            st.markdown(f"") # Separator

        # Display Scoring Result
        if current_state.get("scoring_result"):
            st.markdown(f"### 📊 Org-AI-R Scoring Result")
            st.markdown(f"") # Separator
            scoring_data = current_state["scoring_result"]
            final_score = scoring_data.get('final_score', 0.0)
            st.markdown(f"- **Final Org-AI-R Score:** **{final_score:.1f}**")
            st.markdown(f"- **Details:** {scoring_data.get('details', 'N/A')}")
            st.markdown(f"") # Separator
            st.markdown(r"$$ (S < S_{\text{min}}) \lor (S > S_{\text{max}}) $$")
            st.markdown(r"where $S$ is the Org-AI-R score, $S_{\text{min}}=40$, and $S_{\text{max}}=85$.")
            st.markdown(f"") # Separator

        # Display Value Creation Plan
        if current_state.get("value_creation_plan"):
            st.markdown(f"### 💰 Value Creation Plan")
            st.markdown(f"") # Separator
            value_data = current_state["value_creation_plan"]
            st.markdown(f"- **Current Org-AI-R Score:** {value_data.get('current_score', 0.0):.1f}")
            st.markdown(f"- **Target Org-AI-R Score:** {value_data.get('target_score', 0.0):.1f}")
            st.markdown(f"- **Projected EBITDA Impact:** **{value_data.get('projected_ebitda_impact_pct', 0.0):.1%}**")
            st.markdown(f"- **Timeline (Months):** {value_data.get('timeline_months')}")
            st.markdown(f"- **Key Initiatives:**")
            st.markdown(f"") # Separator
            for initiative in value_data.get('initiatives', []):
                st.markdown(f"  - **{initiative.get('name')}**: Impact: {initiative.get('impact')}, Cost: ${initiative.get('cost_mm', 0.0):.1f}M")
            st.markdown(f"") # Separator
            st.markdown(r"$$ I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD} $$")
            st.markdown(r"where $I$ is the projected EBITDA impact and $\text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD}$ is a predefined threshold (e.g., 7%).")
            st.markdown(f"") # Separator

        # HITL Approval Section
        if current_state.get("requires_approval") and current_state.get("approval_status") == "pending":
            st.markdown(f"---")
            st.subheader("Human-in-the-Loop (HITL) Approval Required")
            st.markdown(f"") # Separator
            st.warning(f"The workflow for '{current_state['company_id']}' requires your review before proceeding.")
            st.markdown(f"**Reason for HITL:** {current_state.get('approval_reason', 'N/A')}")
            st.markdown(f"") # Separator

            st.text_input("Approved By", value=st.session_state["hitl_approval_by"], key="hitl_approval_by")
            st.radio("Decision", ["approved", "rejected"], index=0 if st.session_state["hitl_decision"] == "approved" else 1, key="hitl_decision")
            st.text_area("Notes", value=st.session_state["hitl_notes"], key="hitl_notes")

            st.button("Submit Approval", on_click=submit_approval_callback, type="secondary")
            st.markdown(f"") # Separator

        # Workflow Trace Visualization
        st.markdown(f"---")
        st.subheader("Agent Workflow Trace")
        st.markdown(f"") # Separator
        st.markdown(f"This diagram visualizes the execution path of the multi-agent system, including any HITL pauses.")
        st.markdown(f"") # Separator
        
        trace_id_for_mermaid = current_state.get("trace_id")
        if trace_id_for_mermaid:
            current_trace = global_trace_manager.get_trace(trace_id_for_mermaid)
            if current_trace:
                st.markdown(current_trace.to_mermaid())
            else:
                st.info(f"Trace for ID {trace_id_for_mermaid} not found in manager. It might have already been moved to completed traces.")
        else:
            st.info("No trace ID available for this workflow.")
        st.markdown(f"") # Separator

        # Action buttons (Store Outcome, View Company History)
        if not current_state.get("requires_approval") and current_state.get("completed_at") and not current_state.get("error"):
            col1, col2 = st.columns(2)
            with col1:
                st.button("Store Outcome in Mem0", on_click=store_outcome_callback, type="secondary")
            with col2:
                st.button("View Company History", on_click=view_history_callback, type="secondary")
            st.markdown(f"") # Separator


elif st.session_state["current_page"] == "company_history":
    st.title("Company Assessment History")
    st.markdown(f"") # Separator
    st.markdown(f"Retrieve past assessment outcomes and contextual information for a specific company from Mem0.")
    st.markdown(f"") # Separator

    st.text_input("Company ID for History", value=st.session_state["company_history_id_input"], key="company_history_id_input")
    st.button("Load Company History", on_click=load_company_history_callback, type="primary")
    st.markdown(f"") # Separator

    if st.session_state["company_memories"]:
        st.subheader(f"Memories for {st.session_state['company_memories']['company_id']}")
        st.markdown(f"") # Separator
        if st.session_state["company_memories"]['memory_count'] > 0:
            for i, mem in enumerate(st.session_state["company_memories"]["memories"]):
                st.markdown(f"---")
                st.markdown(f"**Memory {i+1} (ID: {mem.get('id', 'N/A')})**")
                st.markdown(f"") # Separator
                st.markdown(f"**Content:** {mem.get('content', 'N/A')}")
                st.markdown(f"") # Separator
                st.markdown(f"**Metadata:**")
                st.markdown(f"") # Separator
                for k, v in mem.get('metadata', {}).items():
                    st.markdown(f"  - {k}: {v}")
                st.markdown(f"---")
                st.markdown(f"") # Separator
        else:
            st.info(f"No historical memories found for '{st.session_state['company_history_id_input']}'.")
    else:
        st.info("No company history loaded yet.")
```
