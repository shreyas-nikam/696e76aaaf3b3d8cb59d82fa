id: 696e76aaaf3b3d8cb59d82fa_documentation
summary: Lab 10: LangGraph Multi-Agent Orchestration Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 10: LangGraph Multi-Agent Orchestration for AI-Powered Due Diligence

## 1. Introduction: Empowering PE Analysts with AI-Driven Insights
Duration: 0:10:00

<aside class="positive">
This step provides the essential context for understanding the application's purpose, the problem it solves, and the key AI concepts it demonstrates. Understanding these foundational elements will help you grasp the "why" and "what" before diving into the "how."
</aside>

### Application Overview

This codelab guides you through a Streamlit application that showcases an AI-powered multi-agent system for due diligence in Private Equity (PE). The application, named "QuLab: Lab 10: LangGraph Multi-Agent Orchestration," is designed to automate and enhance the initial assessment of a target company's AI-readiness, significantly accelerating the deal pipeline for PE firms.

**Persona:** Sarah, a Software Developer at "Synergy Capital," a forward-thinking Private Equity (PE) firm.
**Organization:** Synergy Capital, specializing in acquiring and growing technology companies.

Currently, PE analysts at Synergy Capital spend considerable time manually sifting through financial filings, talent reports, and market data to assess potential target companies. This process is time-consuming, prone to inconsistencies, and delays critical investment decisions. This application addresses these challenges by providing a proof-of-concept for an AI-powered system that automates the initial assessment, allowing analysts to quickly grasp a company's profile and focus their expertise on high-value strategic considerations.

### Importance and Concepts Explained

The application demonstrates a sophisticated approach to building intelligent systems using **LangGraph**, a library for building robust, stateful, and multi-actor applications with LLMs. It focuses on the following key concepts:

*   **Supervisor Pattern for Agent Coordination:** A central "Supervisor" agent orchestrates the workflow, delegating tasks to specialized agents based on the current state and requirements. This mimics how a human team lead would manage different experts.
*   **Specialist Agents:** The system comprises several domain-specific AI agents, each designed to handle a particular aspect of due diligence:
    *   **SEC Analysis Agent:** Analyzes public financial filings (e.g., 10-K reports) for insights relevant to AI strategy.
    *   **Talent Analysis Agent:** Assesses the company's talent pool, specifically focusing on AI-related roles, skills, and hiring trends.
    *   **Scoring Agent:** Computes a comprehensive "Org-AI-R Score" based on the findings from other agents, indicating the company's overall AI readiness.
    *   **Value Creation Agent:** Identifies potential value creation initiatives to improve the company's AI posture and projects their financial impact.
*   **Human-in-the-Loop (HITL) Approval Gates:** Critical decision points in the workflow where human intervention and approval are required before the process can continue. This ensures oversight and allows human analysts to validate or steer AI recommendations.
*   **Semantic Memory with Mem0:** Integration with Mem0 provides long-term, semantic memory for agents. This allows the system to recall past assessment outcomes and contextual information for companies, enabling more informed decisions and reducing redundant analysis.
*   **Agent Debug Traces:** The application provides a visual trace of the agent workflow using Mermaid diagrams, which is invaluable for understanding, debugging, and optimizing complex multi-agent interactions.

### High-Level Architecture

The multi-agent system operates on a predefined workflow managed by a supervisor.

1.  **Initiation:** A user (PE analyst) initiates an assessment for a target company via the Streamlit UI.
2.  **Supervisor Orchestration:** A central supervisor agent receives the request and, based on the `assessment_type`, routes the task to appropriate specialist agents.
3.  **Specialist Agent Execution:**
    *   The **SEC Analysis Agent** extracts relevant insights from financial documents.
    *   The **Talent Analysis Agent** evaluates the company's AI talent landscape.
4.  **Conditional Routing & HITL:** Based on the findings (e.g., a low score or high projected impact), the workflow might pause for **Human-in-the-Loop (HITL) Approval**.
5.  **Scoring & Value Creation:** If approved (or if HITL wasn't triggered), the **Scoring Agent** calculates the Org-AI-R score, and the **Value Creation Agent** proposes improvement initiatives.
6.  **Outcome Storage:** The final assessment outcome, including key findings and scores, is stored in **Mem0** for future reference.
7.  **Trace Visualization:** Throughout the process, the execution path of the agents is logged and visualized as a Mermaid trace.

This system aims to transform the due diligence process from a manual, time-intensive effort into an efficient, AI-augmented workflow.

## 2. Setting Up Your Environment
Duration: 0:05:00

Before running the Streamlit application, you need to set up your Python environment and configure necessary API keys.

1.  **Clone the Repository (or create files):**
    Ensure you have `app.py` (the provided Streamlit code) and a `source.py` file. The `source.py` file is assumed to contain the LangGraph definitions and agent logic. For this codelab, we will assume `source.py` exists and correctly defines the `run_due_diligence`, `approve_workflow`, `agent_memory`, `verify_agent_memory`, and `global_trace_manager` objects/functions.

2.  **Install Dependencies:**
    Open your terminal or command prompt and run the following command to install the required Python libraries:

    ```bash
    pip install streamlit langchain_community langchain langgraph mem0 openai python-dotenv nest_asyncio
    ```
    <aside class="positive">
    It's good practice to create a `requirements.txt` file with these dependencies and then use `pip install -r requirements.txt`.
    </aside>

3.  **Set Environment Variables:**
    The application will likely require API keys for Large Language Models (LLMs) (e.g., OpenAI) and potentially for Mem0. Create a `.env` file in the root directory of your project and add your keys:

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    MEM0_API_KEY="your_mem0_api_key_here"
    # Other potential environment variables like specific model names, thresholds, etc.
    ```
    <aside class="negative">
    Never hardcode API keys directly into your code. Always use environment variables for sensitive information.
    </aside>

4.  **Understanding `nest_asyncio`:**
    The `app.py` file includes `nest_asyncio.apply()`. This is crucial for Streamlit applications that need to run asynchronous functions (like those in `langchain` and `langgraph`) synchronously within the Streamlit event loop. Without `nest_asyncio`, you might encounter errors if an asyncio event loop is already running when `asyncio.run()` is called.

    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

## 3. Core Workflow: LangGraph Multi-Agent System
Duration: 0:15:00

This step describes the underlying multi-agent architecture and how LangGraph orchestrates the various agents to perform the due diligence process. While the actual LangGraph definition resides in `source.py` (not provided here), understanding the conceptual flow is vital.

### Workflow State

The central mechanism for information sharing between agents is the **workflow state**. This is a dictionary-like object that holds all relevant data generated during the assessment. As each agent completes its task, it updates this state, making its findings available to subsequent agents.

A typical workflow state might look like this:

```python
class CompanyState(TypedDict):
    company_id: str
    assessment_type: Literal["screening", "limited", "full"]
    requested_by: str
    started_at: datetime
    last_update: datetime
    status: Literal["running", "pending_approval", "completed", "error"]
    error: Optional[str]

    # Agent-specific outputs
    sec_analysis: Optional[dict]
    talent_analysis: Optional[dict]
    scoring_result: Optional[dict]
    value_creation_plan: Optional[dict]

    # HITL specific fields
    requires_approval: bool
    approval_reason: Optional[str]
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    approved_by: Optional[str]
    hitl_notes: Optional[str]
    approved_at: Optional[datetime]

    # Trace and Memory
    trace_id: str
    # Other context information...
```

### Specialist Agents

Each agent is an LLM-powered component with a specific role:

*   **SEC Analysis Agent:**
    *   **Input:** `company_id`, `assessment_type`
    *   **Output (updates state):** `sec_analysis` (e.g., `{'findings': 'AI mentioned in risk factors...', 'evidence_count': 5, ...}`)
    *   **Purpose:** Simulates analysis of public filings (like 10-K reports) to identify strategic mentions of AI, risks, and opportunities.

*   **Talent Analysis Agent:**
    *   **Input:** `company_id`, `assessment_type`
    *   **Output (updates state):** `talent_analysis` (e.g., `{'ai_role_count': 15, 'talent_concentration': 0.12, 'key_skills': ['MLOps', 'Generative AI']}`)
    *   **Purpose:** Simulates analysis of talent data (e.g., LinkedIn profiles, job postings) to gauge the company's AI talent pool and strategy.

*   **Scoring Agent:**
    *   **Input:** `sec_analysis`, `talent_analysis`, `assessment_type`
    *   **Output (updates state):** `scoring_result` (e.g., `{'final_score': 72.5, 'details': 'Strong talent, moderate SEC mentions.'}`)
    *   **Purpose:** Synthesizes findings from other agents to compute a quantitative "Org-AI-R Score" indicating AI readiness. It may also introduce conditions for HITL, such as if the score falls below a certain threshold.

*   **Value Creation Agent:**
    *   **Input:** `scoring_result`, `assessment_type`
    *   **Output (updates state):** `value_creation_plan` (e.g., `{'initiatives': [{'name': 'AI Upskilling Program', 'impact': 'High', 'cost_mm': 2.5}], 'projected_ebitda_impact_pct': 0.08}`)
    *   **Purpose:** Proposes concrete initiatives to improve the company's AI capabilities and projects their potential impact on financial metrics like EBITDA. This agent also has conditions for HITL based on projected impact.

### Supervisor Agent and LangGraph

The **Supervisor Agent** acts as the orchestrator. It doesn't perform core analytical tasks itself but decides which specialist agent to invoke next based on the current state and a predefined graph structure. LangGraph is used to define this graph, including:

*   **Nodes:** Each specialist agent or decision point (like HITL) is a node in the graph.
*   **Edges:** Define the flow between nodes.
*   **Conditional Edges:** Allow for dynamic routing. For example, after the Scoring Agent, a conditional edge might check if an approval is needed.

### Human-in-the-Loop (HITL) Integration

HITL is a crucial component, ensuring human oversight at critical junctures. The workflow is designed to pause and await human input if certain conditions are met:

*   **Low Org-AI-R Score:** If the `Scoring Agent` calculates a score below a configured minimum (e.g., 40), the workflow might pause for review.
*   **High Projected EBITDA Impact:** If the `Value Creation Agent` projects a very high EBITDA impact (e.g., above 7%), a human review might be triggered to validate such an aggressive projection.

When HITL is triggered, the `approval_status` in the workflow state changes to "pending", and the Streamlit UI displays an approval form. The workflow resumes only after a human decision ("approved" or "rejected") is submitted.

### Semantic Memory with Mem0

The `agent_memory` object (presumably from the `source.py` module) handles interaction with Mem0. Mem0 allows for:

*   **Storing Assessment Outcomes:** After an assessment is completed, key findings, scores, and other metadata are stored semantically.
*   **Retrieving Company Context:** When a new assessment is initiated or history is viewed, Mem0 can retrieve past assessments, providing valuable historical context to agents or for human review. This prevents agents from "forgetting" past interactions and analyses.

### Conceptual Workflow Diagram

Here's a simplified view of the multi-agent workflow:

```mermaid
graph TD
    A[Start Assessment (Streamlit UI)] --> B(Initialize Workflow State);
    B --> C{Supervisor: Route Task};
    C --> D[SEC Analysis Agent];
    C --> E[Talent Analysis Agent];
    D & E --> F[Scoring Agent];
    F --> G{Scoring Agent: Check for HITL?};
    G -- Score < 40 --> H[HITL: Pending Approval];
    G -- Score >= 40 --> I[Value Creation Agent];
    H -- Approved --> I;
    H -- Rejected --> J[End Workflow: Rejected];
    I --> K{Value Creation Agent: Check for HITL?};
    K -- Projected Impact > 7% --> L[HITL: Pending Approval];
    K -- Projected Impact <= 7% --> M[Store Outcome in Mem0];
    L -- Approved --> M;
    L -- Rejected --> J;
    M --> N[End Workflow: Completed];
    N --> P(Display Results / Trace);
    J --> P;
```

## 4. Deep Dive into the Streamlit Application (`app.py`)
Duration: 0:15:00

The `app.py` file provides the user interface (UI) for interacting with the LangGraph multi-agent system. It uses Streamlit's capabilities for state management, user input, and displaying results.

### Page Configuration and Initialization

*   `st.set_page_config(page_title="QuLab: Lab 10: LangGraph Multi-Agent Orchestration", layout="wide")`: Configures the Streamlit page for a wider layout.
*   `nest_asyncio.apply()`: Applied globally to allow asynchronous functions to run within the Streamlit application's event loop.
*   **Session State Initialization:**
    Streamlit's `st.session_state` is used to maintain the application's state across reruns. This is crucial for:
    *   `current_page`: Tracks which section of the application the user is currently viewing (e.g., "home", "new_assessment").
    *   `company_id_input`, `assessment_type_input`, `requested_by_input`: Store user inputs for a new assessment.
    *   `latest_workflow_state`, `latest_thread_id`: Hold the most recent state and ID of an ongoing or completed LangGraph workflow.
    *   `hitl_approval_by`, `hitl_decision`, `hitl_notes`: Store inputs for Human-in-the-Loop decisions.
    *   `company_memories`: Stores historical assessment data retrieved from Mem0.

### Helper Function: `run_async_function`

```python
def run_async_function(func, *args, **kwargs):
    """
    Helper function to run an async function synchronously.
    With nest_asyncio.apply() called globally, asyncio.run can be called
    even if an event loop is already running, simplifying this function.
    """
    return asyncio.run(func(*args, **kwargs))
```
This utility function allows the synchronous Streamlit environment to execute asynchronous functions, such as `run_due_diligence` and `approve_workflow`, which are defined in `source.py`.

### Callbacks for UI Interactions

Streamlit buttons and input widgets often trigger functions called "callbacks" when their state changes or they are clicked.

*   `start_assessment_callback()`:
    *   Triggered by the "Start Assessment" button.
    *   Sets `current_page` to "assessment_details".
    *   Generates a unique `thread_id` for the LangGraph workflow.
    *   Calls `run_async_function(run_due_diligence, ...)` to initiate the multi-agent workflow.
    *   Updates `st.session_state["latest_workflow_state"]` with the workflow's output.
    *   Provides status messages (warning for HITL, success, or error).

*   `submit_approval_callback()`:
    *   Triggered by the "Submit Approval" button in the HITL section.
    *   Calls `run_async_function(approve_workflow, ...)` with the human's decision.
    *   Updates `st.session_state["latest_workflow_state"]` with the workflow's updated state.

*   `store_outcome_callback()`:
    *   Triggered by the "Store Outcome in Mem0" button.
    *   Extracts relevant information from `latest_workflow_state`.
    *   Calls `run_async_function(agent_memory.store_assessment_outcome, ...)` to persist the findings.

*   `view_history_callback()`:
    *   Triggered by "View Company History" button.
    *   Navigates to the "company_history" page and triggers `load_company_history_callback`.

*   `load_company_history_callback()`:
    *   Triggered by "Load Company History" button or `view_history_callback`.
    *   Calls `run_async_function(agent_memory.get_company_context, ...)` to retrieve data from Mem0.
    *   Stores the retrieved `memories` in `st.session_state`.

### Streamlit Application Layout

The application uses `st.sidebar.radio` for navigation between four main pages:

1.  **Home Page (`st.session_state["current_page"] == "home"`):**
    *   Provides an introduction to the application, its purpose, and the key concepts.
    *   Explains the "why" behind the multi-agent system.

2.  **New Assessment Page (`st.session_state["current_page"] == "new_assessment"`):**
    *   Allows users to input `Company ID`, `Assessment Type` (screening, limited, full), and `Requested By`.
    *   The "Start Assessment" button triggers `start_assessment_callback`.

3.  **Assessment Details Page (`st.session_state["current_page"] == "assessment_details"`):**
    *   Displays the real-time status and detailed outputs of the ongoing or completed assessment.
    *   Shows findings from SEC Analysis, Talent Analysis, Org-AI-R Scoring, and Value Creation Plan.
    *   **Org-AI-R Scoring Formula:**
        $$ (S < S_{\text{min}}) \lor (S > S_{\text{max}}) $$
        where $S$ is the Org-AI-R score, $S_{\text{min}}=40$, and $S_{\text{max}}=85$. This condition might trigger a HITL.
    *   **EBITDA Impact Formula:**
        $$ I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD} $$
        where $I$ is the projected EBITDA impact and $\text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD}$ is a predefined threshold (e.g., 7%). This condition might also trigger a HITL.
    *   **Human-in-the-Loop (HITL) Section:** Appears when `approval_status` is "pending". Provides input fields for `Approved By`, `Decision` (approved/rejected), and `Notes`, with a "Submit Approval" button.
    *   **Agent Workflow Trace:** Uses `st.markdown(current_trace.to_mermaid())` to render a visual graph of the agent's execution path. This is invaluable for debugging and understanding the flow.
    *   Action buttons ("Store Outcome in Mem0", "View Company History") appear upon workflow completion without requiring HITL or after HITL approval.

4.  **Company History Page (`st.session_state["current_page"] == "company_history"`):**
    *   Allows users to input a `Company ID` and click "Load Company History".
    *   Retrieves and displays past assessment outcomes (memories) for that company from Mem0, including their content and metadata.

## 5. Initiating a New Assessment
Duration: 0:05:00

This step walks you through the process of starting a new due diligence assessment using the Streamlit application.

1.  **Navigate to "New Assessment":**
    In the left sidebar, select "New Assessment".

2.  **Enter Assessment Details:**
    You will see input fields to provide details for the target company:
    *   **Company ID:** Enter a unique identifier for the company (e.g., "TechInnovate Inc."). This is pre-filled with a default value.
    *   **Assessment Type:** Choose the depth of the assessment. Options include "screening", "limited", or "full". A "full" assessment typically involves all specialist agents.
    *   **Requested By:** Enter your name or a reference (e.g., "Sarah (Streamlit App)").

    Example inputs:
    *   Company ID: `FutureFusion Corp.`
    *   Assessment Type: `full`
    *   Requested By: `YourName`

3.  **Start the Workflow:**
    Click the "Start Assessment" button.

    <aside class="positive">
    Observe the console where your Streamlit app is running. You might see logs from the LangGraph workflow indicating which agents are being invoked.
    </aside>

4.  **Monitor Progress:**
    The application will transition to the "Assessment Details" page. A spinner will appear, indicating that the multi-agent due diligence workflow is running.
    Once the initial part of the workflow completes, you will see the status update. Depending on the `assessment_type` and the simulated results, the workflow might pause for Human-in-the-Loop (HITL) approval.

    *   If HITL is triggered, a warning message will appear: **"Workflow paused for HITL approval!"**
    *   If the workflow completes without HITL or after HITL approval, a success message will appear: **"Workflow completed successfully."**
    *   If any errors occur, an error message will be displayed.

## 6. Reviewing Assessment Details and Handling HITL
Duration: 0:10:00

Once an assessment is running or completed, the "Assessment Details" page provides a comprehensive view of the findings and allows for human intervention.

1.  **Access Assessment Details:**
    After starting an assessment, the app automatically navigates to this page. You can also manually select "Assessment Details" from the sidebar.

2.  **Review Workflow Status and Agent Outputs:**
    The page will display the current status of the workflow for the `company_id` you entered.
    Scroll down to see the detailed findings from each specialist agent:

    *   **📄 SEC Analysis Findings:** Provides a summary of insights from simulated SEC filings, including findings, evidence count, dimensions covered, and confidence.
    *   **🧑‍💻 Talent Analysis Findings:** Shows details about the company's AI talent, such as AI role count, talent concentration, key skills, and hiring trends.
    *   **📊 Org-AI-R Scoring Result:** Displays the calculated `final_score` and details on how it was derived. Remember the formula for score thresholds:
        $$ (S < S_{\text{min}}) \lor (S > S_{\text{max}}) $$
    *   **💰 Value Creation Plan:** Outlines proposed initiatives to enhance AI capabilities, including current and target Org-AI-R scores, projected EBITDA impact, and a timeline. The EBITDA impact formula used for HITL is:
        $$ I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD} $$

3.  **Handle Human-in-the-Loop (HITL) Approval:**
    If the workflow triggered a HITL event (e.g., due to a low Org-AI-R score or high projected EBITDA impact), a dedicated section will appear:

    *   **"Human-in-the-Loop (HITL) Approval Required"**
    *   It will state the **Reason for HITL**.
    *   You need to provide:
        *   **Approved By:** Your name or role (e.g., "Lead Analyst").
        *   **Decision:** Choose "approved" or "rejected" using the radio buttons.
        *   **Notes:** Add any comments or rationale for your decision.

    After filling in the details, click "Submit Approval". The workflow will then resume or terminate based on your decision.

4.  **Visualize the Agent Workflow Trace:**
    Scroll down to the "Agent Workflow Trace" section. This is a critical debugging and understanding tool.

    *   The application dynamically generates a **Mermaid diagram** representing the execution path of your multi-agent system.
    *   Each box in the diagram represents an agent or a decision point. Arrows indicate the flow of execution.
    *   If HITL was triggered, you will see the workflow pause at the HITL node until your approval is submitted, and then resume.
    *   This visualization helps you understand exactly which agents were called, in what order, and where conditional routing occurred.

## 7. Storing Outcomes and Viewing Company History
Duration: 0:05:00

This step focuses on how the application leverages Mem0 for long-term memory and how you can review past assessment data.

1.  **Store Outcome in Mem0:**
    Once an assessment has completed (and all HITL approvals, if any, have been processed), you will see an "Store Outcome in Mem0" button.

    *   Click this button to save the current assessment's key findings, scores, and metadata into the Mem0 semantic memory store.
    *   A success message, **"Assessment outcome for '{Company ID}' stored in Mem0!"**, will confirm the operation.
    *   Storing outcomes is crucial for building a knowledge base that agents can refer to in future assessments.

2.  **View Company History:**
    After an assessment is completed and optionally stored, you can view its history:

    *   Click the "View Company History" button (if available) or navigate to the "Company History" page from the sidebar.
    *   The `Company ID for History` field will be pre-filled with the last assessed company's ID. You can change this to any other company ID for which you expect history to exist.
    *   Click "Load Company History".

3.  **Review Historical Memories:**
    The application will query Mem0 and display any historical assessment outcomes associated with the entered `Company ID`.

    *   For each memory found, you will see:
        *   **Memory ID:** A unique identifier for the stored memory.
        *   **Content:** A summary of the assessment outcome (e.g., key findings, final score).
        *   **Metadata:** Additional structured information about the memory, such as `assessment_type`, `user_id`, and `created_at`.
    *   If no history is found, an informational message, **"No historical memories found for '{Company ID}'"**, will be displayed.

    <aside class="positive">
    Semantic memory is powerful. Imagine if a new assessment for a company automatically brought up its past "red flags" or "success stories" identified by previous AI assessments or human reviews. This provides a rich context for the current analysis.
    </aside>

## 8. Conclusion and Next Steps
Duration: 0:05:00

Congratulations! You have successfully explored the "QuLab: Lab 10: LangGraph Multi-Agent Orchestration" application.

### What You've Learned

In this codelab, you've gained an understanding of:

*   How a Streamlit application can serve as an interactive front-end for complex AI systems.
*   The principles of multi-agent orchestration using the supervisor pattern with LangGraph.
*   The role of specialized AI agents (SEC Analysis, Talent Analysis, Scoring, Value Creation) in breaking down complex tasks.
*   The importance and implementation of Human-in-the-Loop (HITL) for critical decision points and validation.
*   How semantic memory (with Mem0) can be integrated to provide long-term context and improve agent performance over time.
*   The utility of agent workflow traces for visualizing, debugging, and understanding multi-agent interactions.

This application provides a robust framework for building intelligent, autonomous systems that augment human decision-making, especially in data-intensive fields like Private Equity.

### Potential Improvements and Next Steps

This is a proof-of-concept, and there are many ways to extend and enhance this application:

*   **Real Data Integration:** Instead of simulated data, integrate with actual financial APIs, talent data platforms, or internal databases.
*   **More Sophisticated Agents:** Develop more complex agents that can perform deeper analysis, utilize more tools (e.g., web search, database queries), or interact with external APIs.
*   **Dynamic Agent Selection:** Implement more advanced routing logic in the supervisor, allowing it to dynamically select agents based on the specific nuances of the `company_id` or `assessment_type`.
*   **Feedback Mechanism:** Integrate a feedback loop where human analysts can provide explicit feedback on agent outputs, which can then be used to fine-tune or improve agent performance.
*   **Enhanced UI/UX:** Improve the user interface with more interactive visualizations, better error handling, and more intuitive controls.
*   **Complex HITL Scenarios:** Implement more granular HITL conditions, such as allowing approval on specific sections of a report rather than the entire workflow.
*   **Security and Permissions:** Add user authentication and role-based access control for different types of analysts or decision-makers.
*   **Deployment:** Containerize the application (e.g., using Docker) and deploy it to a cloud platform (e.g., Google Cloud Run, AWS ECS, Azure Container Apps).

By building upon the foundational concepts demonstrated in this codelab, you can create even more powerful and intelligent AI solutions tailored to specific business needs.

<button>
  [Download Project Code (Placeholder)](https://github.com/your-repo-link/qu-lab-10)
</button>
