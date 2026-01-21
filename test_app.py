
import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Mock data for workflow states and memories
mock_workflow_state_success = {
    "company_id": "TechInnovate Inc.",
    "assessment_type": "full",
    "requested_by": "Sarah (Streamlit App)",
    "thread_id": "dd-techinnovate_inc-2026-01-21T14:43:00",
    "started_at": datetime.utcnow() - timedelta(minutes=5),
    "completed_at": datetime.utcnow(),
    "requires_approval": False,
    "approval_status": "approved",
    "sec_analysis": {
        "company_id": "TechInnovate Inc.",
        "findings": "Strong presence in AI-related patents.",
        "evidence_count": 5,
        "dimensions_covered": ["innovation", "market_position"],
        "confidence": "high"
    },
    "talent_analysis": {
        "company_id": "TechInnovate Inc.",
        "ai_role_count": 150,
        "talent_concentration": 0.35,
        "seniority_index": 7.2,
        "key_skills": ["MLOps", "Deep Learning", "NLP"],
        "hiring_trend": "increasing"
    },
    "scoring_result": {
        "final_score": 78.5,
        "details": "High score due to strong talent and innovation."
    },
    "value_creation_plan": {
        "current_score": 78.5,
        "target_score": 90.0,
        "projected_ebitda_impact_pct": 0.12,
        "timeline_months": 18,
        "initiatives": [
            {"name": "AI Platform Modernization", "impact": "High", "cost_mm": 5.0},
            {"name": "Data Governance Program", "impact": "Medium", "cost_mm": 2.5}
        ]
    },
    "trace_id": "test_trace_success"
}

mock_workflow_state_pending = {
    "company_id": "RiskyBet Corp.",
    "assessment_type": "full",
    "requested_by": "Sarah (Streamlit App)",
    "thread_id": "dd-riskybet_corp-2026-01-21T14:45:00",
    "started_at": datetime.utcnow() - timedelta(minutes=2),
    "completed_at": None,
    "requires_approval": True,
    "approval_status": "pending",
    "approval_reason": "Projected EBITDA impact exceeds threshold.",
    "scoring_result": {
        "final_score": 88.0,
        "details": "High score, but significant projected impact."
    },
    "value_creation_plan": {
        "projected_ebitda_impact_pct": 0.08,
        "initiatives": [{"name": "Aggressive Expansion", "impact": "Very High", "cost_mm": 10.0}]
    },
    "trace_id": "test_trace_pending"
}

mock_workflow_state_approved_after_hitl = {
    **mock_workflow_state_pending, # Start with pending state attributes
    "completed_at": datetime.utcnow(),
    "approval_status": "approved",
    "requires_approval": False,
    "hitl_approved_by": "Lead Analyst",
    "hitl_decision_notes": "Approved due to strategic importance."
}

mock_workflow_state_rejected_after_hitl = {
    **mock_workflow_state_pending, # Start with pending state attributes
    "completed_at": datetime.utcnow(),
    "approval_status": "rejected",
    "requires_approval": False,
    "hitl_approved_by": "Lead Analyst",
    "hitl_decision_notes": "Rejected due to high risk and cost."
}

mock_company_memories = {
    "company_id": "RiskyBet Corp.",
    "user_id": "Sarah (Streamlit App)",
    "memory_count": 2,
    "memories": [
        {
            "id": "mem-1",
            "content": "Assessment outcome for RiskyBet Corp. (full): Final Score 78.5. Key findings: SEC Analysis: Strong presence..., Talent Analysis: 150 AI roles..., Value Creation Initiatives: AI Platform..., HITL triggered due to: Projected EBITDA impact...",
            "metadata": {
                "assessment_type": "full",
                "final_score": 78.5,
                "timestamp": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "user_id": "Sarah (Streamlit App)"
            }
        },
        {
            "id": "mem-2",
            "content": "Assessment outcome for RiskyBet Corp. (screening): Final Score 65.0. Key findings: No specific key findings generated during assessment.",
            "metadata": {
                "assessment_type": "screening",
                "final_score": 65.0,
                "timestamp": (datetime.utcnow() - timedelta(days=60)).isoformat(),
                "user_id": "Sarah (Streamlit App)"
            }
        }
    ]
}

# Mock for global_trace_manager
mock_global_trace_manager = MagicMock()
mock_trace_mermaid_content = "graph TD\n    A[Start] --> B(Agent 1)\n    B --> C{Decision}\n    C --> D[Agent 2]\n    D --> E[End]"

class MockTrace:
    def to_mermaid(self):
        return mock_trace_mermaid_content

mock_global_trace_manager.get_trace.return_value = MockTrace()

# Patching the source module's functions and global_trace_manager
@patch("source.run_due_diligence")
@patch("source.approve_workflow")
@patch("source.agent_memory.store_assessment_outcome")
@patch("source.agent_memory.get_company_context")
@patch("source.global_trace_manager", new=mock_global_trace_manager)
class TestStreamlitApp:
    """
    Comprehensive tests for the Streamlit application using AppTest.
    All patches are applied at the class level to avoid repeated patching
    for each test method. The mock objects are passed as arguments to each test.
    """

    def test_initial_page_load(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test that the app loads correctly on the home page."""
        at = AppTest.from_file("app.py").run()

        # Assertions for initial home page content
        assert at.title[0].value == "QuLab: Lab 10: LangGraph Multi-Agent Orchestration"
        assert at.sidebar.radio[0].value == "Home"
        assert at.markdown[2].value == "AI-Powered Due Diligence for Private Equity"
        assert "Key Concepts Addressed in this Application:" in at.markdown[at.markdown.index("### Key Concepts Addressed in this Application:")].value
        # Ensure no accidental calls to mocked functions on initial load
        mock_run_due_diligence.assert_not_called()
        mock_approve_workflow.assert_not_called()
        mock_store_outcome.assert_not_called()
        mock_get_company_context.assert_not_called()


    def test_navigate_to_new_assessment(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test navigation to the New Assessment page."""
        at = AppTest.from_file("app.py").run()

        # Navigate via sidebar
        at.sidebar.radio[0].set_value("New Assessment").run()
        
        assert at.title[0].value == "Initiate New Due Diligence Assessment"
        assert at.text_input[0].value == "TechInnovate Inc." # Default company_id_input
        assert at.selectbox[0].value == "full" # Default assessment_type_input
        assert at.text_input[1].value == "Sarah (Streamlit App)" # Default requested_by_input
        assert at.button[0].label == "Start Assessment"


    def test_start_assessment_success(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test starting a new assessment that completes successfully."""
        mock_run_due_diligence.return_value = mock_workflow_state_success
        at = AppTest.from_file("app.py").run()

        # Navigate and fill form (using defaults, then modifying company_id and assessment_type)
        at.sidebar.radio[0].set_value("New Assessment").run()
        at.text_input[0].set_value("TestCorp").run()
        at.selectbox[0].set_value("limited").run() # Change assessment type
        at.text_input[1].set_value("TestUser").run() # Change requested by

        # Click "Start Assessment"
        at.button[0].click().run()

        # Verify run_due_diligence was called with expected parameters
        mock_run_due_diligence.assert_called_once_with(
            company_id="TestCorp",
            assessment_type="limited",
            requested_by="TestUser",
            thread_id=at.session_state["latest_thread_id"] # Thread ID is dynamically generated
        )

        # Assertions for assessment_details page after success
        assert at.title[0].value == "Due Diligence Assessment Details"
        assert at.success[0].value == "Workflow completed successfully." # Specific message from app.py
        assert "Workflow Status for TechInnovate Inc." in at.markdown[at.markdown.index("### Workflow Status for TechInnovate Inc.")].value
        assert "**Workflow Completed Successfully!**" in at.success.values
        assert "SEC Analysis Findings" in at.markdown[at.markdown.index("### 📄 SEC Analysis Findings")].value
        assert "Talent Analysis Findings" in at.markdown[at.markdown.index("### 🧑‍💻 Talent Analysis Findings")].value
        assert "Org-AI-R Scoring Result" in at.markdown[at.markdown.index("### 📊 Org-AI-R Scoring Result")].value
        assert "Value Creation Plan" in at.markdown[at.markdown.index("### 💰 Value Creation Plan")].value
        assert "Final Org-AI-R Score: **78.5**" in at.markdown[at.markdown.index("- **Final Org-AI-R Score:** **78.5**")].value
        
        assert at.button[0].label == "Store Outcome in Mem0"
        assert at.button[1].label == "View Company History"
        assert at.markdown[at.markdown.index("### Agent Workflow Trace") + 2].value == mock_trace_mermaid_content
        mock_global_trace_manager.get_trace.assert_called_once_with("test_trace_success")


    def test_start_assessment_pending_hitl_and_approve(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test starting an assessment that requires HITL, then approving it."""
        mock_run_due_diligence.return_value = mock_workflow_state_pending
        mock_approve_workflow.return_value = mock_workflow_state_approved_after_hitl
        at = AppTest.from_file("app.py").run()

        # Navigate and start assessment
        at.sidebar.radio[0].set_value("New Assessment").run()
        at.text_input[0].set_value("RiskyBiz").run() # Set specific company ID
        at.button[0].click().run() # Start Assessment

        # Verify run_due_diligence was called
        mock_run_due_diligence.assert_called_once()
        # The company_id in session_state comes from the mocked return value
        assert at.session_state["latest_workflow_state"]["company_id"] == mock_workflow_state_pending["company_id"]

        # Assertions for assessment_details page with HITL pending
        assert at.warning[0].value == "Workflow paused for HITL approval!" # Specific message from app.py
        assert "**Workflow Paused for Human-in-the-Loop (HITL) Approval!**" in at.warning.values
        assert "Reason: Projected EBITDA impact exceeds threshold." in at.markdown[at.markdown.index("**Reason:** Projected EBITDA impact exceeds threshold.")].value
        assert at.text_input[0].value == "Lead Analyst" # Default hitl_approval_by
        assert at.radio[0].value == "approved" # Default hitl_decision
        assert at.text_area[0].value == "" # Default hitl_notes
        assert at.button[0].label == "Submit Approval"
        mock_global_trace_manager.get_trace.assert_called_once_with("test_trace_pending")


        # Simulate HITL approval
        at.text_input[0].set_value("TestApprover").run()
        at.radio[0].set_value("approved").run()
        at.text_area[0].set_value("Looks good, proceed.").run()
        at.button[0].click().run() # Click "Submit Approval"

        # Verify approve_workflow was called
        mock_approve_workflow.assert_called_once_with(
            thread_id=at.session_state["latest_thread_id"],
            approved_by="TestApprover",
            decision="approved",
            notes="Looks good, proceed."
        )

        # Assertions after approval
        assert at.success[0].value == "Workflow approved and resumed/completed!"
        assert at.success[1].value == "**Workflow Completed Successfully!**" # After successful approval
        assert not any("Human-in-the-Loop (HITL) Approval Required" in val for val in at.markdown.values) # Ensure HITL section is gone
        assert at.button[0].label == "Store Outcome in Mem0"
        assert at.button[1].label == "View Company History"
        mock_global_trace_manager.get_trace.assert_called_with("test_trace_pending") # Trace ID should be the same


    def test_submit_hitl_approval_rejected(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test submitting HITL approval with a 'rejected' decision."""
        mock_run_due_diligence.return_value = mock_workflow_state_pending
        mock_approve_workflow.return_value = mock_workflow_state_rejected_after_hitl
        at = AppTest.from_file("app.py").run()

        # Navigate and start assessment to get to HITL pending state
        at.sidebar.radio[0].set_value("New Assessment").run()
        at.text_input[0].set_value("RejectCorp").run()
        at.button[0].click().run()

        # Simulate HITL rejection
        at.radio[0].set_value("rejected").run()
        at.text_area[0].set_value("Too risky, terminate.").run()
        at.button[0].click().run()

        # Verify approve_workflow was called
        mock_approve_workflow.assert_called_once_with(
            thread_id=at.session_state["latest_thread_id"],
            approved_by="Lead Analyst", # Default value
            decision="rejected",
            notes="Too risky, terminate."
        )

        # Assertions after rejection
        assert at.error[0].value == "Workflow rejected and terminated."
        assert at.error[1].value == "**Workflow Ended with Error:** None" # The mock state doesn't have an 'error' key but the app displays "None"
        assert not any("Human-in-the-Loop (HITL) Approval Required" in val for val in at.markdown.values) # Ensure HITL section is gone


    def test_store_outcome(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test storing the assessment outcome in Mem0."""
        mock_run_due_diligence.return_value = mock_workflow_state_success
        at = AppTest.from_file("app.py").run()

        # Start and complete an assessment to enable the "Store Outcome" button
        at.sidebar.radio[0].set_value("New Assessment").run()
        at.button[0].click().run() # This uses the default mock_workflow_state_success
        
        # Verify the "Store Outcome in Mem0" button is present and click it
        assert at.button[0].label == "Store Outcome in Mem0"
        at.button[0].click().run()

        # Verify store_assessment_outcome was called with correct arguments
        mock_store_outcome.assert_called_once_with(
            company_id=mock_workflow_state_success["company_id"],
            assessment_type=mock_workflow_state_success["assessment_type"],
            final_score=mock_workflow_state_success["scoring_result"]["final_score"],
            key_findings=[
                "SEC Analysis: Strong presence in AI-related patents....",
                "Talent Analysis: 150 AI roles, 35.0% talent concentration.",
                "Value Creation Initiatives: AI Platform Modernization (Impact: High); Data Governance Program (Impact: Medium)."
            ],
            user_id=mock_workflow_state_success["requested_by"]
        )
        assert at.success[-1].value == f"Assessment outcome for '{mock_workflow_state_success['company_id']}' stored in Mem0!"


    def test_navigate_to_company_history_and_load(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test navigation to Company History and loading historical context."""
        mock_get_company_context.return_value = mock_company_memories
        at = AppTest.from_file("app.py").run()

        # Navigate via sidebar
        at.sidebar.radio[0].set_value("Company History").run()

        assert at.title[0].value == "Company Assessment History"
        assert at.text_input[0].value == "RiskyBet Corp." # Default company_history_id_input
        assert at.button[0].label == "Load Company History"

        # Click "Load Company History"
        at.button[0].click().run()

        # Verify get_company_context was called
        mock_get_company_context.assert_called_once_with(
            company_id="RiskyBet Corp.",
            user_id="Sarah (Streamlit App)"
        )

        # Assertions for displayed history
        assert at.success[0].value == "Company history loaded!"
        assert at.subheader[0].value == f"Memories for {mock_company_memories['company_id']}"
        assert "Memory 1 (ID: mem-1)" in at.markdown.values
        assert "Content: Assessment outcome for RiskyBet Corp. (full): Final Score 78.5." in at.markdown.values
        assert "Memory 2 (ID: mem-2)" in at.markdown.values
        assert "Content: Assessment outcome for RiskyBet Corp. (screening): Final Score 65.0." in at.markdown.values


    def test_company_history_no_memories(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test loading company history when no memories are found."""
        mock_get_company_context.return_value = {
            "company_id": "NoMemCorp", "user_id": "TestUser", "memory_count": 0, "memories": []
        }
        at = AppTest.from_file("app.py").run()

        # Navigate to Company History
        at.sidebar.radio[0].set_value("Company History").run()
        at.text_input[0].set_value("NoMemCorp").run()
        at.button[0].click().run()

        # Verify get_company_context was called
        mock_get_company_context.assert_called_once()
        assert at.success[0].value == "Company history loaded!"
        assert at.info[0].value == "No historical memories found for 'NoMemCorp'."


    def test_assessment_details_no_workflow_run(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test navigating to Assessment Details when no workflow has been run yet."""
        at = AppTest.from_file("app.py").run()

        # Ensure session state has no latest_workflow_state (should be default for a fresh AppTest instance)
        assert "latest_workflow_state" not in at.session_state or at.session_state["latest_workflow_state"] is None

        # Navigate via sidebar
        at.sidebar.radio[0].set_value("Assessment Details").run()

        assert at.title[0].value == "Due Diligence Assessment Details"
        assert at.info[0].value == "No assessment has been run yet. Please start a new assessment."
        assert at.button[0].label == "Go to New Assessment"

        # Click "Go to New Assessment" button
        at.button[0].click().run()
        assert at.session_state["current_page"] == "new_assessment"
        assert at.title[0].value == "Initiate New Due Diligence Assessment"


    def test_view_company_history_from_details_page(self,
        mock_get_company_context, mock_store_outcome, mock_approve_workflow, mock_run_due_diligence
    ):
        """Test viewing company history directly from a completed assessment's details page."""
        mock_run_due_diligence.return_value = mock_workflow_state_success
        mock_get_company_context.return_value = mock_company_memories
        at = AppTest.from_file("app.py").run()

        # Start and complete an assessment to get to the details page with buttons
        at.sidebar.radio[0].set_value("New Assessment").run()
        at.button[0].click().run() # Triggers start_assessment_callback

        # Verify we are on the details page and the "View Company History" button is present
        assert at.session_state["current_page"] == "assessment_details"
        assert at.button[1].label == "View Company History" # Index 1 for "View Company History"

        # Click "View Company History" button
        at.button[1].click().run()

        # Assertions for navigation and history loading
        assert at.session_state["current_page"] == "company_history"
        assert at.session_state["company_history_id_input"] == mock_workflow_state_success["company_id"]
        mock_get_company_context.assert_called_once_with(
            company_id=mock_workflow_state_success["company_id"],
            user_id=at.session_state["requested_by_input"]
        )
        assert at.success[0].value == "Company history loaded!"
        assert at.subheader[0].value == f"Memories for {mock_company_memories['company_id']}"
