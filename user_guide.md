id: 696e76aaaf3b3d8cb59d82fa_user_guide
summary: Lab 10: LangGraph Multi-Agent Orchestration User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: AI-Powered Due Diligence Multi-Agent Orchestration
## Introduction to AI-Powered Due Diligence
Duration: 0:05:00

Welcome to QuLab: Lab 10, an exploration into **LangGraph Multi-Agent Orchestration** for AI-powered Due Diligence! In this codelab, we'll guide you through a Streamlit application designed to revolutionize how Private Equity (PE) firms assess potential target companies.

<aside class="positive">
This application demonstrates a practical use case for Artificial Intelligence in financial analysis, specifically in the context of Private Equity due diligence. It aims to significantly accelerate and enhance the accuracy of initial company assessments.
</aside>

### The Challenge for Private Equity Analysts

Imagine Sarah, a software developer at "Synergy Capital," a forward-thinking PE firm. Her team faces a common problem: PE analysts spend countless hours manually reviewing financial documents, talent reports, and market data for initial due diligence. This manual process is not only time-consuming but also prone to inconsistencies, causing delays in crucial investment decisions.

### The AI-Powered Solution

This application presents a proof-of-concept for an **AI-powered multi-agent system**. Its goal is to automate the initial assessment of a target company's "AI-readiness." By using **LangGraph**, the system orchestrates specialized AI agents—like an SEC Analysis Agent, a Talent Analysis Agent, a Scoring Agent, and a Value Creation Agent—to produce a structured, consolidated report. This allows human analysts to quickly grasp a company's profile and focus their expertise on high-value strategic considerations, significantly accelerating the deal pipeline.

### Key Concepts You'll Explore

This application beautifully illustrates several advanced AI concepts:

*   **Supervisor Pattern for Agent Coordination:** Learn how a central "supervisor" agent directs the workflow between different specialized agents, ensuring a coherent and efficient assessment process.
*   **Specialist Agents:** Understand the roles of individual AI agents, each designed for a specific task (e.g., analyzing financial filings, evaluating talent, calculating scores, or proposing value creation strategies).
*   **Human-in-the-Loop (HITL) Approval Gates:** Discover how critical decision points in the AI workflow can trigger a pause, requiring human review and approval before proceeding, ensuring oversight and compliance.
*   **Semantic Memory with Mem0:** Explore how the system uses "Mem0" to store and retrieve past assessment outcomes and contextual information, providing a memory for the AI agents.
*   **Agent Debug Traces:** See how the application visualizes the execution path of the multi-agent system, which is invaluable for understanding and troubleshooting complex AI workflows.

By the end of this codelab, you'll have a clear understanding of how these concepts come together to create a powerful, intelligent due diligence system.

To begin, navigate using the sidebar on the left. You can either start a "New Assessment" or explore "Company History."

## Starting a New Assessment
Duration: 0:03:00

In this step, you will initiate a new due diligence assessment for a hypothetical target company. This is where the multi-agent system springs into action.

1.  **Navigate to "New Assessment"**:
    On the left sidebar, select "New Assessment" from the navigation options.

2.  **Input Company Details**:
    You'll see a form with the following fields:
    *   **Company ID**: Enter a unique identifier for the company you wish to assess (e.g., "TechInnovate Inc."). This field is pre-filled with an example for your convenience.
    *   **Assessment Type**: Choose the type of assessment. Options typically include "screening," "limited," or "full." This dictates the depth and breadth of the analysis performed by the agents. For this codelab, we'll stick with the default "full" assessment to see all functionalities.
    *   **Requested By**: Enter your name or an identifier (e.g., "Sarah (Streamlit App)"). This helps track who initiated the assessment and is also used for memory management.

3.  **Start the Assessment**:
    Once you've entered the details, click the **"Start Assessment"** button.

<aside class="positive">
Upon clicking "Start Assessment," the application will kick off the multi-agent workflow. You will see a spinner indicating that the agents are working. The application will then automatically transition you to the "Assessment Details" page, where you can monitor the progress and view the results.
</aside>

## Understanding Assessment Details and Workflow
Duration: 0:10:00

After starting an assessment, you'll be directed to the "Assessment Details" page. This is the core of the application, where you can observe the multi-agent system in action, review its findings, and interact with the Human-in-the-Loop (HITL) approval gates.

1.  **Workflow Status**:
    At the top of the "Assessment Details" page, you'll see a status banner for your assessment.
    *   If the workflow encounters a specific condition (e.g., a low Org-AI-R score or a high projected EBITDA impact), it might pause for **"Human-in-the-Loop (HITL) Approval!"** This means the system requires a human decision before it can proceed.
    *   If an error occurs, it will display an **"Workflow Ended with Error"** message.
    *   Ideally, you'll see **"Workflow Completed Successfully!"** indicating that all agents have finished their tasks.

2.  **Specialist Agent Findings**:
    The page will present the findings from various specialist agents, each focusing on a different aspect of due diligence:

    *   ### 📄 SEC Analysis Findings
        This section provides an overview of the company's regulatory and financial health, as if gleaned from SEC filings.
        *   **Purpose**: The SEC Analysis agent simulates reviewing public financial documents to identify key information, risks, and opportunities related to the company's AI initiatives or general operational landscape.
        *   You'll see a summary of **Findings**, **Evidence Count**, **Dimensions Covered**, and the agent's **Confidence** in its analysis.

    *   ### 🧑‍💻 Talent Analysis Findings
        This section delves into the company's human capital, particularly focusing on AI talent.
        *   **Purpose**: The Talent Analysis agent evaluates the company's workforce composition, identifying the number of AI-related roles, the concentration of talent in AI areas, seniority, key skills, and hiring trends. This helps in understanding the company's internal capabilities and future potential in AI.
        *   Look for metrics like **AI Role Count**, **Talent Concentration**, **Seniority Index**, **Key Skills**, and **Hiring Trend**.

    *   ### 📊 Org-AI-R Scoring Result
        This is a crucial output, providing a consolidated score for the company's AI readiness.
        *   **Purpose**: The Scoring agent synthesizes the information from the SEC and Talent analyses (and potentially other sources) to generate a quantitative "Org-AI-R Score." This score provides a quick snapshot of the company's AI capabilities and potential.
        *   You'll see the **Final Org-AI-R Score** and additional **Details**.
        *   **Human-in-the-Loop (HITL) Trigger**: The application shows a mathematical expression that represents a common trigger for HITL:
            $$ (S < S_{\text{min}}) \lor (S > S_{\text{max}}) $$
            Here, $S$ is the Org-AI-R score, $S_{\text{min}}$ is a minimum threshold (e.g., 40), and $S_{\text{max}}$ is a maximum threshold (e.g., 85). If the score falls outside this desired range, it might trigger a human review.

    *   ### 💰 Value Creation Plan
        If the assessment indicates an opportunity for improvement, a value creation plan is generated.
        *   **Purpose**: The Value Creation agent proposes strategic initiatives to enhance the company's AI readiness and overall value. This typically includes actionable steps with estimated impact and cost.
        *   You'll see the **Current Org-AI-R Score**, a **Target Org-AI-R Score**, **Projected EBITDA Impact**, and a **Timeline**.
        *   Key initiatives will be listed, each with a **Name**, **Impact**, and estimated **Cost**.
        *   **Another HITL Trigger**: The application also shows a mathematical expression for an EBITDA impact trigger for HITL:
            $$ I > \text{settings.HITL\_EBITDA\_PROJECTION\_THRESHOLD} $$
            Here, $I$ is the projected EBITDA impact (e.g., 0.07 for 7%). If the projected impact is significantly high, it may warrant human review to validate the ambitious plans.

3.  **Human-in-the-Loop (HITL) Approval Section**:
    If the workflow is paused for HITL approval, this section will become active.
    *   **Reason for HITL**: The system will explicitly state why human intervention is required (e.g., "Org-AI-R score is below minimum threshold," "Projected EBITDA impact is very high").
    *   **Decision**: You'll be prompted to enter who is "Approved By," select a "Decision" (either "approved" or "rejected"), and provide "Notes" for your decision.
    *   Click **"Submit Approval"** to send your decision back to the multi-agent system, allowing it to either resume or terminate the workflow.

4.  **Agent Workflow Trace**:
    This section is an invaluable tool for understanding the flow of the multi-agent system.
    *   **Purpose**: The diagram visualizes the actual execution path taken by the AI agents, including the sequence of their actions, decision points, and any pauses for HITL. It uses a `Mermaid` diagram format for clear representation.
    *   This trace helps developers and analysts debug, audit, and understand the logic and interactions within the complex multi-agent system.

5.  **Action Buttons**:
    Once the workflow is successfully completed (and if it wasn't rejected during HITL), you'll see two action buttons:
    *   **"Store Outcome in Mem0"**: This button allows you to save the assessment's final outcomes and key findings into Mem0, the application's semantic memory. This is crucial for building a historical context for each company.
    *   **"View Company History"**: This button navigates you directly to the "Company History" page, pre-populating it with the current company's ID, so you can immediately see its past assessments.

<aside class="negative">
Remember that the HITL triggers are vital for ensuring that fully automated systems don't make critical decisions without human oversight, especially in high-stakes environments like private equity.
</aside>

## Managing Historical Data with Mem0
Duration: 0:03:00

The "Company History" page allows you to retrieve and review past assessment outcomes and contextual information for any company from the application's semantic memory, powered by Mem0.

1.  **Navigate to "Company History"**:
    On the left sidebar, select "Company History." If you came from the "Assessment Details" page by clicking "View Company History," the "Company ID for History" field will already be populated.

2.  **Load Company History**:
    *   **Company ID for History**: Enter the ID of the company whose history you want to retrieve. An example is pre-filled.
    *   Click the **"Load Company History"** button.

3.  **Review Memories**:
    If memories exist for the specified company, they will be displayed.
    *   Each memory represents a past assessment outcome or a piece of contextual information stored about that company.
    *   You'll see details like the **Content** (a summary of findings or the assessment's purpose) and associated **Metadata** (e.g., assessment type, final score, requested by, date).

<aside class="positive">
Using Mem0 for semantic memory is powerful because it allows the AI system to learn from past interactions and assessments. This provides context, prevents redundant work, and potentially improves the accuracy and relevance of future analyses for the same company or similar entities. It’s like giving the AI a memory of its past experiences.
</aside>

This concludes your codelab on the AI-Powered Due Diligence Multi-Agent Orchestration application. You've learned how to initiate assessments, understand the outputs of various specialist agents, interact with human-in-the-loop decision points, visualize workflow traces, and leverage semantic memory for historical context. Congratulations!
