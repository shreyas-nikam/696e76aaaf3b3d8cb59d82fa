Here's a comprehensive `README.md` for your Streamlit application lab project, designed to be professional and informative.

---

# QuLab: Lab 10: LangGraph Multi-Agent Orchestration for AI-Powered Due Diligence

🚀 **AI-Powered Due Diligence for Private Equity with LangGraph & Streamlit**

This project demonstrates an advanced multi-agent system built with LangGraph, orchestrated to automate the initial due diligence process for Private Equity (PE) firms. It integrates specialized AI agents, Human-in-the-Loop (HITL) approvals, semantic memory using Mem0, and real-time workflow visualization within an intuitive Streamlit application.

## 🌟 Project Description

This lab project, "QuLab: Lab 10: LangGraph Multi-Agent Orchestration," focuses on empowering PE analysts at a fictional firm, "Synergy Capital," by modernizing their initial due diligence process. Traditionally, analysts spend significant time manually sifting through various data sources (financial filings, talent reports, market data) to assess potential target companies. This manual process is time-consuming, prone to inconsistencies, and often delays critical investment decisions.

The application serves as a proof-of-concept for an AI-powered multi-agent system that automates the initial assessment of a target company's AI-readiness. Using **LangGraph** for orchestration, specialized AI agents (e.g., SEC Analysis, Talent Analysis, Scoring, Value Creation) collaborate to produce a structured, consolidated report. This allows human analysts to quickly grasp a company's profile, focus their expertise on high-value strategic considerations, and significantly accelerate the deal pipeline.

This project is a comprehensive educational exercise demonstrating best practices in agentic AI development, including agent coordination, human interaction, persistent memory, and debugging tools.

## ✨ Features

*   **Multi-Agent Orchestration:** Leverages LangGraph to build a robust and directed workflow for agent interactions, managed by a central supervisor agent.
*   **Specialized AI Agents:**
    *   📄 **SEC Analysis Agent:** Extracts key findings and evidence from public filings.
    *   🧑‍💻 **Talent Analysis Agent:** Assesses AI talent concentration, skills, and hiring trends.
    *   📊 **Org-AI-R Scoring Agent:** Calculates a comprehensive AI-readiness score based on various inputs.
    *   💰 **Value Creation Agent:** Proposes strategic initiatives and estimates potential EBITDA impact.
*   **Human-in-the-Loop (HITL) Approval:** Implements a mechanism where the workflow pauses for human review and approval based on predefined conditions (e.g., extreme scores, high value creation projections).
*   **Semantic Memory with Mem0:** Integrates Mem0 to store and retrieve historical assessment outcomes and company context, enabling agents to leverage past insights and providing a richer analytical history.
*   **Real-time Workflow Visualization:** Generates dynamic Mermaid diagrams to visualize the agent execution trace, including decision points and HITL pauses, aiding in debugging and understanding workflow flow.
*   **Asynchronous Operations:** Utilizes `asyncio` with `nest_asyncio` for seamless execution of asynchronous LangGraph workflows within the synchronous Streamlit environment.
*   **Intuitive Streamlit UI:** Provides an easy-to-use interface for initiating assessments, viewing results, managing HITL approvals, and exploring company history.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   **Python 3.8+**: Ensure you have a compatible Python version installed.
*   **Git**: For cloning the repository.
*   **API Keys**: Access to OpenAI (or another compatible LLM provider) and Mem0.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name.git` with the actual repository URL)*

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    langchain
    langgraph
    mem0
    nest_asyncio
    openai
    # Add any other libraries your source.py might require (e.g., tiktoken, anthropic, etc.)
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application requires API keys for Large Language Models (LLMs) and Mem0. It's recommended to set these as environment variables.

1.  **Create a `.env` file** in the root directory of your project (or set them directly in your shell):
    ```env
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
    MEM0_API_KEY="YOUR_MEM0_API_KEY_HERE"
    # You might also need specific base URLs or model names if not using default OpenAI
    # OPENAI_BASE_URL="https://..."
    # OPENAI_MODEL_NAME="gpt-4o"
    ```
    *Make sure to replace the placeholder values with your actual API keys.*

2.  **`source.py` Note:**
    The core LangGraph multi-agent logic, agent definitions, and Mem0 integration are expected to be defined in a `source.py` file. This `app.py` imports everything from it (`from source import *`). **Crucially, `source.py` must not contain top-level `await` calls that would cause a `SyntaxError` upon import.** Any asynchronous function calls within `source.py` should be handled correctly, typically by calling them from an `async def` function within `source.py` and then using `run_async_function` from `app.py` to execute them.

## 🏃‍♀️ Usage

1.  **Run the Streamlit Application:**
    Ensure your virtual environment is active and navigate to the project root in your terminal.
    ```bash
    streamlit run app.py
    ```

2.  **Access the Application:**
    Your web browser will automatically open to the Streamlit application (usually `http://localhost:8501`).

3.  **Navigate and Interact:**
    *   **Home Page:** Provides an introduction to the project's goals, persona, and key concepts.
    *   **New Assessment:** Input the Company ID, Assessment Type (screening, limited, full), and Requester Name to initiate a new due diligence workflow.
    *   **Assessment Details:** View the progress and detailed findings of the latest assessment. This page will display:
        *   SEC Analysis findings.
        *   Talent Analysis findings.
        *   Org-AI-R Scoring results.
        *   Value Creation Plan.
        *   **Human-in-the-Loop (HITL) Section:** If the workflow pauses for approval, this section will appear, allowing you to review the state, make a decision (approve/reject), and add notes before resuming.
        *   **Agent Workflow Trace:** A dynamic Mermaid diagram visualizes the flow of agents and decisions made throughout the workflow.
    *   **Company History:** Enter a Company ID to retrieve and display all past assessment outcomes and contextual memories stored in Mem0 for that company.

## 📂 Project Structure

```
.
├── .venv/                      # Python virtual environment
├── app.py                      # Main Streamlit application
├── source.py                   # (Expected) Contains LangGraph workflow, agent definitions, Mem0 logic, etc.
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys, etc.)
├── README.md                   # This file
└── quantuniversity_logo.jpg     # (Optional) The logo image used in the sidebar
```

*   **`app.py`**: This is the frontend of the application, handling user input, displaying results, managing Streamlit session state, and orchestrating the calls to the backend LangGraph workflow defined in `source.py`. It also integrates `nest_asyncio` to run async functions.
*   **`source.py`**: This file (which must be created/provided alongside `app.py`) is where the core logic of the multi-agent system resides. It should contain:
    *   Definitions of individual AI agents (SEC, Talent, Scoring, Value Creation).
    *   The LangGraph state definition and graph construction.
    *   The supervisor agent logic.
    *   Functions for interacting with Mem0 (e.g., `store_assessment_outcome`, `get_company_context`).
    *   The `run_due_diligence` and `approve_workflow` functions called by `app.py`.
    *   The `global_trace_manager` for workflow visualization.

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Backend Logic/Orchestration**:
    *   [Python 3.8+](https://www.python.org/)
    *   [LangChain](https://www.langchain.com/)
    *   [LangGraph](https://langchain-ai.github.io/langgraph/)
    *   [nest_asyncio](https://pypi.org/project/nest-asyncio/) (for Streamlit-Asyncio compatibility)
*   **Large Language Models (LLMs)**: [OpenAI GPT models](https://openai.com/) (configurable for other providers via LangChain)
*   **Semantic Memory**: [Mem0](https://www.mem0.ai/)
*   **Workflow Visualization**: [Mermaid.js](https://mermaid.js.org/) (rendered by Streamlit)
*   **Version Control**: [Git](https://git-scm.com/)

## 🤝 Contributing

This project is primarily designed as a lab exercise. However, if you have suggestions for improvements or encounter issues, feel free to:

1.  **Fork** the repository.
2.  **Create a new branch** (`git checkout -b feature/amazing-feature`).
3.  **Commit your changes** (`git commit -m 'Add some amazing feature'`).
4.  **Push to the branch** (`git push origin feature/amazing-feature`).
5.  **Open a Pull Request**.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(You'll need to create a `LICENSE` file in your repository if you don't have one)*

## 📧 Contact

For questions or feedback, please reach out to:

*   **Quant University**
*   **Website:** [https://www.quantuniversity.com/](https://www.quantuniversity.com/)

---

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
