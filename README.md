# MedMatch(Agentic AI Healthcare Assistant) - Project

## Overview

This project implements an advanced AI Healthcare Assistant based on the "Creative Document Processor" assessment requirements. It leverages a multi-step, agentic workflow orchestrated using **LangGraph** to process user queries—including text and optional image uploads—related to medical symptoms and general health information.

The system ingests a knowledge base of medical documents (PDFs, including the Gale Encyclopedia of Medicine) and structured data (CSVs for doctor profiles, patient cases, ICD-10 codes). It utilizes **Google Gemini 2.0 Flash** via the `langchain-google-genai` package for its core language understanding and generation, including multimodal capabilities. External information is accessed via the **Google Custom Search API**.

The primary "creative output" includes context-aware specialist recommendations with refined, jargon-free explanations, answers to general medical questions derived from RAG or web search, and potentially relevant doctor profiles. The user interacts via a **Streamlit** web interface (fulfilling the basic frontend requirement).

This implementation focuses on **Complex Agentic Workflow**, incorporating patterns like Workflow Routing, Tool Use, and Evaluator-Optimizer as described in Anthropic's "Building Effective Agents" guide. It reuses significant elements from a prior MedMatch project.

## Features

*   **Multimodal Input:** Accepts user text queries optionally accompanied by uploaded images.
*   **Intent Recognition:** Classifies user input (`SYMPTOM_TRIAGE`, `MEDICAL_INFORMATION_REQUEST`, `OFF_TOPIC`).
*   **Agentic Workflow Routing (LangGraph):** Dynamically directs processing based on intent.
*   **Contextual Follow-up & Sufficiency Check:** Asks relevant follow-up questions during triage and uses an LLM to determine when enough detail is gathered (replacing hardcoded turn limits).
*   **Triage Relevance Check:** Verifies if the accumulated symptom conversation is medically relevant before detailed analysis.
*   **Retrieval-Augmented Generation (RAG):** Retrieves relevant information from indexed PDF documents (FAISS vector store) via a tool.
*   **RAG Relevance Evaluation:** Uses an LLM to evaluate if RAG context is sufficient/relevant to the query *before* using it or falling back to search.
*   **External Web Search Fallback:** Uses Google Custom Search API tool if internal RAG is insufficient for information requests.
*   **Semantic ICD-10 Matching:** Provides potentially relevant medical codes via a tool (informational only).
*   **Multimodal Specialist Recommendation:** Considers text, image (if provided), RAG context, and ICD codes.
*   **Explanation Refinement (Evaluator-Optimizer Loop):** LLM evaluates and potentially refines the specialist explanation for clarity.
*   **Data-Driven Doctor Recommendation:** Recommends/ranks doctors based on specialty and symptom similarity to past cases.
*   **Transparency:** Option to view RAG context and ICD codes.
*   **User Interface:** Interactive chat built with Streamlit.

## Architecture

The application uses a layered architecture:

1.  **Presentation Layer (UI):** Streamlit (`app_agentic.py`)
2.  **Agent Orchestration Layer (Control Flow):** LangGraph (`graph_builder.py`, `state.py`)
3.  **Core Logic & Tools Layer (Execution Units):** LLMChains (`graph_builder.py`), Tools (`tools.py` - RAG, ICD, Search), Custom Python Functions (`recommend_doctors`, `extract_specialist` in `graph_builder.py`)
4.  **Model Layer (`utils.py`):** Google Gemini 1.5 Flash LLM, Sentence Transformer Embedding Model.
5.  **Data Layer (`utils.py`, `data/`, `vectorstore/`, CSVs, `.pkl`):** PDFs, FAISS index, CSVs, ICD embeddings cache.



## Agentic Workflow Details

This agent uses LangGraph to manage a complex, stateful workflow:

1.  **Intent Classification:** Classifies the user's query (`classify_intent_node`).
2.  **Routing:** Routes based on intent (`SYMPTOM_TRIAGE`, `MEDICAL_INFORMATION_REQUEST`, `OFF_TOPIC`).
3.  **Information Request Path:** Tries RAG, evaluates relevance via LLM, falls back to Google Search if needed, synthesizes answer.
4.  **Symptom Triage Path:**
    *   Gathers symptoms over potentially multiple turns, asking follow-ups based on LLM sufficiency check.
    *   Checks conversation relevance via LLM.
    *   Performs final analysis: RAG, ICD matching (on text), multimodal LLM call for specialist recommendation (using text + image).
    *   Enters Evaluator-Optimizer Loop for explanation clarity.
    *   Extracts specialist name, runs Python-based doctor recommendation using text symptoms.
    *   Formats final response.
5.  **End State:** Graph returns final state to Streamlit for display.

**Agentic Patterns Used:**

*   **Workflow Routing / Prompt Classification**
*   **Tool Use** (RAG, ICD Match, Google Search)
*   **Evaluator-Optimizer** (Explanation Refinement)
*   **(Implicit) Reasoning** (Sufficiency Check, Relevance Check, Final Synthesis)

## Setup Instructions

1.  **Prerequisites:**
    *   Python 3.9+
    *   Git
    *   Git LFS (install from [https://git-lfs.com](https://git-lfs.com) and run `git lfs install` once)
    *   Google Cloud Account (for API Key and Custom Search Engine setup)

2.  **Clone Repository & Pull LFS Files:**
    ```bash
    git clone https://github.com/AayushKhandelwal67/MedMatch-AgenticAIForHealthcare.git
    cd MedMatch-AgenticAIForHealthcare
    
    ```

3.  **Create Environment (Recommended):**
    ```bash
    python -m venv medmatchvenv
    source medmatchvenv/bin/activate  # Linux/macOS
    # OR: .\medmatchvenv\Scripts\activate # Windows
    ```

4.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5.  **Set Up API Keys & IDs:**
    *   Enable "Custom Search API" and "Generative Language API" in your Google Cloud Project.
    *   Create a restricted API Key enabled for both APIs.
    *   Create a Google Programmable Search Engine (search entire web) and get its ID (cx).
    *   Create `.env` file in the project root:
        ```.env
        GOOGLE_API_KEY=<YOUR_GOOGLE_CLOUD_API_KEY_HERE>
        GOOGLE_CSE_ID=<YOUR_PROGRAMMABLE_SEARCH_ENGINE_ID_HERE>
        ```
    *   **DO NOT COMMIT `.env` FILE.**

6.  **Data Files & Caches:**
    *   Necessary CSV/TXT files are included in the repo.
    *   The `data/` folder contains PDFs (including the large encyclopedia via LFS).
    *   The `vectorstore/` and `icd_embeddings.pkl` are included via LFS and downloaded by `git lfs pull`. **No long initial build time is required if LFS pull is successful.**

## Running the Application

1.  **Activate Environment:**
    ```bash
    source medmatchvenv/bin/activate # Or .\medmatchvenv\Scripts\activate
    ```
2.  **Run Streamlit App:**
    ```bash
    streamlit run app_agentic.py
    ```
3.  Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

## Usage Examples / Testing Scenarios

*   **Symptom Triage:** "I have a rash on my elbow", answer follow-ups.
*   **Symptom Triage (Image):** Upload rash image, text: "what is this?".
*   **Medical Information (RAG):** "Explain osteoarthritis."
*   **Medical Information (Web Search):** "Latest CDC guidelines for flu vaccination?"
*   **Off-Topic:** "Capital of Australia?"

Monitor the console output for detailed agent execution logs. Check the UI for responses and the "Show Analysis Details" expander.

## How Requirements Met

*   **Req 1 (Ingestion/Storage):** Ingests PDFs/CSVs, stores embeddings/cache locally (FAISS/Pickle), uses Git LFS for large file distribution.
*   **Req 2 (LLM Pipeline):** Implements complex (>2 steps) LLM flow using LangGraph, incorporating **Routing, Tool Use, and Evaluator-Optimizer** patterns.
*   **Req 3 (Frontend):** Uses Streamlit for a simple, interactive chat interface displaying inputs and processed outputs.
*   **Optional Focus (#4):** Directly addressed via the complex LangGraph implementation with multiple LLM calls, routing, and evaluation loops.

## Testing Approach

Manual testing was performed covering the different agent pathways (intent routing, RAG success/fallback, triage success/relevance failure, multimodal input, evaluator loop triggering) by observing UI outputs and detailed console logs tracking node execution and state changes. (Per assessment Tip #6/7).

**(Stretch Goal):** Future work would involve automated tests (`pytest`) and potentially LLM-as-judge evaluations.

## Decisions and Challenges

### Decisions Made & Rationale

1.  **Framework Choice (LangGraph):** Chosen over simpler LangChain chains to robustly handle the required conditional logic (intent routing, relevance/sufficiency checks, evaluator loops) and manage state effectively in a multi-step agentic process. 
2.  **LLM Choice (Google Gemini 2.0 Flash):** Selected based on the assessment recommendation for a free, capable model with multimodal support, accessed via `langchain-google-genai`. 
3.  **Agentic Patterns Implementation:** Directly implemented **Workflow Routing**, **Tool Use**, and **Evaluator-Optimizer** to meet assessment requirements and demonstrate understanding of the Anthropic guide concepts.
4.  **Dynamic Conversation Flow:** Replaced hardcoded turn limits with an **LLM-based Sufficiency Check** for more natural symptom gathering.
5.  **Intelligent Search Fallback:** Added an **LLM-based RAG Relevance Evaluation** to make the decision between using internal RAG context or external web search more robust for information requests.
6.  **Multimodal Input:** Incorporated image uploads to leverage Gemini's capabilities and handle visual symptoms, enhancing scope and technical demonstration.
7.  **Data Sources:** Used a combination of provided PDFs, a large external PDF (Gale Encyclopedia), and synthetic structured data (CSVs) to create a diverse knowledge base.
8.  **Storage & Distribution (Local + Git LFS):** Used local FAISS/Pickle for storage (meeting core requirement) and **Git LFS** to manage/distribute large cache/data files, balancing ease of setup for end-users with repository size constraints.
9.  **Frontend (Streamlit):** Utilized for rapid UI development, meeting the "simple frontend" requirement.

### Challenges Encountered & Solutions

1.  **Large File Management:** The encyclopedia PDF, FAISS index, and ICD embeddings exceeded GitHub's 100MB limit. *Solution:* Configured and used **Git LFS** and updated cloning instructions to include `git lfs pull`. Documented this requirement clearly.
2.  **Agentic Logic Debugging:** Tracing state and decisions through LangGraph was complex. *Solution:* Added extensive `print` logging in nodes/tools; used Mermaid diagrams for visualization; leveraged console output heavily during testing.
3.  **Prompt Engineering:** Iteratively refined prompts for evaluators, classifiers, and checks to achieve reliable outputs. *Solution:* Focused on clear instructions, tested edge cases.
4.  **Multimodal Integration:** Correctly formatting input for Gemini and managing image state within the graph required careful implementation. *Solution:* Used `HumanMessage` list structure; updated relevant node logic and prompts.
5.  **Dependency Management:** Resolved `protobuf`, `altair`, and LangChain package import errors. *Solution:* Pinned specific compatible versions (`protobuf==3.20.3`, `altair==4.2.2`), installed required integration packages (`langchain-google-community`), and updated imports.
6.  **API Setup & Keys:** Correctly configuring Google Cloud APIs, CSE, and managing keys in `.env` was crucial. *Solution:* Followed documentation, restricted API keys, ensured `.env` loading.
7.  **Initial Load Time vs. Caching:** Balancing the long *initial* data processing time with fast subsequent loads. *Solution:* Relied on effective caching (`@st.cache_resource`, FAISS load/save, Pickle) and documented the first-run expectation.
