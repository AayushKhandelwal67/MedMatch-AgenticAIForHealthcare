# Agentic AI Healthcare Assistant (MedMatch Enhanced)

## Overview

This project implements an advanced AI Healthcare Assistant, developed as part of the "Creative Document Processor" assessment. It leverages a multi-step, agentic workflow orchestrated using **LangGraph** to process user queries (text and optional images) related to medical symptoms and general health information.

The system ingests a knowledge base of medical documents (PDFs, including the Gale Encyclopedia of Medicine) and structured data (CSVs for doctor profiles, patient cases, ICD-10 codes). It utilizes **Google Gemini 1.5 Flash** via the `langchain-google-genai` package for its core language understanding and generation, including multimodal capabilities. External information is accessed via the **Google Custom Search API**.

The primary "creative output" includes context-aware specialist recommendations with refined, jargon-free explanations, answers to general medical questions derived from RAG or web search, and potentially relevant doctor profiles. The user interacts via a **Streamlit** web interface.

This implementation focuses on **Optional Area #4: Complex Agentic Workflow**, incorporating patterns like Workflow Routing, Tool Use, and Evaluator-Optimizer as described in Anthropic's "Building Effective Agents" guide.

## Features

*   **Multimodal Input:** Accepts user text queries optionally accompanied by uploaded images (e.g., of a rash or injury).
*   **Intent Recognition:** Classifies user input into `SYMPTOM_TRIAGE`, `MEDICAL_INFORMATION_REQUEST`, or `OFF_TOPIC`.
*   **Agentic Workflow Routing (LangGraph):** Dynamically directs the processing flow based on intent.
*   **Contextual Follow-up Questions:** Asks relevant questions during symptom triage.
*   **LLM-Powered Sufficiency Check:** Determines if enough symptom detail has been gathered before analysis (replaces hardcoded turn limits).
*   **Triage Relevance Check:** Verifies if the accumulated symptom conversation is medically relevant.
*   **Retrieval-Augmented Generation (RAG):** Retrieves relevant information from indexed PDF documents (FAISS vector store) via a tool.
*   **RAG Relevance Evaluation:** Uses an LLM to evaluate if RAG context is sufficient/relevant to the query before using it.
*   **External Web Search Fallback:** Uses Google Custom Search API tool if internal RAG is insufficient for information requests.
*   **Semantic ICD-10 Matching:** Identifies potentially relevant ICD-10 codes via a tool (informational only).
*   **LLM-Powered Specialist Recommendation:** Synthesizes symptoms, image (if provided), RAG context, and ICD codes.
*   **Explanation Refinement (Evaluator-Optimizer Loop):** LLM evaluates and potentially refines the specialist explanation for clarity.
*   **Data-Driven Doctor Recommendation:** Recommends/ranks doctors based on specialty and symptom similarity to past cases.
*   **Transparency:** Option to view RAG context and ICD codes used.
*   **User Interface:** Interactive chat built with Streamlit.

## Architecture

The application uses a layered architecture:

1.  **Presentation Layer (UI):** Streamlit (`app_agentic.py`)
2.  **Agent Orchestration Layer (Control Flow):** LangGraph (`graph_builder.py`, `state.py`)
3.  **Core Logic & Tools Layer (Execution Units):** LLMChains (`graph_builder.py`), Tools (`tools.py` - RAG, ICD, Search), Custom Python Functions (`recommend_doctors`, `extract_specialist` in `graph_builder.py`)
4.  **Model Layer (`utils.py`):** Google Gemini 1.5 Flash LLM, Sentence Transformer Embedding Model.
5.  **Data Layer (`utils.py`, `data/`, `vectorstore/`, CSVs, `.pkl`):** PDFs, FAISS index, CSVs, ICD embeddings cache.

