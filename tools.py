# tools.py
import os
from langchain_google_community import GoogleSearchAPIWrapper # <--- Corrected Import
from langchain_core.tools import tool
from dotenv import load_dotenv # Ensure load_dotenv is imported
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import traceback
import streamlit as st # Import for potential access to cached resources if needed

# --- Tool Configuration ---
RAG_K = 5 # Number of chunks to retrieve for RAG context
ICD_TOP_N = 5 # Number of top ICD codes to consider matching
ICD_SIM_THRESHOLD = 0.25 # Similarity threshold for ICD matching

# --- Accessing Global Resources ---
# Load components needed by tools using the utility functions.
print("tools.py: Loading shared resources via utils...")
try:
    # Assuming utils.py correctly loads these using @st.cache_resource
    from utils import load_embedding_model, load_vector_store, load_icd_data_and_embeddings
    embedding_model_t = load_embedding_model()
    vector_store_t = load_vector_store(embedding_model_t)
    icd_codes_t, icd_embeddings_t = load_icd_data_and_embeddings(embedding_model_t)
    print("tools.py: Shared resources loaded.")
except ImportError:
    print("tools.py: Error importing from utils. Make sure utils.py is in the same directory.")
    embedding_model_t, vector_store_t, icd_codes_t, icd_embeddings_t = None, None, [], None
except Exception as e_load:
    print(f"tools.py: Error loading components via utils: {e_load}")
    embedding_model_t, vector_store_t, icd_codes_t, icd_embeddings_t = None, None, [], None

# --- Load Google CSE Credentials ---
load_dotenv() # Ensure .env is loaded
GOOGLE_API_KEY_CSE = os.getenv("GOOGLE_API_KEY") # Key enabled for Custom Search API
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- Tool Definitions ---

@tool
def retrieve_relevant_documents(user_symptoms: str) -> str:
    """
    Retrieves relevant text chunks from the medical knowledge base (PDF documents)
    based on the provided user symptoms or medical query. Use this tool to gather
    contextual information before answering questions or making recommendations.
    Returns a formatted string containing the source and content of the relevant chunks,
    separated by '=====', or an error message.
    NOTE: This tool currently only processes TEXT input ('user_symptoms').
    """
    tool_name = "retrieve_relevant_documents"
    print(f"\n--- Tool: {tool_name} ---")
    print(f"Input symptoms (first 100 chars): '{user_symptoms[:100]}...'")

    # Check if resources loaded correctly
    if not vector_store_t or not isinstance(vector_store_t, FAISS):
        error_msg = "Error: Vector store for document retrieval is not available or not loaded."
        print(f"{tool_name}: {error_msg}")
        return error_msg
    if not user_symptoms:
        error_msg = "Error: No symptoms or query provided for retrieval."
        print(f"{tool_name}: {error_msg}")
        return error_msg

    try:
        # Ensure the index is populated
        if not hasattr(vector_store_t, 'index') or vector_store_t.index.ntotal == 0:
             error_msg = "Error: Vector store index is empty or not initialized."
             print(f"{tool_name}: {error_msg}")
             return error_msg

        retriever = vector_store_t.as_retriever(search_type="similarity", search_kwargs={"k": RAG_K})
        # Use invoke for newer LangChain versions
        retrieved_docs = retriever.invoke(user_symptoms)

        if not retrieved_docs:
             result_msg = "No relevant documents found in the knowledge base for the given query."
             print(f"{tool_name}: {result_msg}")
             return result_msg

        # Include source metadata in the output string
        context_parts = []
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown Source')
            # Extract just the filename from the source path
            source_filename = os.path.basename(source) if source != 'Unknown Source' else source
            page_content = str(doc.page_content) if hasattr(doc, 'page_content') else ""
            context_parts.append(f"Source: {source_filename}\n---\n{page_content.strip()}")

        context_string = "\n=====\n".join(context_parts)
        result_msg = context_string if context_string else "No relevant text content found in retrieved documents."
        print(f"{tool_name}: Retrieved {len(retrieved_docs)} document chunks. Context length: {len(result_msg)}")

        return result_msg

    except Exception as e:
        error_msg = f"Error during RAG context retrieval tool execution: {e}"
        print(f"{tool_name}: {error_msg}")
        traceback.print_exc()
        return f"Error retrieving relevant medical information: {e}"

@tool
def match_relevant_icd_codes(user_symptoms: str) -> str:
    """
    Matches potentially relevant ICD-10 medical codes to the user's described text symptoms
    using semantic similarity search against a database of ICD code descriptions.
    Returns a string listing matched codes and their similarity scores, or an error message.
    NOTE: This tool currently only processes TEXT input ('user_symptoms').
    """
    tool_name = "match_relevant_icd_codes"
    print(f"\n--- Tool: {tool_name} ---")
    print(f"Input symptoms (first 100 chars): '{user_symptoms[:100]}...'")

    # Check if necessary components are loaded and valid
    if icd_embeddings_t is None or not isinstance(icd_embeddings_t, np.ndarray) or icd_embeddings_t.size == 0:
        error_msg = "Error: ICD code embeddings are not available or not loaded correctly."
        print(f"{tool_name}: {error_msg}")
        return error_msg
    if not icd_codes_t:
        error_msg = "Error: ICD code list is not available."
        print(f"{tool_name}: {error_msg}")
        return error_msg
    if not embedding_model_t:
        error_msg = "Error: Embedding model is not available for ICD matching."
        print(f"{tool_name}: {error_msg}")
        return error_msg
    if not user_symptoms:
        error_msg = "Error: No symptoms provided for ICD matching."
        print(f"{tool_name}: {error_msg}")
        return error_msg

    try:
        # Embed the text query
        query_emb_list = embedding_model_t.embed_query(user_symptoms)
        query_emb = np.array([query_emb_list]).astype('float32')

        # Perform cosine similarity search
        sims = cosine_similarity(icd_embeddings_t, query_emb).flatten()
        sorted_indices = np.argsort(sims)[::-1]

        # Collect matches above threshold
        matched = []
        for idx in sorted_indices[:ICD_TOP_N]: # Look at top N potential matches
            # Check index bounds and similarity threshold
            if idx < len(icd_codes_t) and sims[idx] >= ICD_SIM_THRESHOLD:
                 code = icd_codes_t[idx]
                 score = sims[idx]
                 matched.append(f"{code} (Similarity: {score:.2f})") # Include score

        if not matched:
            result_msg = "No relevant ICD codes found with sufficient similarity."
            print(f"{tool_name}: {result_msg}")
            return result_msg
        else:
            result_str = "; ".join(matched)
            print(f"{tool_name}: Matched ICD codes: {result_str}")
            return result_str

    except Exception as e:
        error_msg = f"Error matching ICD codes tool: {e}"
        print(f"{tool_name}: {error_msg}")
        traceback.print_exc()
        return f"Error during ICD code matching: {e}"


# --- NEW: Google Search Tool using Custom Search API ---
@tool
def google_search(query: str) -> str:
    """
    Performs a Google search using the Google Search API Wrapper (leveraging
    Custom Search Engine if GOOGLE_CSE_ID is set). Use for general queries
    when internal knowledge base fails. Returns search results snippets.
    """
    tool_name = "google_search" # Keep generic tool name for the graph
    print(f"\n--- Tool: {tool_name} ---")
    print(f"Input query: '{query[:100]}...'")

    # Check for credentials
    if not GOOGLE_API_KEY_CSE:
        error_msg = "Error: Google API key (GOOGLE_API_KEY) not found in environment variables."
        print(f"{tool_name}: {error_msg}")
        return error_msg
    if not GOOGLE_CSE_ID:
        # Decide: Fail or allow generic Google Search? Let's fail for CSE specific tool.
        # If you want generic search without CSE, remove this check and don't pass google_cse_id below.
        error_msg = "Error: Google Custom Search Engine ID (GOOGLE_CSE_ID) not found for specific search."
        print(f"{tool_name}: {error_msg}")
        return error_msg

    try:
        # Instantiate the wrapper - it uses CSE ID if provided
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY_CSE,
            google_cse_id=GOOGLE_CSE_ID,
            k=3 # Limit results
        )
        # Run the search
        results = search_wrapper.run(query) # Returns a formatted string
        print(f"{tool_name}: Search successful. Result length: {len(results)}")
        # Add prefix
        return f"Web search results (Google CSE):\n---\n{results}\n---" if results else "No relevant results found via web search."

    except Exception as e:
        error_msg = f"Error during Google Search tool execution: {e}"
        print(f"{tool_name}: {error_msg}")
        traceback.print_exc()
        # Add specific error checks if needed based on API responses
        return f"Error performing web search: {e}"