# graph_builder.py
import operator
from typing import Annotated, Sequence, List, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import re
import traceback
import base64 # Needed for image encoding

# Import state, prompts, tools, utils
from state import AgentState
import prompts
import tools
from utils import load_llm, load_specialist_list, load_dataframes, load_embedding_model # Load needed components

# --- Load necessary components/data (Leverages caching from utils) ---
print("graph_builder.py: Loading shared resources via utils...")
llm = load_llm()
specialist_list_g = load_specialist_list()
doctor_df_g, cases_df_g = load_dataframes()
embedding_model_g = load_embedding_model() # Needed for recommend_doctors
print("graph_builder.py: Shared resources loaded.")

# --- Helper Function Definitions ---

def extract_specialist(recommendation_text: Optional[str], specialists_list: List[str]) -> Optional[str]:
    """Extracts specialist category from text using a predefined list."""
    if not recommendation_text or not specialists_list: return None
    print(f"Attempting to extract specialist from: '{recommendation_text[:100]}...'")
    recommendation_lower = recommendation_text.lower()
    specialists_list_sorted = sorted(specialists_list, key=len, reverse=True) # Match longer names first
    for specialist in specialists_list_sorted:
        try:
            pattern = r'\b' + re.escape(specialist.lower()) + r'\b' # Word boundaries
            if re.search(pattern, recommendation_lower):
                print(f"Extracted specialist: {specialist}")
                return specialist
        except re.error as e:
            print(f"Regex error processing specialist '{specialist}': {e}")
            continue
    print("No specific specialist match found.")
    return None

# --- PASTE YOUR FULL recommend_doctors FUNCTION HERE ---
# Ensure it's the complete version from the previous step, including imports if needed within it
def recommend_doctors(
    category: Optional[str],
    symptoms: str,
    doctor_df: Optional[pd.DataFrame],
    cases_df: Optional[pd.DataFrame],
    embeddings_model # Expects the loaded embedding model instance
) -> Optional[pd.DataFrame]:
    """Finds and recommends doctors based on category and symptoms."""
    print(f"\n--- recommend_doctors Function (Agent Context) ---")
    print(f"Category: {category}, Symptoms: '{symptoms[:50]}...'")
    # Use is None checks for DataFrames which is more robust
    if doctor_df is None or cases_df is None or embeddings_model is None or not category or not symptoms:
        print("Recommend Doctors: Missing required inputs.")
        return pd.DataFrame()

    try:
        # Filtering
        filtered_doctor_df = doctor_df[doctor_df["Specialty"].str.lower() == category.lower()].copy()
        filtered_cases_df = cases_df[cases_df["Specialty"].str.lower() == category.lower()].copy()

        if filtered_cases_df.empty:
            print(f"Debug: No historical cases found for category: {category}")
            return pd.DataFrame()

        # Ensure necessary columns exist and handle potential NaNs
        required_case_cols = ["Symptom Description", "Doctor ID", "Patient Feedback Rating"]
        if not all(col in filtered_cases_df.columns for col in required_case_cols):
             print("Debug: Required columns missing in patient cases data.")
             return pd.DataFrame()

        filtered_cases_df.dropna(subset=required_case_cols, inplace=True)
        if filtered_cases_df.empty:
             print(f"Debug: No cases with complete data found for category: {category}")
             return pd.DataFrame()

        # Embeddings & FAISS
        symptom_texts = filtered_cases_df["Symptom Description"].tolist()
        print(f"Generating embeddings for {len(symptom_texts)} cases in {category}...")
        embeddings_list = embeddings_model.embed_documents(symptom_texts)
        embeddings_np = np.array(embeddings_list).astype("float32")

        if embeddings_np.size == 0:
            print("Debug: Failed to generate case embeddings (result is empty).")
            return pd.DataFrame()

        d = embeddings_np.shape[1]
        temp_index = faiss.IndexFlatL2(d)
        temp_index.add(embeddings_np)
        print(f"Temp FAISS index built: {temp_index.ntotal} vectors.")

        # Case Details Map
        case_details_map = [{'Doctor ID': filtered_cases_df.iloc[i]["Doctor ID"],
                           'Symptom': filtered_cases_df.iloc[i]["Symptom Description"],
                           'Rating': filtered_cases_df.iloc[i]["Patient Feedback Rating"]}
                          for i in range(len(filtered_cases_df))]

        # User Query Embedding & Search (Using TEXT symptoms from state)
        print(f"Embedding user text symptoms for doctor search: '{symptoms[:100]}...'")
        query_embedding_list = embeddings_model.embed_query(symptoms)
        query_embedding = np.array([query_embedding_list]).astype("float32")

        top_k_cases = 10
        k_to_search = min(top_k_cases, temp_index.ntotal)
        if k_to_search == 0:
            print("Debug: Nothing to search in index (k_to_search is 0)")
            return pd.DataFrame()

        print(f"Starting FAISS search for top {k_to_search} cases...")
        distances, indices = temp_index.search(query_embedding, k_to_search)
        print(f"FAISS search completed. Indices shape: {indices.shape}")

        # Gather Similar Cases & Scores
        similar_cases_data = []
        if indices.size > 0 and indices[0][0] != -1:
            valid_indices_found = 0
            for i_idx in indices[0]:
                if i_idx >= 0 and i_idx < len(case_details_map) and i_idx < len(embeddings_np):
                    case_info = case_details_map[i_idx]
                    case_embedding = embeddings_np[i_idx]
                    if query_embedding.shape[1] == case_embedding.shape[0]:
                         sim_score = cosine_similarity(query_embedding, [case_embedding])[0][0]
                         similar_cases_data.append({
                             'Doctor ID': case_info["Doctor ID"],
                             'Symptom': case_info["Symptom"],
                             'Rating': case_info["Rating"],
                             'Similarity Score': round(float(sim_score), 4)
                         })
                         valid_indices_found += 1
                    else: print(f"Warning: Embedding shape mismatch idx {i_idx}")
                else: print(f"Warning: Invalid index {i_idx}")
            print(f"Processed {valid_indices_found} valid indices.")
        else: print("Debug: FAISS search returned no valid indices.")

        if not similar_cases_data:
             print("Debug: No similar cases found after processing FAISS results.")
             return pd.DataFrame()

        similar_cases_df_result = pd.DataFrame(similar_cases_data)
        print(f"Created similar_cases_df_result with shape {similar_cases_df_result.shape}")

        # Aggregate, Merge, Rank
        similar_cases_df_result.dropna(subset=["Doctor ID", "Rating", "Similarity Score"], inplace=True)
        if similar_cases_df_result.empty:
            print("Debug: No cases left after dropping NaNs.")
            return pd.DataFrame()

        print("Aggregating scores per doctor...")
        doctor_scores_from_similar = similar_cases_df_result.groupby("Doctor ID").agg(
            avg_rating_from_similar=('Rating', 'mean'),
            num_similar_cases=('Doctor ID', 'count'),
            max_similarity_score=('Similarity Score', 'max')
        ).reset_index()
        print(f"Aggregated scores shape: {doctor_scores_from_similar.shape}")

        if "Doctor ID" not in filtered_doctor_df.columns:
            print("Error: 'Doctor ID' column missing in filtered_doctor_df.")
            return pd.DataFrame()
        try:
            doctor_scores_from_similar['Doctor ID'] = doctor_scores_from_similar['Doctor ID'].astype(str)
            filtered_doctor_df['Doctor ID'] = filtered_doctor_df['Doctor ID'].astype(str)
        except Exception as e_type:
            print(f"Warning: Could not convert Doctor ID to string for merge: {e_type}")
            return pd.DataFrame() # Fail merge if IDs can't be string

        print("Merging aggregated scores with doctor profiles...")
        recommended_doctors = pd.merge(
            doctor_scores_from_similar, filtered_doctor_df, on="Doctor ID", how="left")
        print(f"Merged recommendations shape: {recommended_doctors.shape}")

        # Rank
        sort_cols = ["max_similarity_score", "avg_rating_from_similar", "num_similar_cases"]
        sort_cols_present = [col for col in sort_cols if col in recommended_doctors.columns]
        if sort_cols_present:
             print(f"Ranking doctors by: {sort_cols_present}")
             recommended_doctors = recommended_doctors.sort_values(by=sort_cols_present, ascending=[False] * len(sort_cols_present))
        else: print("Warning: No columns found to rank doctors.")

        # Select Top N & Clean
        top_n_doctors = 5
        print(f"Selecting top {top_n_doctors} doctors...")
        final_recommendation_df = recommended_doctors.head(top_n_doctors).copy()
        print(f"Final recommendation DF shape before cleaning: {final_recommendation_df.shape}")

        final_recommendation_df.rename(columns={
            'avg_rating_from_similar': 'Avg Rating (Similar Cases)',
            'num_similar_cases': 'Similar Cases Found',
            'max_similarity_score': 'Max Similarity Score'}, inplace=True)

        display_cols = ["Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)", "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"]
        print("Cleaning final columns and types...")
        for col in display_cols:
            if col not in final_recommendation_df.columns:
                print(f"Adding missing display column: {col}")
                final_recommendation_df[col] = None
            if col in final_recommendation_df.columns:
                if final_recommendation_df[col].isnull().all():
                    if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']: final_recommendation_df[col] = ""
                    else: final_recommendation_df[col] = 0
                    print(f"Column '{col}' is all null, filled with default.")
                    continue
                try:
                    if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']: final_recommendation_df[col] = final_recommendation_df[col].apply(lambda x: str(x) if pd.notna(x) else "").astype(str)
                    elif col in ['Avg Rating (Similar Cases)', 'Max Similarity Score']: final_recommendation_df[col] = pd.to_numeric(final_recommendation_df[col], errors='coerce').fillna(0.0).round(2)
                    elif col in ['Similar Cases Found', 'Years of Experience']: final_recommendation_df[col] = pd.to_numeric(final_recommendation_df[col], errors='coerce').fillna(0).astype(int)
                except Exception as type_e:
                    print(f"Warning: Type conversion failed for column '{col}': {type_e}")
                    if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']: final_recommendation_df[col] = ""
                    else: final_recommendation_df[col] = 0

        print("Returning final recommendation DataFrame from recommend_doctors.")
        return final_recommendation_df

    except Exception as e:
        print(f"An unexpected error occurred within recommend_doctors function: {e}")
        traceback.print_exc()
        return pd.DataFrame() # Return empty DataFrame on error
# --- END OF recommend_doctors FUNCTION ---


# --- Chain Initialization ---
# Define variables first, initializing to None
intent_chain: Optional[LLMChain] = None
followup_chain: Optional[LLMChain] = None
relevance_check_chain: Optional[LLMChain] = None
medical_info_chain: Optional[LLMChain] = None
final_specialist_chain: Optional[LLMChain] = None
explanation_evaluator_chain: Optional[LLMChain] = None
explanation_refiner_chain: Optional[LLMChain] = None
symptom_sufficiency_chain: Optional[LLMChain] = None # Added sufficiency chain
rag_relevance_evaluator_chain: Optional[LLMChain] = None # Added this line


if llm:
    try:
        print("Initializing LangChain LLMChains...")
        intent_chain = LLMChain(llm=llm, prompt=prompts.intent_classifier_prompt)
        followup_chain = LLMChain(llm=llm, prompt=prompts.followup_prompt)
        relevance_check_chain = LLMChain(llm=llm, prompt=prompts.relevance_check_prompt)
        medical_info_chain = LLMChain(llm=llm, prompt=prompts.medical_info_prompt)
        final_specialist_chain = LLMChain(llm=llm, prompt=prompts.final_specialist_prompt)
        explanation_evaluator_chain = LLMChain(llm=llm, prompt=prompts.explanation_evaluator_prompt)
        explanation_refiner_chain = LLMChain(llm=llm, prompt=prompts.explanation_refiner_prompt)
        symptom_sufficiency_chain = LLMChain(llm=llm, prompt=prompts.symptom_sufficiency_prompt)
        # Ensure rag_relevance_evaluator_prompt exists in prompts.py
        rag_relevance_evaluator_chain = LLMChain(llm=llm, prompt=prompts.rag_relevance_evaluator_prompt) # Initialize new chain
        print("LLMChains initialized.")
    except AttributeError as attr_e:
         print(f"ERROR: Failed to initialize chains. Missing prompt definition? {attr_e}")
         traceback.print_exc()
         # Set all to None
         intent_chain, followup_chain, relevance_check_chain, medical_info_chain, \
         final_specialist_chain, explanation_evaluator_chain, explanation_refiner_chain, \
         symptom_sufficiency_chain, rag_relevance_evaluator_chain = \
             None, None, None, None, None, None, None, None, None
    except Exception as chain_e:
        print(f"ERROR: Failed to initialize one or more LLMChains: {chain_e}")
        traceback.print_exc()
        # Set all to None
        intent_chain, followup_chain, relevance_check_chain, medical_info_chain, \
        final_specialist_chain, explanation_evaluator_chain, explanation_refiner_chain, \
        symptom_sufficiency_chain, rag_relevance_evaluator_chain = \
            None, None, None, None, None, None, None, None, None
else:
    print("Warning: LLM not loaded, chains cannot be initialized.")

# --- End Chain Initialization ---


MAX_REFINE_LOOPS = 2 # Max times to run the refine->evaluate loop

# --- Helper Function for Multimodal LLM Input ---
def _prepare_llm_input(state: AgentState, use_history=True, include_image=True):
    """Helper to construct input for LLM, potentially multimodal."""
    content_list = []
    # Add text part
    text_input = state.get('user_query', '') # Or combine history based on use_history flag
    if text_input:
         content_list.append({"type": "text", "text": text_input})

    # Add image part if available and requested
    image_bytes = state.get('uploaded_image_bytes') if include_image else None
    if image_bytes:
        try:
            # Encode bytes as base64 for data URI scheme
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            # TODO: Detect mime type more reliably if needed (e.g., using 'magic' library)
            mime_type = "image/jpeg" # Assume jpeg for now, adjust if handling png etc.
            image_part = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
            }
            content_list.append(image_part)
            print("Prepared multimodal input with image.")
        except Exception as e_img:
            print(f"Error encoding image for LLM input: {e_img}") # Log error but continue with text

    # Handle history - Gemini prefers roles interleaved
    # Combine history with current input for context if needed by the specific node
    # This example focuses on passing the *latest* query + image multimodally
    # Complex history handling might be needed for some nodes
    if not content_list: return None # Cannot invoke with empty content

    # Return list of messages for models like Gemini
    # Needs adaptation based on specific chain/LLM requirements
    # For direct LLM call:
    # messages = [HumanMessage(content=content_list)]
    # return messages

    # For simple chain invoke expecting dict: return prepared content list?
    # Needs careful checking of how each chain/LLM handles multimodal input.
    # Let's try returning the list of content parts, assuming nodes adapt.
    return content_list


# --- Node Function Definitions ---

def classify_intent_node(state: AgentState) -> dict:
    """Classifies the user's intent considering text and optional image."""
    print("\n--- Node: classify_intent ---")
    if not intent_chain:
        print("Error: Intent chain not available.")
        return {"user_intent": "OFF_TOPIC"}

    # Prepare multimodal input for the LLM call
    # Pass current query and image (if any)
    history_str = "\n".join([f"{msg.get('role','?')}: {msg.get('content','?')[:50]}..." for msg in state['conversation_history'][:-1]]) # Summary for prompt text
    latest_query = state.get('user_query', '')
    image_bytes = state.get('uploaded_image_bytes')

    # Construct multimodal message content
    llm_input_content = [{"type": "text", "text": prompts.intent_classifier_prompt.format(conversation_history=history_str, user_query=latest_query)}]
    if image_bytes:
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = "image/jpeg" # Assume JPEG
            llm_input_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
            })
            print("Added image to intent classification input.")
        except Exception as e_enc:
             print(f"Warning: Could not encode image for intent check: {e_enc}")

    print(f"Classifying intent for query: '{latest_query[:100]}...'")
    try:
        # Invoke LLM directly with multimodal message
        # Note: LLMChain might not directly support multimodal invoke this way,
        # direct llm.invoke is often needed for multimodal messages.
        # Let's call the LLM directly for multimodal nodes.
        if not llm: raise ValueError("LLM not loaded")
        message = HumanMessage(content=llm_input_content)
        response = llm.invoke([message]) # Pass list of messages
        intent = response.content.strip().upper()

        valid_intents = ['SYMPTOM_TRIAGE', 'MEDICAL_INFORMATION_REQUEST', 'OFF_TOPIC']
        if intent not in valid_intents:
            print(f"Warning: LLM returned invalid intent '{intent}'. Defaulting to OFF_TOPIC.")
            intent = 'OFF_TOPIC'
        print(f"Intent classified as: {intent}")
        # Clear previous triage state
        return {
            "user_intent": intent, "uploaded_image_bytes": None, # Consume image after use
            "is_relevant": None, "rag_context": None, "matched_icd_codes": None,
            "initial_explanation": None, "evaluator_critique": None, "loop_count": 0,
            "final_explanation": None, "recommended_specialist": None,
            "doctor_recommendations": None, "no_doctors_found_specialist": None,
            "final_response": None
            }
    except Exception as e:
        print(f"Error in intent classification LLM call: {e}")
        traceback.print_exc()
        return {"user_intent": "OFF_TOPIC", "uploaded_image_bytes": None} # Fail safe, consume image


def gather_symptoms_node(state: AgentState) -> dict:
    """Asks follow-up or determines if enough info exists via LLM check, considering images."""
    print("\n--- Node: gather_symptoms ---")
    # Accumulate TEXT symptoms
    current_symptoms = state.get('accumulated_symptoms', "")
    latest_query = state.get('user_query', '')
    if latest_query and state.get('user_intent') == 'SYMPTOM_TRIAGE':
         new_symptoms = f"{current_symptoms}\nUser: {latest_query}".strip()
         # Add placeholder if image was provided with this query
         if state.get('uploaded_image_bytes'):
             new_symptoms += " [User provided an image]"
    else:
         new_symptoms = current_symptoms
    updates = {"accumulated_symptoms": new_symptoms}

    # LLM Sufficiency Check (still primarily text-based for simplicity)
    sufficient_info = False
    if not symptom_sufficiency_chain:
        print("Warning: Sufficiency check chain unavailable. Assuming sufficient.")
        sufficient_info = True # Fallback
    elif not new_symptoms.strip():
        print("No text symptoms accumulated yet. Asking first follow-up.")
        sufficient_info = False
    else:
        try:
            print(f"Checking sufficiency of TEXT: '{new_symptoms[:100]}...'")
            response = symptom_sufficiency_chain.invoke({"accumulated_symptoms": new_symptoms})
            sufficiency_result = response.get('text', 'NO').strip().upper()
            sufficient_info = (sufficiency_result == 'YES')
            print(f"Sufficiency check result: '{sufficiency_result}' -> Sufficient={sufficient_info}")
        except Exception as e:
            print(f"Error during sufficiency check: {e}. Proceeding cautiously.")
            sufficient_info = True # Fail safe

    # Decide Action
    if not sufficient_info:
        print("Information deemed insufficient. Asking follow-up question...")
        if not followup_chain:
            updates["final_response"] = "Error: Follow-up service unavailable."
        else:
            try:
                # Prepare potentially multimodal input for follow-up LLM
                followup_input_content = [{"type": "text", "text": prompts.followup_prompt.format(accumulated_symptoms=new_symptoms, user_query=latest_query)}]
                image_bytes = state.get('uploaded_image_bytes')
                if image_bytes:
                     encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                     mime_type = "image/jpeg" # Assume JPEG
                     followup_input_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}})
                     print("Added image to follow-up input.")

                # Use direct LLM call for multimodal
                if not llm: raise ValueError("LLM not loaded")
                message = HumanMessage(content=followup_input_content)
                response = llm.invoke([message])
                followup_question = response.content.strip()
                updates["final_response"] = followup_question
                print(f"Generated Follow-up: {followup_question}")
            except Exception as e:
                print(f"Error asking follow-up question: {e}")
                traceback.print_exc()
                updates["final_response"] = "Sorry, I encountered an technical difficulty asking a follow-up."
    else:
        print("Information deemed sufficient for analysis.")
        # Do not set final_response

    # Consume image after processing in this node
    updates["uploaded_image_bytes"] = None
    return updates


def check_triage_relevance_node(state: AgentState) -> dict:
    """Checks if the accumulated TEXT symptoms seem medically relevant."""
    print("\n--- Node: check_triage_relevance ---")
    # This node primarily checks text relevance
    if not relevance_check_chain:
        print("Warning: Relevance check chain unavailable. Assuming relevant.")
        return {"is_relevant": True} # Fail safe

    symptoms_to_check = state.get('accumulated_symptoms', "")
    # Remove image placeholder for relevance check if present
    symptoms_text_only = symptoms_to_check.replace("[User provided an image]", "").strip()

    if not symptoms_text_only:
        print("No text symptoms accumulated to check relevance for. Assuming irrelevant.")
        return {"is_relevant": False}

    print(f"Checking relevance of TEXT: '{symptoms_text_only[:100]}...'")
    try:
        response = relevance_check_chain.invoke({"accumulated_symptoms": symptoms_text_only})
        relevance_text = response.get('text', 'NO').strip().upper()
        is_relevant = (relevance_text == 'YES')
        print(f"Relevance check result: {relevance_text} -> Relevant={is_relevant}")
        return {"is_relevant": is_relevant}
    except Exception as e:
        print(f"Error during relevance check: {e}")
        traceback.print_exc()
        return {"is_relevant": True} # Fail safe, assume relevant


def handle_info_request_node(state: AgentState) -> dict:
    """
    Handles general medical information requests.
    1. Attempts retrieval from internal RAG.
    2. Uses LLM to evaluate if RAG context is relevant/sufficient for the query.
    3. If RAG is insufficient, falls back to Google Search tool.
    4. If Search fails, returns an error.
    5. Uses LLM to synthesize an answer based on the chosen context (RAG or Search).
    """
    print("\n--- Node: handle_info_request ---")
    user_query = state.get('user_query', '')
    if not user_query:
        # Consume image if passed erroneously
        return {"final_response": "Error: No query provided for information request.", "uploaded_image_bytes": None}

    print(f"Handling info request for query: '{user_query[:100]}...'")
    context = ""
    source_used = "Internal Knowledge Base (RAG)" # Default source
    updates = {"uploaded_image_bytes": None} # Consume image if passed to this node

    # 1. Attempt Internal RAG
    print("Attempting retrieval from internal knowledge base...")
    rag_context = tools.retrieve_relevant_documents.invoke({"user_symptoms": user_query})
    updates["rag_context"] = rag_context # Store RAG context for potential display

    # Define basic failure conditions for RAG tool itself
    rag_tool_failed_msgs = ["Error:", "No relevant documents found.", "No relevant text content found in retrieved documents."]
    rag_tool_failed = any(rag_context.startswith(msg) for msg in rag_tool_failed_msgs)

    is_rag_sufficient = False # Default to insufficient
    if not rag_tool_failed:
        # 2. Evaluate RAG Context Relevance using LLM (if RAG tool didn't fail)
        print("Evaluating relevance of retrieved RAG context...")
        if not rag_relevance_evaluator_chain:
            print("Warning: RAG relevance evaluator chain unavailable. Assuming RAG is sufficient.")
            is_rag_sufficient = True # Fallback if evaluator fails
        else:
            try:
                # Ensure chain is callable
                if callable(getattr(rag_relevance_evaluator_chain, "invoke", None)):
                    eval_response = rag_relevance_evaluator_chain.invoke({
                        "rag_context": rag_context,
                        "user_query": user_query
                    })
                    eval_result = eval_response.get('text', 'NO').strip().upper()
                    is_rag_sufficient = (eval_result == 'YES')
                    print(f"RAG Context Relevance LLM Eval: '{eval_result}' -> Sufficient={is_rag_sufficient}")
                else:
                    print("Error: rag_relevance_evaluator_chain not callable. Assuming sufficient.")
                    is_rag_sufficient = True # Fallback
            except Exception as eval_e:
                print(f"Error during RAG relevance evaluation: {eval_e}. Assuming RAG is sufficient.")
                traceback.print_exc()
                is_rag_sufficient = True # Fallback on error

    # 3. Decide whether to use RAG context or fallback to Web Search
    if is_rag_sufficient:
        print("Using context from Internal RAG.")
        context = rag_context
    else:
        # Log why fallback is happening
        if rag_tool_failed:
             print(f"Internal RAG tool failed ({rag_context[:60]}...). Falling back to web search.")
        else:
             print(f"Internal RAG context deemed insufficient by LLM Eval. Falling back to web search.")

        # 4. Fallback to Google Search Tool
        # Check if the tool function exists before calling
        if hasattr(tools, 'google_search') and callable(getattr(tools.google_search, "invoke", None)):
            search_results = tools.google_search.invoke({"query": user_query})
        else:
             print("Error: Google search tool is not available or not configured correctly.")
             search_results = "Error: Web search tool unavailable."


        search_failed_msgs = ["Error:", "No relevant results found via web search."]
        is_search_sufficient = not any(search_results.startswith(msg) for msg in search_failed_msgs)

        if is_search_sufficient:
            print("Using context from Web Search.")
            context = search_results
            source_used = "Web Search (Google CSE)"
            # Clear RAG context in state if using search, as it wasn't relevant
            updates["rag_context"] = None
        else:
            print(f"Web search also failed or found nothing ({search_results[:60]}...). Cannot answer.")
            updates["final_response"] = "I couldn't find specific information on that topic in my knowledge base or via web search. Please try rephrasing."
            return updates # Return immediately

    # 5. Synthesize Answer using LLM with the chosen context
    if not medical_info_chain:
        updates["final_response"] = "Error: Information processing service unavailable."
        return updates

    try:
        print(f"Invoking LLM for Q&A based on context from {source_used}...")
        # Ensure chain is callable
        if callable(getattr(medical_info_chain, "invoke", None)):
            response = medical_info_chain.invoke({
                "rag_context": context, # Pass either RAG context or Search results
                "user_query": user_query
            })
            answer = response.get('text', f"Sorry, I couldn't process the information found from {source_used}.").strip()

            # Add appropriate disclaimer
            if source_used == "Web Search (Google CSE)":
                answer += f"\n\n*Disclaimer: Info based on web search ({source_used}). May not be accurate/complete. Not medical advice. Consult a professional.*"
            else: # RAG source
                answer += f"\n\n*Disclaimer: Info based on internal documents. Not medical advice. Consult a professional.*"

            updates["final_response"] = answer
            print("Generated Q&A response.")
        else:
             print("Error: medical_info_chain is not callable.")
             updates["final_response"] = "Error: Q&A service configured incorrectly."

    except Exception as e:
        print(f"Error during final info request LLM call ({source_used}): {e}")
        traceback.print_exc()
        updates["final_response"] = f"Sorry, I encountered an error trying to answer your question using {source_used} results."

    return updates

def perform_final_analysis_node(state: AgentState) -> dict:
    """Performs RAG, ICD matching (text), and initial multimodal specialist recommendation."""
    print("\n--- Node: perform_final_analysis ---")
    # Ensure symptoms are present
    symptoms = state.get('accumulated_symptoms')
    # Remove image placeholder from text symptoms for text-based tools
    symptoms_text_only = symptoms.replace("[User provided an image]", "").strip() if symptoms else ""

    if not symptoms_text_only: # Need some text symptoms even if image exists
        print("Error: No accumulated text symptoms found for final analysis.")
        # Set error state and return
        return {"initial_explanation": "Error: Cannot perform analysis without symptom text.", "loop_count": 0}

    updates = {"loop_count": 0} # Initialize loop count
    try:
        print("Performing RAG retrieval on text symptoms...")
        context = tools.retrieve_relevant_documents.invoke({"user_symptoms": symptoms_text_only})
        updates["rag_context"] = context # Store context (even if error or no results)

        print("Performing ICD code matching on text symptoms...")
        icd_codes_str = tools.match_relevant_icd_codes.invoke({"user_symptoms": symptoms_text_only})
        updates["matched_icd_codes"] = icd_codes_str # Store codes (even if error or no results)

        # Check if RAG or ICD produced errors before calling final LLM
        # Allow "No relevant..." messages to proceed, but block on "Error:"
        # Correct the check to see if string STARTS with Error:
        rag_failed = context.strip().startswith("Error:")
        icd_failed = icd_codes_str.strip().startswith("Error:")

        if rag_failed or icd_failed:
            error_parts = []
            if rag_failed: error_parts.append("context retrieval failed")
            if icd_failed: error_parts.append("ICD matching failed")
            error_msg = f"Error: Failed prerequisite step(s) ({', '.join(error_parts)}) for specialist recommendation."
            print(error_msg)
            updates["initial_explanation"] = error_msg
            return updates # Stop analysis here

        # Prepare multimodal input for the final specialist LLM
        # Use the specific prompt for this step
        formatted_prompt_text = prompts.final_specialist_prompt.format(
            accumulated_symptoms=symptoms, # Pass full accumulated string including image placeholder
            rag_context=context if not context.startswith("No relevant") else "No specific context found.", # Pass placeholder if none found
            matched_icd_codes=icd_codes_str if not icd_codes_str.startswith("No relevant") else "None found." # Pass placeholder
        )
        final_llm_input_content = [{"type": "text", "text": formatted_prompt_text}]

        # Add image if present in the current state turn
        # NOTE: Assumes image relates to the *entire* accumulated context.
        # More complex logic might be needed to track which image belongs where.
        image_bytes = state.get('uploaded_image_bytes')
        if image_bytes:
            try:
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                # TODO: Detect mime type properly if needed
                mime_type = "image/jpeg" # Assume JPEG for now
                final_llm_input_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
                })
                print("Added image to final specialist recommendation input.")
            except Exception as e_enc:
                print(f"Warning: Could not encode image for final analysis: {e_enc}")


        print("Invoking final specialist recommendation LLM directly (multimodal)...")
        if not llm: raise ValueError("LLM not loaded")

        # Create the multimodal message for the LLM
        message_for_final_llm = HumanMessage(content=final_llm_input_content)
        response_obj = llm.invoke([message_for_final_llm]) # Call LLM directly

        # Extract text content from the response object
        initial_explanation = response_obj.content.strip() if hasattr(response_obj, 'content') else "Error: Invalid response from LLM."
        updates["initial_explanation"] = initial_explanation
        print(f"Initial explanation generated: '{initial_explanation[:100]}...'")

    except Exception as e:
        error_msg = f"An unexpected error occurred during final analysis node: {e}"
        print(error_msg)
        traceback.print_exc()
        updates["initial_explanation"] = error_msg # Store error message

    updates["evaluator_critique"] = None # Clear previous critique
    updates["uploaded_image_bytes"] = None # Consume image after use in this node
    return updates

# --- Other Node Functions (evaluate, refine, extract, prepare, handlers) ---
# These generally remain the same as the previous version, operating primarily on text state fields
# Ensure they correctly handle cases where 'initial_explanation' might be an error message.

def evaluate_explanation_node(state: AgentState) -> dict:
    """Evaluates the clarity and simplicity of the generated specialist explanation."""
    print(f"\n--- Node: evaluate_explanation (Loop: {state.get('loop_count', 0)}) ---")
    if not explanation_evaluator_chain:
        print("Warning: Evaluator chain unavailable. Accepting explanation as is.")
        return {"evaluator_critique": "OK"}

    explanation = state.get("initial_explanation")
    if not explanation or explanation.startswith("Error:"):
         print("Skipping evaluation: No valid initial explanation provided.")
         return {"evaluator_critique": "OK"}

    print(f"Evaluating explanation: '{explanation[:100]}...'")
    try:
        # This chain expects text input
        response = explanation_evaluator_chain.invoke({"initial_explanation": explanation})
        critique = response.get('text', "OK").strip()
        if not critique.upper().startswith("REVISE"): critique = "OK"
        print(f"Evaluator critique: {critique}")
        return {"evaluator_critique": critique}
    except Exception as e:
        print(f"Error during explanation evaluation: {e}")
        traceback.print_exc()
        return {"evaluator_critique": "OK"} # Fail safe

def refine_explanation_node(state: AgentState) -> dict:
    """Attempts to refine the explanation based on evaluator critique."""
    print("\n--- Node: refine_explanation ---")
    if not explanation_refiner_chain:
        print("Warning: Refiner chain unavailable. Cannot refine.")
        return {"loop_count": state.get('loop_count', 0) + 1} # Increment loop count anyway

    explanation = state.get("initial_explanation")
    critique = state.get("evaluator_critique")

    if not explanation or explanation.startswith("Error:") or not critique or critique == "OK":
        print("Skipping refinement: No valid input or critique is OK.")
        return {"loop_count": state.get('loop_count', 0) + 1} # Increment loop count

    print(f"Refining explanation based on critique: {critique}")
    try:
        # This chain expects text input
        response = explanation_refiner_chain.invoke({
            "initial_explanation": explanation,
            "evaluator_critique": critique
        })
        refined_explanation = response.get('text', explanation).strip()
        print(f"Explanation refined: '{refined_explanation[:100]}...'")
        return {"initial_explanation": refined_explanation, "loop_count": state.get('loop_count', 0) + 1}
    except Exception as e:
        print(f"Error during explanation refinement: {e}")
        traceback.print_exc()
        return {"loop_count": state.get('loop_count', 0) + 1} # Increment loop count on error

def extract_specialist_and_doctors_node(state: AgentState) -> dict:
    """Extracts specialist name from final explanation and calls doctor recommendation logic."""
    print("\n--- Node: extract_specialist_and_doctors ---")
    final_explanation = state.get("initial_explanation", "Error: No final explanation available.")
    updates = {"final_explanation": final_explanation}

    if final_explanation.startswith("Error:"):
        print("Skipping specialist/doctor steps due to prior error.")
        updates.update({"recommended_specialist": None, "doctor_recommendations": None, "no_doctors_found_specialist": None})
        return updates

    extracted_specialist = extract_specialist(final_explanation, specialist_list_g)
    updates["recommended_specialist"] = extracted_specialist

    if extracted_specialist:
        print(f"Calling recommend_doctors for: {extracted_specialist}")
        symptoms_text_only = state.get('accumulated_symptoms','').replace("[User provided an image]", "").strip()
        if not symptoms_text_only: print("Warning: No text symptoms available for recommend_doctors")

        doctors_df = recommend_doctors(
            category=extracted_specialist, symptoms=symptoms_text_only,
            doctor_df=doctor_df_g, cases_df=cases_df_g, embeddings_model=embedding_model_g
        )
        if isinstance(doctors_df, pd.DataFrame) and not doctors_df.empty:
            print(f"Found {len(doctors_df)} doctor recommendations.")
            updates["doctor_recommendations"] = doctors_df
            updates["no_doctors_found_specialist"] = None
        else:
            print(f"No doctors found or recommended for {extracted_specialist}.")
            updates["no_doctors_found_specialist"] = extracted_specialist
            updates["doctor_recommendations"] = None
    else:
        print("No specific specialist extracted.")
        updates["no_doctors_found_specialist"] = "Unknown"
        updates["doctor_recommendations"] = None

    return updates

def handle_off_topic_node(state: AgentState) -> dict:
    print("\n--- Node: handle_off_topic ---")
    return {"final_response": prompts.off_topic_response, "uploaded_image_bytes": None} # Consume image

def handle_irrelevant_triage_node(state: AgentState) -> dict:
    print("\n--- Node: handle_irrelevant_triage ---")
    return {
        "final_response": prompts.irrelevant_triage_response,
        "rag_context": "N/A (Triage irrelevant)",
        "matched_icd_codes": "N/A (Triage irrelevant)",
        "uploaded_image_bytes": None # Consume image
        }

def prepare_final_output_node(state: AgentState) -> dict:
    print("\n--- Node: prepare_final_output ---")
    # Set final_response based on final_explanation if not already set by other handlers
    if state.get("final_response") is None:
        final_explanation = state.get("final_explanation", "Processing complete.")
        return {"final_response": final_explanation}
    return {} # Keep existing final_response


# --- Conditional Edge Logic Functions --- (Keep as previously defined)
def route_based_on_intent(state: AgentState) -> str:
    intent = state.get("user_intent", "OFF_TOPIC")
    print(f"Routing Edge: Intent is '{intent}'.")
    if intent == "SYMPTOM_TRIAGE": return "gather_symptoms"
    elif intent == "MEDICAL_INFORMATION_REQUEST": return "handle_info_request"
    else: return "handle_off_topic"

def should_continue_symptom_gathering(state: AgentState) -> str:
    if state.get("final_response"):
        print("Routing Edge: Follow-up question asked. Ending turn.")
        return END
    else:
        print("Routing Edge: Symptom gathering complete. Proceeding to relevance check.")
        return "check_triage_relevance"

def route_based_on_relevance(state: AgentState) -> str:
    is_relevant = state.get("is_relevant", False)
    print(f"Routing Edge: Relevance is '{is_relevant}'.")
    if is_relevant: return "perform_final_analysis"
    else: return "handle_irrelevant_triage"

def route_based_on_evaluation(state: AgentState) -> str:
    critique = state.get("evaluator_critique", "OK")
    loop = state.get("loop_count", 0)
    print(f"Routing Edge: Evaluation critique starts with '{critique[:6]}...', Loop count={loop}")
    if critique.upper().startswith("REVISE") and loop < MAX_REFINE_LOOPS:
        print("Decision: Refine explanation.")
        return "refine_explanation"
    else:
        if loop >= MAX_REFINE_LOOPS: print("Decision: Max refinement loops reached.")
        print("Decision: Explanation accepted.")
        return "extract_specialist_and_doctors"


# --- Graph Construction Function --- (Keep as previously defined)
def build_graph():
    if not llm: return None
    print("Building LangGraph workflow...")
    workflow = StateGraph(AgentState)
    print("Adding nodes...")
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("gather_symptoms", gather_symptoms_node)
    workflow.add_node("check_triage_relevance", check_triage_relevance_node)
    workflow.add_node("handle_info_request", handle_info_request_node)
    workflow.add_node("handle_off_topic", handle_off_topic_node)
    workflow.add_node("handle_irrelevant_triage", handle_irrelevant_triage_node)
    workflow.add_node("perform_final_analysis", perform_final_analysis_node)
    workflow.add_node("evaluate_explanation", evaluate_explanation_node)
    workflow.add_node("refine_explanation", refine_explanation_node)
    workflow.add_node("extract_specialist_and_doctors", extract_specialist_and_doctors_node)
    workflow.add_node("prepare_final_output", prepare_final_output_node)
    workflow.set_entry_point("classify_intent")
    print("Adding edges...")
    workflow.add_conditional_edges("classify_intent", route_based_on_intent, {"gather_symptoms": "gather_symptoms", "handle_info_request": "handle_info_request", "handle_off_topic": "handle_off_topic"})
    workflow.add_conditional_edges("gather_symptoms", should_continue_symptom_gathering, {"check_triage_relevance": "check_triage_relevance", END: END})
    workflow.add_conditional_edges("check_triage_relevance", route_based_on_relevance, {"perform_final_analysis": "perform_final_analysis", "handle_irrelevant_triage": "handle_irrelevant_triage"})
    workflow.add_edge("perform_final_analysis", "evaluate_explanation")
    workflow.add_conditional_edges("evaluate_explanation", route_based_on_evaluation, {"extract_specialist_and_doctors": "extract_specialist_and_doctors", "refine_explanation": "refine_explanation"})
    workflow.add_edge("refine_explanation", "evaluate_explanation")
    workflow.add_edge("extract_specialist_and_doctors", "prepare_final_output")
    workflow.add_edge("prepare_final_output", END)
    workflow.add_edge("handle_info_request", END)
    workflow.add_edge("handle_off_topic", END)
    workflow.add_edge("handle_irrelevant_triage", END)
    print("Compiling the graph...")
    compiled_graph = workflow.compile()
    print("LangGraph compiled successfully.")
    return compiled_graph

# --- End Graph Construction ---

# Optional main block for testing graph build standalone
if __name__ == '__main__':
    print("\nTesting graph builder standalone...")
    try:
        if not llm: print("LLM missing, build check might fail.")
        graph_app_test = build_graph()
        if graph_app_test: print("\nStandalone build check successful!")
        else: print("\nStandalone build check resulted in None graph app.")
    except Exception as e:
        print(f"\nError during standalone graph build test: {e}")
        traceback.print_exc()