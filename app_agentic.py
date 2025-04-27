# app_agentic.py
import streamlit as st
import os
import sys
import nltk
import pandas as pd
from dotenv import load_dotenv
import traceback
from typing import Dict, List, Optional, Any # For type hinting
from PIL import Image # Needed to check image type/validity
import io # Needed to read image bytes

# --- Must be first Streamlit command ---
st.set_page_config(page_title="Agentic AI Healthcare Assistant", layout="wide")

# --- Now safe to import custom modules & utils ---
from state import AgentState
from utils import (
    load_embedding_model, load_llm, load_vector_store,
    load_icd_data_and_embeddings, load_dataframes, load_specialist_list
)
# Import the graph builder and build the graph app
# This should be called only once if possible
from graph_builder import build_graph

# --- NLTK Download ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
except Exception as e:
     print(f"Could not check/download NLTK data: {e}")

# --- Load Global Components ---
print("Loading application components...")
embedding_model = load_embedding_model()
llm = load_llm()
vector_store = load_vector_store(embedding_model)
icd_codes_g, icd_embeddings_g = load_icd_data_and_embeddings(embedding_model)
doctor_df_g, cases_df_g = load_dataframes()
specialist_list_g = load_specialist_list()
print("Finished loading application components.")

# --- Build LangGraph Agent ---
@st.cache_resource
def get_compiled_graph():
    """Builds or retrieves the compiled LangGraph agent."""
    print("Attempting to build/retrieve compiled LangGraph agent...")
    if llm and embedding_model and vector_store and specialist_list_g:
        try:
            compiled_app = build_graph()
            if compiled_app:
                 print("LangGraph agent compiled successfully.")
                 return compiled_app
            else:
                 st.error("Graph building function returned None.")
                 return None
        except Exception as e:
            st.error(f"Failed to build the agent workflow graph: {e}")
            print(f"CRITICAL: Graph build failed: {e}")
            traceback.print_exc()
            return None
    else:
        missing_comps = [comp for comp, name in zip([llm, embedding_model, vector_store, specialist_list_g], ['LLM', 'Embeddings', 'VectorStore', 'SpecialistList']) if comp is None]
        error_msg = f"Core components ({', '.join(missing_comps)}) failed to load earlier. Agent cannot be built."
        st.error(error_msg)
        print(error_msg)
        return None

graph_app = get_compiled_graph()

# --- Streamlit UI Elements ---
st.title("🩺 MedMatch(Agentic AI Healthcare Assistant)")

st.warning("Disclaimer: This is a prototype AI for informational purposes only. It cannot provide diagnosis or medical advice. Visual analysis is experimental. Always consult with a qualified healthcare professional.")

# --- Initialize Streamlit Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
    initial_greeting = "Hello! How can I help you today? Describe symptoms, upload an image (optional), or ask a medical question."
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

if 'agent_final_state' not in st.session_state:
    st.session_state.agent_final_state = None

if 'agent_input_history' not in st.session_state:
     st.session_state.agent_input_history = []
     if st.session_state.messages and st.session_state.messages[0]['role'] == 'assistant':
         st.session_state.agent_input_history.append(st.session_state.messages[0])


# --- Display Chat Interface ---
chat_container = st.container(height=450, border=True)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display text content
            if isinstance(message["content"], str):
                 st.markdown(message["content"])
            # Display image if message content indicates one (e.g., user uploaded)
            elif isinstance(message["content"], dict) and message["content"].get("type") == "image_display":
                 st.image(message["content"]["image_bytes"], caption="Uploaded Image", width=200)
                 if message["content"].get("text"): # Display accompanying text if any
                     st.markdown(message["content"]["text"])


# --- Separate Display Area for Results (Table, Expander) ---
results_container = st.container()

# --- User Input Area ---
# Use columns for better layout of text input and file uploader
input_col, upload_col = st.columns([4, 1])

with input_col:
    prompt = st.chat_input("Describe symptoms or ask a question...")

with upload_col:
    # Add file uploader for images
    uploaded_file = st.file_uploader(
        "Upload Image (Optional)",
        type=["png", "jpg", "jpeg", "webp"],
        key="file_uploader" # Assign a key
    )

# --- Agent Execution Logic (Triggered by User Text Input OR File Upload + Text) ---
# We trigger when the text prompt is submitted
if prompt:
    user_text_input = prompt
    user_image_bytes = None
    image_caption = None

    # Process uploaded file if it exists when prompt is submitted
    if uploaded_file is not None:
        try:
            # Read image bytes
            user_image_bytes = uploaded_file.getvalue()
            # Optional: Validate image format/size
            img = Image.open(io.BytesIO(user_image_bytes))
            print(f"Image uploaded: {uploaded_file.name}, size: {len(user_image_bytes)} bytes, format: {img.format}")
            image_caption = f"Image: {uploaded_file.name}" # For display history
            # Clear the uploader after processing by resetting its key or form
            # This requires more complex state management or using st.form
            # Simple approach: User needs to manually clear it if needed.
        except Exception as e_img:
            st.error(f"Error processing uploaded image: {e_img}")
            user_image_bytes = None # Don't proceed with corrupted image

    # 1. Add user input to display history (handle text + optional image)
    if user_image_bytes:
        # Store as a dict for special display handling
        display_content = {
            "type": "image_display",
            "text": user_text_input,
            "image_bytes": user_image_bytes
        }
        st.session_state.messages.append({"role": "user", "content": display_content})
        # For agent history, represent image existence conceptually or pass bytes later
        # Let's add text + image marker to agent history
        st.session_state.agent_input_history.append({"role": "user", "content": f"{user_text_input} [Image Uploaded: {uploaded_file.name}]"})
    else:
        # Just add text
        st.session_state.messages.append({"role": "user", "content": user_text_input})
        st.session_state.agent_input_history.append({"role": "user", "content": user_text_input})

    # 2. Display user message immediately (handled by chat history display loop)

    # 3. Check if the agent graph is ready
    if not graph_app:
        error_msg = "Sorry, the AI agent is currently unavailable."
        with chat_container:
             with st.chat_message("assistant"): st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.agent_final_state = None
        st.rerun()
    else:
        # 4. Prepare the input state for the graph
        current_agent_history = st.session_state.agent_input_history
        accumulated_symptoms_str = "\n".join([ str(msg.get('content','')) for msg in current_agent_history ]) # Simple concat

        initial_graph_input: AgentState = {
            "conversation_history": current_agent_history,
            "user_query": user_text_input,
            "uploaded_image_bytes": user_image_bytes, # Pass image bytes for this turn
            "image_prompt_text": user_text_input if user_image_bytes else None, # Text associated with image
            "accumulated_symptoms": accumulated_symptoms_str,
            "loop_count": 0,
            "user_intent": None, "is_relevant": None, "rag_context": None,
            "matched_icd_codes": None, "initial_explanation": None,
            "evaluator_critique": None, "final_explanation": None,
            "recommended_specialist": None, "doctor_recommendations": None,
            "no_doctors_found_specialist": None, "final_response": None
        }

        # 5. Execute the agent graph
        final_state: Optional[Dict[str, Any]] = None
        with st.spinner("Agent processing your request..."):
            try:
                print("\nInvoking Agent Graph...")
                final_state = graph_app.invoke(initial_graph_input, {"recursion_limit": 15}) # Increased limit slightly
                st.session_state.agent_final_state = final_state
                print("\n--- Agent Run Completed: Final State ---")
                if final_state:
                    # Log state concisely
                    for key, value in final_state.items():
                         if key == "conversation_history": print(f"  {key}: List len {len(value)}")
                         elif key == "uploaded_image_bytes": print(f"  {key}: {'Present' if value else 'None'}")
                         elif isinstance(value, pd.DataFrame): print(f"  {key}: DataFrame shape {value.shape}")
                         elif isinstance(value, list) and len(value) > 5: print(f"  {key}: List len {len(value)}")
                         elif isinstance(value, str) and len(value) > 100 : print(f"  {key}: {value[:100]}...")
                         elif value is not None: print(f"  {key}: {value}")
                print("--------------------------------------\n")

            except Exception as e:
                print(f"Error invoking agent graph: {e}")
                traceback.print_exc()
                error_msg = f"An error occurred while processing your request: {e}"
                # Error will be added to history below
                st.session_state.agent_final_state = None # Clear state on error

        # 6. Extract primary text response for chat history
        agent_response = final_state.get("final_response") if final_state else None

        # Add agent's response (or error) to display history
        if agent_response:
            st.session_state.messages.append({"role": "assistant", "content": agent_response})
            st.session_state.agent_input_history.append({"role": "assistant", "content": agent_response})
        elif graph_app and final_state is None: # Error during invoke
             error_msg = f"An error occurred during processing: Check logs." # Already printed traceback
             st.session_state.messages.append({"role": "assistant", "content": error_msg})
             # Don't add error to agent_input_history
        elif graph_app and final_state is not None and not agent_response: # Graph ran but no response text
            fallback_response = "Processing complete. Please let me know if there's anything else."
            print("Warning: Agent finished but no 'final_response' in state. Using fallback.")
            st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            st.session_state.agent_input_history.append({"role": "assistant", "content": fallback_response})
        # Else: Error case where graph_app itself was None already handled

        # 7. Rerun Streamlit to update chat display and show results below
        st.rerun()


# --- Display Results Section (Reads from stored final state) ---
if st.session_state.agent_final_state:
    final_state_to_display = st.session_state.agent_final_state
    with results_container:
        # Display Doctor Recommendations Table if available
        doctors_df = final_state_to_display.get("doctor_recommendations")
        if isinstance(doctors_df, pd.DataFrame) and not doctors_df.empty:
            st.markdown("---")
            st.markdown("##### ✅ Recommended Doctors")
            display_cols = ["Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)", "Max Similarity Score", "Similar Cases Found", "Years of Experience", "Affiliation"]
            display_cols_present = [col for col in display_cols if col in doctors_df.columns]
            if display_cols_present:
                try:
                    st.dataframe(doctors_df[display_cols_present], hide_index=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying doctor table: {e}")
                    print(f"Error displaying doctors DataFrame: {e}")
                    st.write("Doctor data (raw):", doctors_df)
            else: st.warning("Doctor data generated, but required display columns missing.")

        # Display "No Doctors Found" Message
        no_docs_specialist = final_state_to_display.get("no_doctors_found_specialist")
        if no_docs_specialist:
             if not (isinstance(doctors_df, pd.DataFrame) and not doctors_df.empty): st.markdown("---")
             if no_docs_specialist != "Unknown": st.info(f"ℹ️ No doctor profiles found for **{no_docs_specialist}** matching symptoms in data.")
             else: st.info("ℹ️ Could not determine a specialist to search for doctors.")

        # Display Context Expander
        rag_context = final_state_to_display.get("rag_context")
        icd_codes = final_state_to_display.get("matched_icd_codes")
        accumulated_symptoms_display = final_state_to_display.get("accumulated_symptoms")

        # Determine if there's relevant context/codes to show
        show_expander = False
        # Check if rag_context is valid string and not an error/NA message from the tool itself
        if rag_context and isinstance(rag_context, str) and not rag_context.startswith(("N/A", "Error:")) and rag_context != "No relevant documents found.":
            show_expander = True
        # Check if icd_codes is valid string and not an error/NA message from the tool itself
        if icd_codes and isinstance(icd_codes, str) and not icd_codes.startswith(("N/A", "Error:")) and icd_codes != "No relevant ICD codes found with sufficient similarity.":
             show_expander = True

        if show_expander:
            # Add separator only if needed
            if not (isinstance(doctors_df, pd.DataFrame) and not doctors_df.empty) and not no_docs_specialist:
                 st.markdown("---")

            with st.expander("🔍 Show Analysis Details (Context & Codes)", expanded=False):
                st.markdown("**Symptoms Used for Analysis (Accumulated Text):**")
                st.text(accumulated_symptoms_display.replace("[User provided an image]", "").strip() if accumulated_symptoms_display else "N/A") # Show cleaned text

                st.markdown("**Matched ICD10 Codes (Informational):**")
                st.text(icd_codes if icd_codes else "N/A")

                st.markdown("**Retrieved Context from Knowledge Base:**")
                if rag_context and isinstance(rag_context, str) and not rag_context.startswith(("N/A", "Error:")) and rag_context != "No relevant documents found.":
                     # Split the context string by the '=====' separator used in the tool
                     context_blocks = rag_context.split("\n=====\n")
                     for i, block in enumerate(context_blocks):
                         # Parse the source and content from each block
                         source_line = ""
                         content_text = block # Default to the whole block if parsing fails
                         if block.startswith("Source:") and "\n---\n" in block:
                             parts = block.split("\n---\n", 1)
                             source_line = parts[0].strip() # e.g., "Source: filename.pdf"
                             content_text = parts[1].strip() if len(parts) > 1 else ""
                         # Display source and content
                         st.markdown(f"--- **Chunk {i+1}** ({source_line}) ---")
                         st.text_area(f"chunk_content_{i}", value=content_text, height=150, disabled=True, key=f"expander_rag_chunk_{i}", label_visibility="collapsed")
                else:
                    # Display the RAG context value even if it's an error message from the tool or N/A
                    st.text(rag_context if rag_context else "N/A")

# --- Sidebar Elements ---
st.sidebar.header("About")
st.sidebar.info("Agentic AI Healthcare Assistant using LangGraph, Gemini (Multimodal), RAG, and data analysis.")
st.sidebar.header("Data Sources")
st.sidebar.markdown(f"- Local PDFs, articles and Gale Encyclopedia of Medicine in `data/` folder")
st.sidebar.markdown(f"- ICD-10 Mapping")
st.sidebar.markdown(f"- Synthetic Doctor/Case CSVs")
st.sidebar.markdown("*(Note: Data is for demo purposes)*")

# --- Restart Button ---
if st.sidebar.button("🔄 Restart Conversation"):
    print("Restarting conversation...")
    keys_to_clear = ['messages', 'agent_final_state', 'agent_input_history']
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]
    # Re-initialize empty lists and add greeting
    st.session_state.messages = []
    st.session_state.agent_input_history = []
    initial_greeting = "Hello! How can I help you today? Describe symptoms, upload an image (optional), or ask a general medical question."
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.agent_input_history.append({"role": "assistant", "content": initial_greeting})
    st.rerun()