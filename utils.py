# utils.py
import os
import pickle
import pandas as pd
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Import Gemini
from dotenv import load_dotenv
import streamlit as st # Use Streamlit's caching
import traceback

# --- Configuration Constants ---
# Define paths and model names here for consistency
DATA_FOLDER = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
ICD_CSV_PATH = "ICD10-Disease-Mapping.csv"
ICD_CACHE_PATH = "icd_embeddings.pkl"
DOCTOR_PROFILES_PATH = "doctor_profiles_all_specialties.csv"
PATIENT_CASES_PATH = "merged_reviews_new.csv"
SPECIALIST_LIST_PATH = "specialist_categories_list.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Using Gemini Flash as recommended in the project tips
LLM_MODEL_NAME = "gemini-2.0-flash"

# --- Cached Loading Functions ---

@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace embedding model using Streamlit's cache."""
    print("Attempting to load embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"FATAL ERROR loading embedding model: {e}")
        traceback.print_exc()
        st.error(f"Fatal Error: Could not load embedding model '{EMBEDDING_MODEL_NAME}'. Check installation and model name.")
        return None

@st.cache_resource
def load_llm():
    """Loads the Google Gemini LLM using Streamlit's cache."""
    print(f"Attempting to load LLM: {LLM_MODEL_NAME}...")
    load_dotenv() # Load GOOGLE_API_KEY from .env file in the root directory
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("FATAL ERROR: GOOGLE_API_KEY environment variable not found.")
        st.error("Fatal Error: GOOGLE_API_KEY not found. Please ensure it's set in a .env file in the project root.")
        return None
    try:
        # It's good practice for some models like Gemini to handle system prompts
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=api_key,
            convert_system_message_to_human=True # Helps manage conversation flow
            # Optional: Add safety settings if needed, e.g., blocking thresholds
            # from google.generativeai.types import HarmCategory, HarmBlockThreshold
            # safety_settings={
            #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            # }
        )
        # Perform a simple test invocation
        try:
            # Simple text invoke first
            test_result = llm.invoke("Hello!")
            print(f"LLM '{LLM_MODEL_NAME}' loaded and text test invocation successful.")
            # Optional: Add multimodal test if needed, but might be slow/costly on load
            # from langchain_core.messages import HumanMessage
            # llm.invoke([HumanMessage(content=[{"type": "text", "text":"describe this image"}, {"type": "image_url", "image_url": {"url": "YOUR_SAMPLE_IMAGE_URL_OR_DATA_URI"}}])])
            # print("LLM multimodal test invocation successful (if run).")
        except Exception as test_e:
            print(f"Warning: LLM loaded but test invocation failed: {test_e}")
            traceback.print_exc()
            st.warning(f"LLM loaded, but a test call failed. Check API key permissions or model availability: {test_e}")
            # Return the LLM object anyway, assume it might work for specific tasks
            # return None # Safer alternative: return None if test fails
        return llm
    except Exception as e:
        print(f"FATAL ERROR loading LLM '{LLM_MODEL_NAME}': {e}")
        traceback.print_exc()
        st.error(f"Fatal Error: Could not load LLM '{LLM_MODEL_NAME}'. Check API key, model name, and required packages (langchain-google-genai).")
        return None

@st.cache_resource
def load_dataframes():
    """Loads Doctor profiles and Patient cases CSVs using Streamlit's cache."""
    print("Attempting to load dataframes (Doctors, Cases)...")
    doctor_df, cases_df = None, None
    try:
        if os.path.exists(DOCTOR_PROFILES_PATH):
            doctor_df = pd.read_csv(DOCTOR_PROFILES_PATH)
            print(f"Loaded {len(doctor_df)} doctor profiles from '{DOCTOR_PROFILES_PATH}'.")
        else:
            print(f"Warning: Doctor profiles file not found at '{DOCTOR_PROFILES_PATH}'.")
            st.warning(f"Doctor profiles file not found: {DOCTOR_PROFILES_PATH}")

        if os.path.exists(PATIENT_CASES_PATH):
            # Optimise loading if the file is very large by specifying dtypes or using usecols
            # cases_df = pd.read_csv(PATIENT_CASES_PATH, usecols=['Specialty', 'Symptom Description', 'Doctor ID', 'Patient Feedback Rating'], dtype={'Specialty': 'category', 'Doctor ID': 'str'})
            cases_df = pd.read_csv(PATIENT_CASES_PATH)
            print(f"Loaded {len(cases_df)} patient cases from '{PATIENT_CASES_PATH}'.")
        else:
             print(f"Warning: Patient cases file not found at '{PATIENT_CASES_PATH}'.")
             st.warning(f"Patient cases file not found: {PATIENT_CASES_PATH}")

    except Exception as e:
        print(f"Error loading dataframes: {e}")
        traceback.print_exc()
        st.error(f"Error loading doctor/case data from CSVs: {e}")
        # Return None for both if either fails, or handle individually based on need
        doctor_df, cases_df = None, None

    return doctor_df, cases_df

@st.cache_resource
def load_specialist_list():
    """Loads the list of specialist names using Streamlit's cache."""
    print("Attempting to load specialist list...")
    specialist_list = []
    if not os.path.exists(SPECIALIST_LIST_PATH):
        print(f"Warning: Specialist list file not found at '{SPECIALIST_LIST_PATH}'.")
        st.warning(f"Specialist list file not found: {SPECIALIST_LIST_PATH}")
        return specialist_list # Return empty list

    try:
        with open(SPECIALIST_LIST_PATH, "r", encoding='utf-8') as f: # Specify encoding
            specialist_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(specialist_list)} specialists from '{SPECIALIST_LIST_PATH}'.")
    except Exception as e:
        print(f"Error loading specialist list: {e}")
        traceback.print_exc()
        st.error(f"Error loading specialist list: {e}")
        specialist_list = [] # Return empty list on error

    return specialist_list

# Need to pass embedding_model instance to functions needing it
@st.cache_resource
def load_icd_data_and_embeddings(_embeddings): # Pass embedding model instance
    """
    Loads ICD codes/descriptions and their embeddings using Streamlit's cache.
    Computes embeddings if cache is missing, invalid, or outdated.
    Requires the loaded embedding model instance as input.
    """
    print("Attempting to load ICD data and embeddings...")
    icd_codes_list, icd_embeddings_array = [], None

    if not _embeddings:
        print("Error: Cannot load/compute ICD embeddings because the embedding model is not loaded.")
        st.error("Cannot load ICD data: Embedding model failed to load.")
        return icd_codes_list, icd_embeddings_array # Return empty/None

    if not os.path.exists(ICD_CSV_PATH):
        print(f"Warning: ICD mapping CSV file not found at '{ICD_CSV_PATH}'.")
        st.warning(f"ICD mapping file not found: {ICD_CSV_PATH}")
        return icd_codes_list, icd_embeddings_array # Return empty/None

    # Load CSV Data
    try:
        icd_df = pd.read_csv(ICD_CSV_PATH, dtype=str)
        icd_df.columns = [col.strip(" '\"") for col in icd_df.columns]
        required_cols = ["ICD-10-CM CODE", "ICD-10-CM CODE DESCRIPTION"]
        if not all(col in icd_df.columns for col in required_cols):
            print(f"Error: Required columns missing in '{ICD_CSV_PATH}'. Need {required_cols}")
            st.error(f"Required columns missing in ICD CSV: {required_cols}")
            return [], None

        icd_df.dropna(subset=required_cols, inplace=True) # Drop rows missing essential data
        icd_df["ICD-10-CM CODE"] = icd_df["ICD-10-CM CODE"].str.strip(" '\"")
        icd_df["ICD-10-CM CODE DESCRIPTION"] = icd_df["ICD-10-CM CODE DESCRIPTION"].str.strip(" '\"").str.lower()
        icd_codes_list = icd_df["ICD-10-CM CODE"].tolist()
        icd_descriptions = icd_df["ICD-10-CM CODE DESCRIPTION"].tolist()
        print(f"Loaded {len(icd_codes_list)} ICD codes/descriptions from CSV.")

    except Exception as e:
        print(f"Error loading or processing ICD CSV '{ICD_CSV_PATH}': {e}")
        traceback.print_exc()
        st.error(f"Error loading ICD CSV: {e}")
        return [], None # Return empty/None

    # Load or Compute Embeddings
    cached_embeddings = None
    if os.path.exists(ICD_CACHE_PATH) and os.path.getsize(ICD_CACHE_PATH) > 0:
        print(f"Cache file found at '{ICD_CACHE_PATH}'. Attempting to load...")
        try:
            with open(ICD_CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            # Validate cache structure and content
            if isinstance(cached_data, dict) and 'embeddings' in cached_data and 'descriptions' in cached_data:
                if isinstance(cached_data['embeddings'], np.ndarray) and cached_data['descriptions'] == icd_descriptions:
                    cached_embeddings = cached_data['embeddings']
                    print(f"Successfully loaded valid ICD embeddings from cache ({cached_embeddings.shape}).")
                else:
                    print("Cache content mismatch (descriptions or type). Recomputing embeddings.")
            else:
                 print("Cache format invalid (expected dict with 'embeddings' and 'descriptions'). Recomputing.")
        except Exception as e:
            print(f"Error loading ICD cache from '{ICD_CACHE_PATH}': {e}. Recomputing.")
            traceback.print_exc()

    if cached_embeddings is None:
        print(f"Computing ICD embeddings for {len(icd_descriptions)} descriptions (this may take a while)...")
        try:
            # Ensure embedding model is callable
            if not callable(getattr(_embeddings, "embed_documents", None)):
                 raise TypeError("Provided embedding model cannot embed documents.")

            with st.spinner(f"Computing embeddings for {len(icd_descriptions)} ICD codes..."):
                icd_embeddings_list = _embeddings.embed_documents(icd_descriptions)
            icd_embeddings_array = np.array(icd_embeddings_list).astype('float32')
            print(f"ICD embeddings computed ({icd_embeddings_array.shape}).")
            # Try to cache the new embeddings
            try:
                with open(ICD_CACHE_PATH, "wb") as f:
                    pickle.dump({'embeddings': icd_embeddings_array, 'descriptions': icd_descriptions}, f)
                print(f"New ICD embeddings cached to '{ICD_CACHE_PATH}'.")
            except Exception as e_cache:
                  print(f"Warning: Could not save ICD embeddings cache to '{ICD_CACHE_PATH}': {e_cache}")
                  st.warning(f"Could not save ICD embeddings cache: {e_cache}")
        except Exception as e_compute:
            print(f"Error computing ICD embeddings: {e_compute}")
            traceback.print_exc()
            st.error(f"Failed to compute ICD embeddings: {e_compute}")
            return icd_codes_list, None # Return codes but None embeddings
    else:
        icd_embeddings_array = cached_embeddings # Use the valid cache

    return icd_codes_list, icd_embeddings_array

@st.cache_resource
def load_vector_store(_embeddings): # Pass embedding model instance
    """
    Loads or builds the FAISS vector store using Streamlit's cache.
    Requires the loaded embedding model instance.
    """
    print("Attempting to load Vector Store (FAISS index)...")
    if not _embeddings:
        print("Error: Cannot load vector store because the embedding model is not loaded.")
        st.error("Cannot load Vector Store: Embedding model failed to load.")
        return None

    vectorstore = None
    # Attempt to load from cache first
    if os.path.exists(DB_FAISS_PATH):
        print(f"Vector store cache path found at '{DB_FAISS_PATH}'. Attempting to load...")
        try:
            vectorstore = FAISS.load_local(DB_FAISS_PATH, _embeddings, allow_dangerous_deserialization=True)
            # Optional: Add a quick check, e.g., vectorstore.index.ntotal > 0
            if hasattr(vectorstore, 'index') and vectorstore.index.ntotal > 0:
                print(f"FAISS vectorstore loaded successfully from cache ({vectorstore.index.ntotal} vectors).")
            else:
                print("Warning: Loaded vector store seems empty or invalid. Will attempt rebuild.")
                vectorstore = None # Force rebuild if seems empty
        except Exception as e:
            print(f"Error loading vectorstore from cache '{DB_FAISS_PATH}': {e}. Attempting rebuild.")
            traceback.print_exc()
            vectorstore = None # Ensure rebuild is triggered

    # Build if cache loading failed or cache doesn't exist
    if vectorstore is None:
        print("Building vector store from documents...")
        if not os.path.exists(DATA_FOLDER):
            print(f"Error: Data folder '{DATA_FOLDER}' not found. Cannot build vector store.")
            st.error(f"Data folder '{DATA_FOLDER}' not found. Cannot build vector store for RAG.")
            return None

        try:
            # Load documents
            loader = DirectoryLoader(path=DATA_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader, recursive=True, silent_errors=True, show_progress=True)
            documents = loader.load()
            if not documents:
                print(f"Warning: No PDF documents successfully loaded from '{DATA_FOLDER}'. Cannot build vector store.")
                st.warning(f"No PDF documents found in '{DATA_FOLDER}'. Cannot build vector store for RAG.")
                return None

            print(f"Loaded {len(documents)} documents for indexing.")
            # Split documents
            # Consider adjusting chunk size/overlap based on document characteristics
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""], # More separators
                chunk_size=700, # Slightly larger chunk size?
                chunk_overlap=100 # More overlap?
            )
            doc_chunks = splitter.split_documents(documents)
            if not doc_chunks:
                print("Error: Document splitting resulted in zero chunks.")
                st.error("Failed to split documents into chunks for vector store.")
                return None

            print(f"Split documents into {len(doc_chunks)} chunks.")
            # Create FAISS index
            print("Creating FAISS index from chunks (this may take some time)...")
            with st.spinner(f"Building knowledge base index from {len(doc_chunks)} document chunks..."):
                vectorstore = FAISS.from_documents(doc_chunks, embedding=_embeddings)

            if not hasattr(vectorstore, 'index') or vectorstore.index.ntotal == 0:
                 print("Error: FAISS index creation resulted in an empty index.")
                 st.error("Failed to create a valid FAISS index from document chunks.")
                 return None

            print(f"FAISS index created with {vectorstore.index.ntotal} vectors.")
            # Save the newly built index
            try:
                if not os.path.exists("vectorstore"): os.makedirs("vectorstore")
                vectorstore.save_local(DB_FAISS_PATH)
                print(f"FAISS vectorstore built and saved to '{DB_FAISS_PATH}'.")
            except Exception as e_save:
                 print(f"Error saving FAISS vectorstore to '{DB_FAISS_PATH}': {e_save}")
                 st.warning(f"Built vector store but failed to save it: {e_save}")

        except Exception as e_build:
            print(f"Error building FAISS vectorstore: {e_build}")
            traceback.print_exc()
            st.error(f"Could not build FAISS vectorstore: {e_build}")
            vectorstore = None

    return vectorstore