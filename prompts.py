# prompts.py
from langchain_core.prompts import PromptTemplate

# --- Intent Classification (Updated for potential image) ---
intent_classifier_template = """
Analyze the latest user input, which may include text and/or an image, in the context of the conversation history. Classify the primary intent of the *latest user input*.

Conversation History (may include text and image references):
{conversation_history}

Latest User Input Text: {user_query}
(An image may also have been provided simultaneously with this text query).

Possible Intents:
- SYMPTOM_TRIAGE: User is describing medical symptoms, health complaints (textually or visually in an image), or asking for help identifying a potential issue. Prioritize this if an image shows a clear medical concern (rash, injury, etc.) even if text is minimal.
- MEDICAL_INFORMATION_REQUEST: User is asking a general question about a medical condition, treatment, term, or procedure (primarily text-based query).
- OFF_TOPIC: User query/image is unrelated to medical symptoms or information (e.g., greetings, politics, general knowledge, non-medical images).

Classify the intent of the *latest user input (text and/or image)*. Respond with only ONE of the intent labels:
SYMPTOM_TRIAGE
MEDICAL_INFORMATION_REQUEST
OFF_TOPIC
Intent:"""
intent_classifier_prompt = PromptTemplate(
    template=intent_classifier_template,
    input_variables=["conversation_history", "user_query"] # Image passed directly in message content
)

# --- Basic Follow-up (Updated for potential image) ---
followup_template = """
You are a compassionate AI medical assistant. The user is describing symptoms, potentially including text and an image.
Current accumulated TEXT symptoms:
{accumulated_symptoms}

Latest user input text:
{user_query}
(User may have also provided an image related to this input).

Ask a single, clear, concise follow-up question to gather more details.
- If an image was provided with the latest input, your question can reference it (e.g., "Does the rash in the image itch?").
- Otherwise, base the question on the text symptoms.
Focus on aspects like onset, duration, severity, location, appearance (if image present), or related factors. Do not give advice or diagnoses. Keep the question brief.
Follow-up Question:"""
followup_prompt = PromptTemplate(
    template=followup_template,
    input_variables=["accumulated_symptoms", "user_query"] # Image passed directly in message content
)

# --- Triage Relevance Check (Updated for potential image) ---
# Note: This check still primarily focuses on the *accumulated text* for simplicity,
# but acknowledges an image might provide implicit context.
relevance_check_template = """
Analyze the following accumulated symptom description from a user chat. The user might have also provided images during the conversation, although only the text is summarized here.
---
{accumulated_symptoms}
---
Based on the accumulated TEXT description, is this conversation primarily focused on discussing personal medical symptoms, health conditions, or seeking medical guidance related to personal health?
Respond with only the word 'YES' or 'NO'.
Relevance:"""
relevance_check_prompt = PromptTemplate(
    template=relevance_check_template,
    input_variables=["accumulated_symptoms"]
)

# --- ADDED: Symptom Sufficiency Check Prompt ---
symptom_sufficiency_template = """
Analyze the accumulated symptom description provided by a user. Does it contain enough specific detail (e.g., nature of symptom, location, severity, duration, onset, triggers, associated factors) to make a *preliminary assessment* for suggesting a potential medical specialist?

Focus on whether you have concrete details to work with, not on making a diagnosis. Consider that an image might have been provided separately, but evaluate based primarily on the text description's detail level.

Accumulated Text Symptoms:
---
{accumulated_symptoms}
---

Is there enough specific symptom information in the text description to proceed with analysis? Respond with only the word 'YES' or 'NO'.
Sufficient Information:"""
symptom_sufficiency_prompt = PromptTemplate(
    template=symptom_sufficiency_template,
    input_variables=["accumulated_symptoms"]
)
# --- END ADDED ---

# --- Medical Info Q&A (RAG based) ---
medical_info_template = """
You are an AI assistant providing information based *only* on the following context documents retrieved from a medical knowledge base. Do NOT provide medical advice, opinions, or diagnoses. Do not use any external knowledge beyond the provided context.
Answer the user's question factually based *only* on the text within the context. If the context does not contain the answer, clearly state that the information is not available in the provided documents. Keep the answer concise.

Context Documents:
---
{rag_context}
---

User Question: {user_query}

Answer:"""
medical_info_prompt = PromptTemplate(
    template=medical_info_template,
    input_variables=["rag_context", "user_query"]
)

# --- Final Specialist Recommendation (Updated for potential image) ---
final_specialist_template = """
You are a knowledgeable medical assistant AI. Your task is to synthesize the provided information and recommend a medical specialist.

Accumulated Patient Text Symptoms:
{accumulated_symptoms}

Relevant Context from Medical Knowledge Base (Text-based):
{rag_context}

Potentially Relevant ICD10 Codes (Informational Only, Text-based):
{matched_icd_codes}

User-Provided Image (Visual Context, if available):
[Image is provided as input, analyze its content if present]

Based *strictly* on the Accumulated Patient Text Symptoms, the Relevant Context, the potentially relevant ICD codes, AND the user-provided Image (if available), determine the most appropriate type of medical specialist for the patient to consult.
Provide *only* the specialist category name (e.g., Cardiologist, Dermatologist, Neurologist, Primary Care Physician) followed by a brief, 1-2 sentence explanation. This explanation *must be directly derived* from the provided 'Relevant Context' or sensible interpretation of the symptoms/image, stating *why* that specialist relates to the combined information.
If the overall information (text + image) is clearly insufficient or contradictory, recommend consulting a 'Primary Care Physician' first for a general evaluation.
Do NOT provide any medical advice, diagnosis, treatment suggestions, or contact information. Focus solely on the specialist recommendation and its justification.

Recommendation:"""
final_specialist_prompt = PromptTemplate(
    template=final_specialist_template,
    input_variables=["accumulated_symptoms", "rag_context", "matched_icd_codes"]
    # Image data passed directly in the multimodal message content
)

# --- Explanation Clarity Evaluator (Remains Text-Based) ---
explanation_evaluator_template = """
You are an AI evaluating another AI's explanation for recommending a medical specialist. Your goal is to assess if the explanation is clear, simple, and avoids unexplained jargon for a general audience (layperson).

Explanation to Evaluate:
---
{initial_explanation}
---

Evaluation Criteria:
1.  **Clarity & Simplicity:** Is the language easy to understand? Are sentences reasonably short and direct?
2.  **Jargon:** Are there medical terms used that a layperson likely wouldn't know? (e.g., 'idiopathic', 'bradycardia', 'stenosis', specific test names without context).

Evaluation Task:
Review the explanation.
- If it meets the criteria (clear, simple, no unexplained jargon), respond with only the word 'OK'.
- If it needs improvement, respond with 'REVISE' followed by a brief, specific critique pointing out the jargon or unclear parts. Focus only on clarity and jargon, not medical accuracy.

Examples of Output:
OK
REVISE - Uses the term 'paresthesia' without explanation.
REVISE - Sentence structure is complex. The term 'etiology' is jargon.

Your Evaluation:"""
explanation_evaluator_prompt = PromptTemplate(
    template=explanation_evaluator_template,
    input_variables=["initial_explanation"]
)

# --- Explanation Refiner (Remains Text-Based) ---
explanation_refiner_template = """
You are an AI assistant rewriting a medical specialist recommendation explanation to improve its clarity for a layperson, based on specific feedback.

Original Explanation:
---
{initial_explanation}
---

Critique / Areas to Improve:
---
{evaluator_critique}
---

Rewrite the 'Original Explanation' to directly address the 'Critique'. Your goals are:
1.  Replace or provide simple explanations for any specific jargon terms mentioned in the critique.
2.  Simplify sentence structures identified as complex.
3.  Ensure the core reason for the specialist recommendation remains accurate and consistent with the original meaning.
4.  Do NOT add any new medical information, advice, or opinions.

Output only the rewritten explanation.

Rewritten Explanation:"""
explanation_refiner_prompt = PromptTemplate(
    template=explanation_refiner_template,
    input_variables=["initial_explanation", "evaluator_critique"]
)

# --- Basic Deflection Messages ---
# Defined as constants for direct use in nodes
off_topic_response = "I am designed to assist with preliminary medical information and symptom discussion. I can't help with topics outside of that scope. Do you have any health-related questions or symptoms you'd like to discuss?"
irrelevant_triage_response = "Based on our conversation, it doesn't seem like we focused on specific medical symptoms for triage. If you have health concerns you'd like assistance understanding or finding potential specialist types for, please describe them clearly. For any medical advice or diagnosis, please consult a qualified healthcare professional."

rag_relevance_evaluator_template = """
Evaluate whether the following retrieved context is likely sufficient and relevant to directly answer the user's specific question. Focus on direct answerability, not just topic overlap.

Retrieved Context:
---
{rag_context}
---

User Question: {user_query}

Is the context directly relevant and likely sufficient to answer the specific question? Respond with only the word 'YES' or 'NO'.
Relevant & Sufficient:"""
rag_relevance_evaluator_prompt = PromptTemplate(
    template=rag_relevance_evaluator_template,
    input_variables=["rag_context", "user_query"]
)