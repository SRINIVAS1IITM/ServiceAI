import os
import datetime
import json
import requests # Make sure this is installed (pip install requests) and in requirements.txt
from typing import Optional, Dict, Tuple # Ensure Optional and Dict are imported from typing

# Comment out or remove OpenAI related imports and .env loading for OpenAI
# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

from .kb_manager import kb_manager
from .models import KBArticle

# --- Ollama Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat" # Ollama's chat endpoint
OLLAMA_MODEL_NAME = "phi3:mini" # Our chosen efficient model

FEATURE_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'feature_requests.log')

# --- LLM Helper ---
def get_llm_response(prompt: str, system_message: Optional[str] = None) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1 # Lower temperature for more deterministic classification/extraction
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60) # Added timeout
        response.raise_for_status()
        
        response_data = response.json()
        
        if 'message' in response_data and 'content' in response_data['message']:
            return response_data['message']['content'].strip()
        else:
            print(f"Unexpected response structure from Ollama: {response_data}")
            return "Error: Could not parse response from local LLM."

    except requests.exceptions.Timeout:
        print(f"Error: Timeout calling local Ollama API for model {OLLAMA_MODEL_NAME}.")
        return "Error: Local LLM request timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling local Ollama API: {e}")
        if e.response is not None:
            print(f"Ollama API Response Content: {e.response.text}")
        return "Error: Could not get response from local LLM."
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Ollama: {e}")
        print(f"Ollama API Raw Response: {response.text if 'response' in locals() else 'No response object'}")
        return "Error: Could not parse JSON response from local LLM."
    except Exception as e:
        print(f"An unexpected error occurred in get_llm_response: {e}")
        return "Error: Unexpected issue with local LLM interaction."

# --- Agent Functions ---

def classify_intent(query: str) -> str:
    system_message = (
        "You are an expert intent classifier. "
        "Classify the user's query into ONE of the following categories: "
        "'Technical Support', 'Product Feature Request', 'Sales Lead', or 'General Inquiry'. "
        "Respond with ONLY the category name. For example, if the query is 'How do I reset my password?', respond: Technical Support"
    )
    prompt = f"Customer query: \"{query}\"\n\nCategory:" # Changed "Classification" to "Category" for clarity
    intent_raw = get_llm_response(prompt, system_message)
    
    valid_intents = ["Technical Support", "Product Feature Request", "Sales Lead", "General Inquiry"]
    
    # More robust parsing for potentially verbose LLM outputs
    for valid_intent in valid_intents:
        if valid_intent.lower() in intent_raw.lower():
            # Check if it's a reasonably direct match, not just a substring in a long sentence
            if len(intent_raw) < len(valid_intent) + 30: # Allow some flexibility
                 return valid_intent
    
    # If the raw response itself is one of the valid intents (case-insensitive)
    if intent_raw.title() in valid_intents: # .title() to match "Technical Support"
        return intent_raw.title()

    print(f"Warning: Local LLM (phi3:mini-instruct) returned unexpected intent: '{intent_raw}'. Query: '{query}'. Defaulting to 'General Inquiry'.")
    return "General Inquiry"

def extract_feature_details(query: str) -> str:
    system_message = (
        "You are an AI assistant. Read the user's query and extract the core feature being requested. "
        "Summarize the feature concisely in one short phrase or sentence. "
        "If no specific feature is clear, respond with ONLY the words: No specific feature identified"
    )
    prompt = f"User query: \"{query}\"\n\nConcise feature summary:"
    details = get_llm_response(prompt, system_message)

    if "no specific feature identified" in details.lower():
        return "No specific feature identified" # Return the exact phrase
    
    # Phi-3 might still add conversational fluff. Try to clean it.
    # Example: "The user is requesting a dark mode feature." -> "a dark mode feature"
    # This is a simple heuristic, more advanced parsing might be needed for complex cases.
    if "user is requesting" in details.lower():
        details = details.lower().split("user is requesting", 1)[-1].strip()
    if "feature being requested is" in details.lower():
        details = details.lower().split("feature being requested is", 1)[-1].strip()
    if details.startswith("a "): details = details[2:]
    if details.startswith("an "): details = details[3:]

    if not details or len(details) > 150: # If extraction failed or is too long
        print(f"Warning: Feature extraction for '{query}' resulted in: '{details}'. Falling back.")
        return "Could not clearly identify a concise feature summary."
        
    return details.strip().capitalize() # Capitalize the first letter

def log_feature_request(query: str, feature_summary: str):
    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"[{timestamp}] Feature Summary: {feature_summary}\nOriginal Query: {query}\n---\n"
    try:
        with open(FEATURE_LOG_FILE, 'a') as f:
            f.write(log_entry)
        print(f"Logged feature request: {feature_summary}")
    except Exception as e:
        print(f"Error logging feature request: {e}")

def check_missing_sales_info(query: str) -> Optional[str]:
    # This function doesn't use LLM, so no changes needed for model switch.
    if "company" in query.lower() or "organization" in query.lower():
        return None 
    return "Could you please provide your company name?"

def analyze_sentiment(query: str) -> str:
    system_message = (
        "You are a sentiment analysis expert. Classify the sentiment of the user's query "
        "as ONE of the following: 'positive', 'neutral', 'negative', or 'very negative'. "
        "Respond with ONLY the sentiment category. For example, if the query is 'I hate this product', respond: very negative"
    )
    prompt = f"User query: \"{query}\"\n\nSentiment:"
    sentiment_raw = get_llm_response(prompt, system_message)
    
    valid_sentiments = ["positive", "neutral", "negative", "very negative"]

    for valid_sentiment in valid_sentiments:
        if valid_sentiment.lower() in sentiment_raw.lower():
             # Check if it's a reasonably direct match
            if len(sentiment_raw) < len(valid_sentiment) + 20:
                return valid_sentiment.lower()
    
    if sentiment_raw.lower() in valid_sentiments:
        return sentiment_raw.lower()

    print(f"Warning: Local LLM (phi3:mini-instruct) returned unexpected sentiment: '{sentiment_raw}'. Query: '{query}'. Defaulting to 'neutral'.")
    return "neutral"

# --- Main Orchestration Logic ---
def process_customer_inquiry(query: str) -> Tuple[str, str, bool, Optional[str], Optional[KBArticle], Dict]:
    """
    Processes the customer inquiry and returns:
    (response_message, intent, needs_escalation, escalation_reason, kb_article_found, debug_info)
    """
    # This part of the logic largely remains the same as it depends on the outputs of the above functions.
    # Ensure the debug_info captures the raw responses if needed for tuning.
    response_message = "I'm sorry, I'm having trouble understanding your request right now."
    needs_escalation = False
    escalation_reason = None
    kb_article_found: Optional[KBArticle] = None
    debug_info = {'llm_model_used': OLLAMA_MODEL_NAME} # Add model info to debug

    intent = classify_intent(query)
    debug_info['intent_classification_raw_output_from_llm'] = intent # Storing the processed one for now
    debug_info['intent_classification_final'] = intent
    
    sentiment = analyze_sentiment(query)
    debug_info['sentiment_raw_output_from_llm'] = sentiment # Storing processed
    debug_info['sentiment_final'] = sentiment

    if sentiment == "very negative":
        needs_escalation = True
        escalation_reason = "Very negative sentiment detected."

    if intent == "Technical Support":
        kb_article = kb_manager.search_kb(query, top_k=1, threshold=0.7) 
        debug_info['kb_search_query'] = query
        if kb_article:
            kb_article_found = kb_article
            topic_for_response = kb_article.topic
            response_message = (
                f"Thanks for reaching out! Regarding your question about '{topic_for_response}', "
                f"here's some information: {kb_article.answer}\n\n"
                "Does this resolve your issue?"
            )
            if not needs_escalation:
                needs_escalation = False 
                escalation_reason = "Awaiting user confirmation on KB solution."
            debug_info['kb_article_found'] = kb_article.model_dump()
        else:
            response_message = (
                "Thanks for your query. I couldn't find an immediate answer in our knowledge base, "
                "but I've routed your request to our Technical Support team. "
                "They will get back to you shortly."
            )
            if not needs_escalation:
                needs_escalation = True
                escalation_reason = "KB search failed for technical support query."
            debug_info['kb_search_result'] = "No relevant article found or score too low."

    elif intent == "Product Feature Request":
        feature_summary = extract_feature_details(query)
        debug_info['extracted_feature_raw_output_from_llm'] = feature_summary # Storing processed
        debug_info['extracted_feature_final'] = feature_summary

        if "no specific feature identified" not in feature_summary.lower() and \
           "could not clearly identify" not in feature_summary.lower() and \
           len(feature_summary) > 3 : # Basic check if a feature was actually extracted
            log_feature_request(query, feature_summary)
            response_message = (
                f"Thank you for your suggestion! We've logged your feature request for: "
                f"'{feature_summary}' for our product team to review."
            )
        else:
            response_message = (
                "Thanks for your feedback! Could you please provide more details about the feature you'd like to see, or describe it differently?"
            )
        if not needs_escalation:
            needs_escalation = False 

    elif intent == "Sales Lead":
        response_message = (
            "Thanks for your interest in our products/services! Our sales team will be in touch soon. "
        )
        missing_info_question = check_missing_sales_info(query)
        if missing_info_question:
            response_message += f"In the meantime, {missing_info_question.lower()} "
            if not needs_escalation:
                needs_escalation = False
                escalation_reason = "Awaiting more information from potential sales lead."
        else:
            response_message += (
                "In the meantime, could you tell us more about your needs? For example, "
                "how many team members do you have, or what specific challenges are you trying to solve?"
            )
            if not needs_escalation:
                needs_escalation = True
                escalation_reason = "Sales lead identified and initial info gathered."
        debug_info['sales_info_check'] = "Company name likely present" if not missing_info_question else "Company name likely missing"

    elif intent == "General Inquiry":
        response_message = (
            "Thanks for your query. I'm not sure how to help with that directly. "
            "I've routed your request to our general support team. "
            "They will get back to you shortly."
        )
        if not needs_escalation:
            needs_escalation = True
            escalation_reason = "General inquiry requiring human review."
    
    if needs_escalation and not escalation_reason:
        escalation_reason = "Escalation triggered by unhandled condition or complex query."

    return response_message, intent, needs_escalation, escalation_reason, kb_article_found, debug_info