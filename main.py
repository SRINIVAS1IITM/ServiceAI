from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QueryRequest, AgentResponse, KBArticle
from .agent_logic import process_customer_inquiry
from .kb_manager import kb_manager # To ensure it's initialized on startup

app = FastAPI(title="Customer Support Agent API")

# CORS Middleware (allow frontend to call)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust if your React app runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # This ensures kb_manager is initialized when the app starts
    # The import itself might do it, but this is more explicit.
    if kb_manager.index is None:
        print("KB Manager index was not initialized prior to startup event. Forcing init.")
        kb_manager._load_kb() # Ensure KB data is loaded
        kb_manager._build_index() # Ensure index is built
    print("Customer Support Agent API started.")
    if kb_manager.index:
        print(f"KB Manager ready with {kb_manager.index.ntotal} indexed documents.")
    else:
        print("KB Manager index is not available.")


@app.post("/chat", response_model=AgentResponse)
async def handle_chat(request: QueryRequest):
    try:
        user_query = request.query
        if not user_query or not user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        (
            response_message, 
            intent, 
            needs_escalation, 
            escalation_reason, 
            kb_article_found_data,
            debug_info
        ) = process_customer_inquiry(user_query)
        
        kb_article_response: KBArticle | None = None
        if kb_article_found_data:
            kb_article_response = KBArticle(
                id=kb_article_found_data.id,
                topic=kb_article_found_data.topic,
                answer=kb_article_found_data.answer,
                score=kb_article_found_data.score
            )

        return AgentResponse(
            message=response_message,
            intent=intent,
            escalate=needs_escalation,
            escalation_reason=escalation_reason,
            kb_article_found=kb_article_response,
            debug_info=debug_info
        )

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Customer Support Agent API is running."}

# To run: uvicorn backend.app.main:app --reload --port 8000
# (run from the root `customer-support-agent/` directory)