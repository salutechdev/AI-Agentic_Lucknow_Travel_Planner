import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent_logic import get_agent_executor
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Lucknow Tour Guide API",
    description="API for the AI-powered Lucknow travel planner.",
    version="1.0.0"
)

# Initialize the agent once when the server starts
agent_executor = get_agent_executor()

class QueryRequest(BaseModel):
    """Request model synchronized with the stateful memory layer."""
    query: str = Field(..., min_length=2, max_length=500)
    session_id: str = Field(..., description="Unique tracking identifier for the user chat session.")

class QueryResponse(BaseModel):
    """Response model for the agent's answer."""
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Receives user query, attaches session context to historical state, 
    tracks processing performance, and returns the response.
    """
    if not agent_executor:
        return {"response": "Agent not initialized. Please check server logs."}
    
    start_time = time.time()
    
    try:
        # Build the runtime configuration context required by RunnableWithMessageHistory
        runtime_config = {"configurable": {"session_id": request.session_id}}
        
        # Invoke the stateful agent with inputs and session configurations
        result = agent_executor.invoke(
            {"input": request.query},
            config=runtime_config
        )
        
        duration = time.time() - start_time
        logger.info(f"Query for session {request.session_id} processed successfully in {duration:.2f}s")
        
        return {"response": result.get("output", "Sorry, I couldn't process that.")}
    except Exception as e:
        logger.error(f"Error during agent invocation for session {request.session_id}: {e}")
        return {"response": "An internal error occurred while processing your request."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
