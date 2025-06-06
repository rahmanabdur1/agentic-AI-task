# multi_agent_sports_qa/main.py

from fastapi import FastAPI, HTTPException, Request # Import Request for templates
from fastapi.responses import HTMLResponse # For returning HTML directly
from fastapi.staticfiles import StaticFiles # To serve CSS/JS
from fastapi.templating import Jinja2Templates # For HTML templating (though simple in this case)
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables from .env file at the very beginning of the application startup.
load_dotenv()

# Import your agent system. The singleton instance will be created and initialized
# automatically when this module is imported.
from agent_system import agent_system_instance

# Initialize the FastAPI application
app = FastAPI(
    title="AI Multi-Agent Sports PDF Q&A System",
    description="An API to answer questions, summarize documents, and provide citations using a LangChain multi-agent system, powered by your sports-related PDF documents, now with a web UI!",
    version="1.0.0"
)

# Mount the static files directory. Files in 'static' can be accessed via /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2Templates to load templates from the 'static' directory
# This allows FastAPI to render HTML files.
templates = Jinja2Templates(directory="static")

# Pydantic model for validating the incoming request body for the API endpoint
class QueryRequest(BaseModel):
    query: str

# Pydantic model for validating the outgoing response body
class QueryResponse(BaseModel):
    query: str
    answer: str
    summary: str
    citations: List[str]

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event. This function runs once when the application starts.
    It simply confirms that the AgentSystem has been initialized (due to the singleton pattern).
    """
    print("FastAPI app starting up: AgentSystem is ready.")

# Route to serve the main HTML page at the root URL ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main web UI for the Q&A system.
    """
    # Renders the index.html template. The 'request' object must be passed to the template.
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processes a user query through the multi-agent system, using PDF documents.
    
    Args:
        request (QueryRequest): The incoming request containing the user's query.

    Returns:
        QueryResponse: The structured response with answer, summary, and citations.

    Raises:
        HTTPException: If an internal server error occurs during processing.
    """
    try:
        print(f"\n--- API Request Received: Query: '{request.query}' ---")
        
        # Invoke the LangChain agent system with the user's query.
        response = await agent_system_instance.query_agent(request.query)

        # Return the structured response, ensuring all fields are present.
        return QueryResponse(
            query=request.query,
            answer=response.get("answer", "No answer found."),
            summary=response.get("summary", "No summary provided."),
            citations=response.get("citations", [])
        )
    except Exception as e:
        # Catch any unexpected errors during processing and return a 500 Internal Server Error.
        print(f"API Error during query processing: {e}")
        # Return a more generic error to the frontend, log full error on server
        raise HTTPException(status_code=500, detail=f"An internal server error occurred. Please check server logs for more details.")