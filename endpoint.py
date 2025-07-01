from fastapi import FastAPI, Request
from utils_copy10 import run_graph_with_tracing
from pydantic import BaseModel
from uuid import uuid4

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/run-agent")
async def run_agent(request: PromptRequest):
    try:
        # Extract the prompt from the request body
        input = {"prompt":request.prompt, "id": str(uuid4())}
        
        # Run the graph with tracing using the provided prompt
        result = run_graph_with_tracing(input)
        
        # Return the result as a JSON response
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}