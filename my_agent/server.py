from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from agent import root_agent

app = FastAPI(title="Aura Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Depending on the ADK version, it might be run() or chat() or __call__
        if hasattr(root_agent, "chat"):
            response = root_agent.chat(req.message)
        elif hasattr(root_agent, "run"):
            response = root_agent.run(req.message)
        else:
            response = root_agent(req.message)
            
        # Ensure it's a string
        if not isinstance(response, str):
            if hasattr(response, "text"):
                response = response.text
            elif hasattr(response, "content"):
                response = response.content
            else:
                response = str(response)

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
