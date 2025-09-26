
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware (only once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body model
class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    This endpoint receives a message from the frontend, runs the agent,
    and returns the agent's full response.
    """

    return "reply"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






