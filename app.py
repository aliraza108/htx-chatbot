# chat_api.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agents import Runner, Agent, set_tracing_disabled, FileSearchTool

# --- API Key ---
os.environ["OPENAI_API_KEY"] = "sk-YOUR_KEY_HERE"

set_tracing_disabled(True)

# --- Agents ---
Triage_Agent = Agent(
    name="Triage Agent",
    instructions="""
You are the Agent.
""",
    tools=[FileSearchTool(max_num_results=10,
                          vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"])],
    model="gpt-4o-mini",
)

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    try:
        body = await req.json()
        query = (body.get("query") or body.get("message") or "").strip()
        if not query:
            return JSONResponse({"reply": "❌ Empty query"}, status_code=400)

        # Run agent
        result = await Runner.run(Triage_Agent, input=query)
        final_output = result.final_output

        # Return as JSON
        return JSONResponse({
            "reply": final_output
        })

    except Exception as e:
        return JSONResponse({
            "reply": "⚠ Agent error, please try again.",
            "error": str(e)
        }, status_code=500)
