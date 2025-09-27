# chat_api.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agents import Runner, Agent, set_tracing_disabled, FileSearchTool

# --- API Key ---
part1 = 'sk-proj'
part2 = '-Mfz8dQmnuGhRCYwOE419Eno6ayP0yD2gTgoOLZz3uW6DFvIhO'
part3 = '--LuMyTVIVl1P8_OS4_4b-FtJT3BlbkFJ5yZqmw_sp8QdqC2GcD1gCKPyvdzf_MLCv1xaT8T_oY8wPdliX5c10uvhFRRJD1MJTJJMTqk4UA'
OPENAI_API_KEY = part1 + part2 + part3
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Disable tracing for speed
set_tracing_disabled(True)

# --- Agent Setup ---
Triage_Agent = Agent(
    name="Triage Agent",
    instructions="""
You are the Triage Agent.
- Handle queries about products or store info.
- Return a complete answer in plain text.
""",
    tools=[FileSearchTool(
        max_num_results=30,
        vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"]
    )],
    model="gpt-4o-mini",
)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Shopify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(req: Request):
    """
    Input:  { "query": "..." }
    Output: { "reply": final_output }
    """
    body = await req.json()
    query = (body.get("query") or body.get("message") or "").strip()

    try:
        # Run the agent
        result = await Runner.run(Triage_Agent, input=query)
        final_output = result.final_output

        # Return raw output
        return {"reply": final_output}

    except Exception as e:
        return JSONResponse({
            "reply": "⚠ Agent error, please try again.",
            "error": str(e),
        }, status_code=500)
