import os
import asyncio
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

set_tracing_disabled(True)

# --- Simple Triage Agent ---
Triage_Agent = Agent(
    name="Triage Agent",
    instructions="""
You are the Triage Agent.
Answer questions about products or store info directly.
Return plain text for normal queries.
""",
    tools=[FileSearchTool(max_num_results=30, vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"])],
    model="gpt-4o-mini",
)

# --- FastAPI app ---
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
    """
    Input:  { "query": "..." }
    Output: { "reply": "..." }
    """
    body = await req.json()
    query = (body.get("query") or body.get("message") or "").strip()

    try:
        # Run agent with a max timeout
        result = await asyncio.wait_for(Runner.run(Triage_Agent, input=query), timeout=15)
        output = result.final_output

        # Make sure output is string
        if not isinstance(output, str):
            output = str(output)

        return JSONResponse({"reply": output})

    except asyncio.TimeoutError:
        return JSONResponse({"reply": "⏱ Request took too long, please try again."}, status_code=504)
    except Exception as e:
        return JSONResponse({"reply": "⚠ Agent error, please try again.", "error": str(e)}, status_code=500)
