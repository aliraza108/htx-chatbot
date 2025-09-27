# chat_api.py
import re
import os
from dataclasses import dataclass, asdict
from typing import Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import function_tool, Runner, Agent, set_tracing_disabled, FileSearchTool

# --- API Key ---
part1 = 'sk-proj'
part2 = '-Mfz8dQmnuGhRCYwOE419Eno6ayP0yD2gTgoOLZz3uW6DFvIhO'
part3 = '--LuMyTVIVl1P8_OS4_4b-FtJT3BlbkFJ5yZqmw_sp8QdqC2GcD1gCKPyvdzf_MLCv1xaT8T_oY8wPdliX5c10uvhFRRJD1MJTJJMTqk4UA'
OPENAI_API_KEY = part1 + part2 + part3
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

set_tracing_disabled(True)

Triage_Agent = Agent(
    name="Triage Agent",
    instructions="""
You are the Triage Agent.
- If query is about products → hand off to Product_Agent
- If about store info → hand off to StoreInfoAgent
- If both → delegate and combine results
""",
    tools=[FileSearchTool(max_num_results=30,
                          vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"])],
    
    model="gpt-4o-mini",
)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    """
    Input:  { "query": "..." }
    Output: { "products": [...], "reply": "..." }
    """
    body = await req.json()
    query = (body.get("query") or body.get("message") or "").strip()
    try:
        result = await Runner.run(Triage_Agent, input=query)
        return result.final_output
    except Exception as e:
        print("Agent error:", e)   # ✅ shows in Vercel logs
        return JSONResponse({
            "reply": "⚠ Agent error, please try again.",
            "error": str(e),
        }, status_code=500)
