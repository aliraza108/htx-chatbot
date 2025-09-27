from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from agents import Agent, FileSearchTool, set_tracing_disabled

# --- API Key setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
set_tracing_disabled(True)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "App running. Use /chat or /debug"}

# --- Debug endpoint (direct OpenAI, no tools) ---
@app.get("/debug")
async def debug():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say the name of your planet in one word."}],
        )
        return {
            "python": {
                "version": os.sys.version,
                "platform": os.uname().sysname,
            },
            "env": {"OPENAI_API_KEY": {"present": OPENAI_API_KEY is not None}},
            "runner_test": {"ok": True, "final_output": response.choices[0].message.content},
        }
    except Exception as e:
        return {"runner_test": {"ok": False, "error": str(e)}}

# --- Chat endpoint (still can wire into agents if you like) ---
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")

    # Direct OpenAI call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
    )

    return JSONResponse({"reply": response.choices[0].message.content})
