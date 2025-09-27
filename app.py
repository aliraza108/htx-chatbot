# chat_api.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai

# --- API Key ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set in Vercel dashboard
openai.api_key = OPENAI_API_KEY

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to Shopify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    query = (body.get("query") or body.get("message") or "").strip()
    if not query:
        return {"reply": "⚠ No query provided"}

    try:
        # Call OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=500
        )
        final_output = response.choices[0].message.content
        return {"reply": final_output}

    except Exception as e:
        return JSONResponse({
            "reply": "⚠ Agent error, please try again.",
            "error": str(e),
        }, status_code=500)
