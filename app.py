# app.py

import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    api_key = os.environ.get("OPENAI_API_KEY", "NOT FOUND")
    print("OPENAI_API_KEY:", api_key)  # will show in Vercel logs
    return {"OPENAI_API_KEY": api_key}
