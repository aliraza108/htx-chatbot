# app.py

import os
import re
import json
import logging
from typing import Any, List
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import (
    Runner,
    Agent,
    set_tracing_disabled,
    FileSearchTool,
)

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# API Key
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

set_tracing_disabled(True)

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class ProductOutput:
    Product_Title: str
    Product_Price: int
    Product_Link: str
    Product_Image: str
    Product_description: str
    Product_qty: str

# -----------------------------
# Agents
# -----------------------------
Product_Agent = Agent(
    name="Product Agent",
    instructions="Answer product-related queries only.",
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    model="gpt-4o-mini",
    output_type=List[ProductOutput],
)

StoreInfo_Agent = Agent(
    name="Store Info Agent",
    instructions="Answer store-related queries only.",
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    model="gpt-4o-mini",
)

Triage_Agent = Agent(
    name="Triage Agent",
    instructions="Route queries to the correct agent.",
    handoffs=[Product_Agent, StoreInfo_Agent],
    model="gpt-4o-mini",
)

runner = Runner(Triage_Agent)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utils
# -----------------------------
def parse_price(s: Any):
    if s is None:
        return 0
    m = re.search(r"(\d+(?:[\.,]\d+)?)", str(s))
    if not m:
        return 0
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return 0

def normalize_item(item: Any):
    if hasattr(item, "__dict__"):
        item = item.__dict__
    if not isinstance(item, dict):
        return None
    return {
        "title": item.get("Product_Title") or "",
        "price": parse_price(item.get("Product_Price")),
        "link": item.get("Product_Link") or "",
        "image": item.get("Product_Image") or "",
        "description": item.get("Product_description") or "",
        "qty": item.get("Product_qty") or "",
    }

def serialize_output(output: Any):
    if isinstance(output, list):
        return [normalize_item(i) for i in output if normalize_item(i)]
    return [normalize_item(output)] if normalize_item(output) else []

# -----------------------------
# Routes
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("query") or data.get("message")
    if not user_message:
        return JSONResponse({"error": "query/message is required"}, status_code=400)

    try:
        logger.info(f"User message: {user_message}")
        result = await runner.run(user_message)  # ensure async
        logger.info(f"Raw result: {result}")
        products = serialize_output(result)
        return JSONResponse({"products": products})
    except Exception as e:
        logger.exception("Error while processing request")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "FastAPI chatbot on Vercel is running 🚀"}
