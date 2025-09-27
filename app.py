import os
import re
import json
from typing import Any, List
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import (
    function_tool,
    Runner,
    Agent,
    set_tracing_disabled,
    FileSearchTool,
)

# ==========================================
# OpenAI API Key from Vercel environment
# ==========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env variable")

set_tracing_disabled(True)

# ==========================================
# Dataclasses
# ==========================================
@dataclass
class ProductOutput:
    Product_Title: str
    Product_Price: int
    Product_Link: str
    Product_Image: str
    Product_description: str
    Product_qty: str

# ==========================================
# Agents
# ==========================================
Product_Agent = Agent(
    name="Product Agent",
    instructions="""
You are the Product Agent.

Rules:
- Only return a JSON array of product objects.
- Each product must strictly follow this schema:
[
  {
    "Product_Title": "string",
    "Product_Price": 0,
    "Product_Link": "string",
    "Product_Image": "string",
    "Product_description": "string",
    "Product_qty": "string"
  }
]
- No markdown, no text commentary.
- If nothing is found, return [].
""",
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
    instructions="""
You are the Store Info Agent.

Answer only store-related queries using the FileSearchTool.
Do not invent details.
""",
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
    instructions="""
Route queries:
- If product-related → Product Agent
- If store info-related → Store Info Agent
""",
    handoffs=[Product_Agent, StoreInfo_Agent],
    model="gpt-4o-mini",
)

runner = Runner(Triage_Agent)

# ==========================================
# FastAPI setup
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Utils
# ==========================================
def parse_price(s: Any):
    if s is None:
        return 0
    m = re.search(r"(\d+(?:[\.,]\d+)?)", str(s))
    if not m:
        return 0
    val = m.group(1).replace(",", ".")
    try:
        return float(val) if "." in val else int(val)
    except:
        return 0

def normalize_item(item: Any):
    if hasattr(item, "__dict__"):
        item = item.__dict__
    if not isinstance(item, dict):
        return None
    return {
        "title": item.get("Product_Title") or item.get("title") or "",
        "price": parse_price(item.get("Product_Price") or item.get("price")),
        "link": item.get("Product_Link") or item.get("link") or "",
        "image": item.get("Product_Image") or item.get("image") or "",
        "description": item.get("Product_description") or item.get("description") or "",
        "qty": item.get("Product_qty") or item.get("qty") or "",
    }

def serialize_output(output: Any):
    if isinstance(output, list):
        return [normalize_item(i) for i in output if normalize_item(i)]
    return [normalize_item(output)] if normalize_item(output) else []
    
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("query") or data.get("message")
    if not user_message:
        return JSONResponse({"error": "query/message is required"}, status_code=400)

    try:
        result = await runner.run(user_message)
        products = serialize_output(result)
        return JSONResponse({"products": products})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "FastAPI multi-agent chatbot running 🚀"}
