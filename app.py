# chat_api.py

import os
import json
from typing import Any, List
from dataclasses import dataclass, asdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import (
    function_tool,
    FileSearchTool,
    Runner,
    Agent,
    set_tracing_disabled,
    set_default_openai_api,
    set_default_openai_client,
    AsyncOpenAI,
)

# =====================
# API Key Setup
# =====================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env variable")

# Configure OpenAI client
client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPENAI_API_KEY,
)
set_default_openai_api("chat_completions")
set_default_openai_client(client=client)
set_tracing_disabled(True)

# =====================
# Output Schemas
# =====================
@dataclass
class ProductOutput:
    Product_Title: str
    Product_Price: int
    Product_Link: str
    Product_Image: str
    Product_description: str
    Product_qty: str

@dataclass
class StoreInfoOutput:
    Store_name: str
    Store_description: str
    Store_address: str
    Store_domain: str
    Store_logo: str
    Store_products: List[str]

# =====================
# Agents
# =====================
Product_Agent = Agent(
    name="Product Agent",
    instructions="You are responsible for product-related queries. Only use the FileSearchTool to fetch product information.",
    tools=[
        FileSearchTool(
            name="Product Search Tool",
            path="product.json",
            instructions="Fetch product data based on user query.",
            output_type=List[ProductOutput],
        )
    ],
)

StoreInfo_Agent = Agent(
    name="StoreInfo Agent",
    instructions="You handle store information queries. Use the FileSearchTool to fetch store details.",
    tools=[
        FileSearchTool(
            name="Store Info Tool",
            path="storeinfo.json",
            instructions="Fetch store info (name, description, address, etc).",
            output_type=List[StoreInfoOutput],
        )
    ],
)

Triage_Agent = Agent(
    name="Triage Agent",
    instructions="Decide whether the query is about products or store info. If about products, hand off to Product Agent. If about store info, hand off to StoreInfo Agent. Otherwise, reply that you can only provide product or store details.",
    handoffs=[Product_Agent, StoreInfo_Agent],
)

# Runner setup
runner = Runner(Triage_Agent)

# =====================
# FastAPI Setup
# =====================
app = FastAPI()

# Allow CORS (useful for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Utility Functions
# =====================
def normalize_item(item: Any):
    """Convert agent output into JSON-serializable dicts."""
    if hasattr(item, "__dict__"):
        item = item.__dict__
    if not isinstance(item, dict):
        return None
    return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in item.items()}

def serialize_output(output: Any):
    """Normalize runner output for JSON response."""
    if isinstance(output, list):
        return [normalize_item(i) for i in output if normalize_item(i) is not None]
    return normalize_item(output)

# =====================
# Routes
# =====================
@app.post("/chat")
async def chat(request: Request):
    """Main chatbot endpoint."""
    data = await request.json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    try:
        # Run the triage agent
        result = await runner.run(user_message)
        response_data = serialize_output(result)
        return JSONResponse(content={"response": response_data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    return {"message": "FastAPI Chatbot with Multi-Agent Routing is running 🚀"}
