# chat_api.py
import re
import json
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

# --- Dataclass Schema ---
@dataclass
class Product_output:
    Product_Title: str
    Product_Price: int
    Product_Link: str
    Product_Image: str
    Product_description: str
    Product_qty: str

# --- Agents ---
Product_Agent = Agent(
    name="Product Agent",
    instructions="""
You are the Product Agent.
Return ONLY a JSON array of product objects, each matching this schema:
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
If no product found, return [].
""",
    tools=[FileSearchTool(max_num_results=30,
                          vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"])],
    model="gpt-4o-mini",
    output_type=list[Product_output],
)

StoreInfoAgent = Agent(
    name="Store Info Agent",
    instructions="""
You are the Store Info Agent.
Answer ONLY using store data (policies, contact, shipping, etc.).
If info missing, say it is unavailable.
""",
    tools=[FileSearchTool(max_num_results=30,
                          vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"])],
    model="gpt-4o-mini",
)

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
    handoffs=[Product_Agent, StoreInfoAgent],
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

# --- Helpers ---
def parse_price(s: Any) -> float | int:
    if not s:
        return 0
    m = re.search(r"(\d+(?:[\.,]\d+)?)", str(s))
    if not m:
        return 0
    val = m.group(1).replace(",", ".")
    return float(val) if "." in val else int(val)
def parse_markdown_products(text: str):
    """
    Extract products from markdown-like output.
    Example:
    1. **Title** - **Price**: $9.99 - **Description**: ... - **Link**: [View Product](URL) ![Image](URL)
    """
    products = []
    if not text:
        return products

    # Split by numbered list items (1., 2., etc.)
    parts = re.split(r"\n?\s*\d+\.\s*", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        title = re.search(r"\*\*(.*?)\*\*", part)
        price = re.search(r"\*\*Price\*\*[:\s-]*\$?([\d\.,]+)", part) or re.search(r"Price[:\s-]*\$?([\d\.,]+)", part)
        desc = re.search(r"\*\*Description\*\*[:\s-]*(.*?)(?: - \*\*| - \[|$)", part)
        link = re.search(r"\[View Product\]\((https?:\/\/[^\)\s]+)\)", part)
        image = re.search(r"!\[.*?\]\((https?:\/\/[^\)\s]+)\)", part)

        products.append({
            "title": title.group(1) if title else "",
            "price": float(price.group(1).replace(",", "")) if price else 0,
            "description": desc.group(1).strip() if desc else "",
            "link": link.group(1) if link else "",
            "image": image.group(1) if image else "",
            "qty": ""
        })
    return products


def normalize_item(item: Any) -> dict | None:
    """Map raw agent product output → frontend schema"""
    if hasattr(item, "__dict__"):
        item = item.__dict__
    if not isinstance(item, dict):
        return None
    return {
        "title": item.get("Product_Title") or item.get("title") or "",
        "price": parse_price(item.get("Product_Price") or item.get("price") or 0),
        "link": item.get("Product_Link") or item.get("link") or "",
        "image": item.get("Product_Image") or item.get("image") or "",
        "description": item.get("Product_description") or item.get("description") or "",
        "qty": item.get("Product_qty") or item.get("qty") or "",
    }

# --- Chat Endpoint ---
from fastapi import Request
from fastapi.responses import JSONResponse
import asyncio
from typing import List

# Example simple FAQ shortcuts
FAQS = {
    "hi": "👋 Hi there! How can I help you today?",
    "hello": "👋 Hello! Ask me about hemp products, pricing, or delivery.",
    "what is your name": "🤖 I'm HTX Chatbot, your product assistant.",
    "delivery charges": "🚚 Delivery is FREE for orders over $50. Otherwise $4.99."
}

def normalize_item(item):
    """Normalize product object into dict for cards"""
    try:
        return {
            "title": getattr(item, "Product_Title", None) or item.get("Product_Title"),
            "price": getattr(item, "Product_Price", None) or item.get("Product_Price"),
            "description": getattr(item, "Product_description", None) or item.get("Product_description"),
            "image": getattr(item, "Product_Image", None) or item.get("Product_Image"),
            "link": getattr(item, "Product_Link", None) or item.get("Product_Link"),
            "qty": getattr(item, "Product_qty", None) or item.get("Product_qty"),
        }
    except Exception:
        return None


@app.post("/chat")
async def chat(req: Request):
    """
    Input:  { "query": "..." }
    Output: { "products": [...], "reply": "..." }
    """
    body = await req.json()
    query = (body.get("query") or body.get("message") or "").strip().lower()

    try:
        # --- Quick FAQ handling (no LLM) ---
        if query in FAQS:
            return JSONResponse({
                "products": [],
                "reply": FAQS[query]
            })

        # --- If asking for products ---
        if "product" in query or "hemp" in query:
            try:
                result = await asyncio.wait_for(
                    Runner.run(Triage_Agent, input=query),
                    timeout=4  # max 4s wait
                )
                final_output = result.final_output
            except asyncio.TimeoutError:
                return JSONResponse({
                    "products": [],
                    "reply": "⚡ Sorry, request took too long. Please try again."
                })

            products: List[dict] = []
            reply_text = ""

            # Case: structured list
            if isinstance(final_output, list):
                for it in final_output:
                    n = normalize_item(it)
                    if n:
                        products.append(n)

            # Case: dict / dataclass
            elif isinstance(final_output, dict) or hasattr(final_output, "__dict__"):
                n = normalize_item(final_output)
                if n:
                    products.append(n)

            # Case: plain text
            elif isinstance(final_output, str):
                reply_text = final_output

            return JSONResponse({
                "products": products,
                "reply": reply_text
            })

        # --- Default fallback (fast text only) ---
        return JSONResponse({
            "products": [],
            "reply": "🤔 I didn’t understand that. Try asking about hemp products or delivery."
        })

    except Exception as e:
        return JSONResponse({
            "products": [],
            "reply": "⚠️ Agent error, please try again.",
            "error": str(e),
        }, status_code=500)

