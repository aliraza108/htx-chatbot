
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import re
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dataclasses import asdict
from typing import Any, List
from agents import function_tool, Runner, Agent, set_tracing_disabled, FileSearchTool
import os
from dataclasses import dataclass
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env variable")

set_tracing_disabled(True)
@dataclass
class Product_output:
    Product_Title: str
    Product_Price: int
    Product_Link: str
    Product_Image: str
    Product_description: str
    Product_qty: str
Product_Agent = Agent(
    name="Product Agent",
    instructions="""
You are the Product Agent.

Rules:
- You must ONLY return a JSON array of product objects.
- Each product must strictly follow this JSON schema:
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
- Never return markdown, text, or extra commentary.
- If no product is found, return [].
""",
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    model="gpt-4o-mini",
    output_type=List[Product_output],   # force list of products
)



StoreInfoAgent = Agent(
    name="SEO Agent",
    instructions = """
You are the Store Info Agent.

Your role:
- Provide store-related information strictly using the FileSearchTool (vector store).
- Do not invent details. Only share what exists in the store data.
- You may answer questions about:
  - Store contact details
  - Privacy policies
  - Shipping and return policies
  - Collections
  - Age restrictions
  - General store information

Guidelines:
- If information is missing from the FileSearchTool, state that it is unavailable.
- Be concise and factual, quoting directly from the retrieved content.
"""
,
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    model='gpt-4o-mini')

Triage_Agent = Agent(
    name="SEO Agent",
    instructions = """
You are the Triage Agent.

Your role:
- Act as a task router between agents.
- You have access to the FileSearchTool and can hand off tasks to:
  - Product_Agent → For product-related queries (titles, prices, links, images, descriptions, quantity).
  - StoreInfoAgent → For store information queries (policies, contact, collections, age restrictions, etc.).

Decision-making:
- Analyze the user query carefully.
- If the query is about products, hand off to Product_Agent.
- If the query is about store information, hand off to StoreInfoAgent.
- If the query contains both, delegate accordingly and combine outputs in a clear final response.

Guidelines:
- Never answer directly; always rely on FileSearchTool or delegate to the correct agent.
- Be transparent in responses: indicate which agent handled which part of the query.
- Final output must combine and structure the delegated results so the user sees a unified answer.
"""
,
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    handoffs=[Product_Agent,StoreInfoAgent],
    model='gpt-4o-mini')


app = FastAPI()

# Add CORS middleware (only once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_price(s: str):
    """Extract numeric price from string, return number or 0."""
    if s is None:
        return 0
    m = re.search(r"(\d+(?:[\.,]\d+)?)", str(s))
    if not m:
        return 0
    val = m.group(1).replace(",", ".")
    try:
        # Return float if decimals, else int
        if "." in val:
            return float(val)
        return int(val)
    except:
        try:
            return float(val)
        except:
            return 0


def normalize_item(item: Any):
    """Take dict or dataclass-like and map to normalized product schema."""
    if hasattr(item, "_dict_"):
        item = item._dict_
    if not isinstance(item, dict):
        return None
    return {
        "title": item.get("Product_Title") or item.get("title") or item.get("name") or "",
        "price": parse_price(item.get("Product_Price") or item.get("price") or item.get("Price") or 0),
        "link": item.get("Product_Link") or item.get("Product_Link".lower()) or item.get("link") or item.get("url") or "",
        "image": item.get("Product_Image") or item.get("Product_Image".lower()) or item.get("image") or "",
        "description": item.get("Product_description") or item.get("description") or "",
        "qty": item.get("Product_qty") or item.get("qty") or ""
    }


def parse_markdown_products(text: str):
    """
    Try to extract product entries from markdown-like text similar to:
    1. *TITLE* - *Price: $X - **Description: ... - **Image*: ![](URL) - [View Product](URL)
    Returns list of normalized items
    """
    products = []
    if not text:
        return products

    # Split by numbered headings (1., 2., etc.)
    parts = re.split(r"\n\s*\d+\.\s*", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Title in *Title*
        title = (re.search(r"\\(.?)\\", part) or re.search(r"###\s(.)", part) or {}).group(1) if (re.search(r"\\(.?)\\", part) or re.search(r"###\s*(.*)", part)) else ""
        # Price like *Price: $12.99 or - **Price*: $12.99
        price = (re.search(r"\\*Price\\[:\s-]\$?([\d\.,]+)", part) or re.search(r"Price[:\s-]*\$?([\d\.,]+)", part))
        price_val = parse_price(price.group(1)) if price else 0
        # description after Description:
        desc = (re.search(r"\\*Description\\[:\s-](.?)(?:\n| - \\| - \[|$)", part, re.S) or {}).group(1) if re.search(r"\\Description\\[:\s-]", part) else ""
        # image markdown ![](...)
        img = (re.search(r"!\[.?\]\((https?:\/\/[^\)\s]+)\)", part) or {}).group(1) if re.search(r"!\[.?\]\(", part) else ""
        # link [View Product](...)
        link = (re.search(r"\[View Product\]\((https?:\/\/[^\)\s]+)\)", part) or re.search(r"\((https?:\/\/htxvape\.com[^\)\s]+)\)", part))  
        link_val = link.group(1) if link else ""
        # Fallback title: first line if missing
        if not title:
            first_line = part.splitlines()[0]
            # strip markdown symbols
            title = re.sub(r"[*#\-`\[\]]", "", first_line).strip()

        products.append({
            "title": title,
            "price": price_val,
            "link": link_val,
            "image": img,
            "description": desc.strip(),
            "qty": ""
        })
    return products


async def ensure_structured_with_product_agent(query: str):
    """
    Force-call Product_Agent with a JSON-only instruction and try to parse its output.
    Returns list of normalized products (may be empty).
    """
    # Build a forcing prompt — short and strict
    force_prompt = (
        "RETURN ONLY JSON. You are the Product Agent. For the user's query below, return a JSON array of objects "
        "matching the schema: [{"
        "\"Product_Title\": \"string\", \"Product_Price\": 0, \"Product_Link\": \"string\", "
        "\"Product_Image\": \"string\", \"Product_description\": \"string\", \"Product_qty\": \"string\" }]. "
        "Do NOT add any commentary or markdown. If no results, return [].\n\nQuery: "
        + query
    )
    try:
        # Use Runner to directly run the product agent
        forced_result = await Runner.run(Product_Agent, input=force_prompt)
        forced_out = forced_result.final_output
        # normalize if list/dict/dataclass
        if isinstance(forced_out, list):
            normalized = []
            for it in forced_out:
                n = normalize_item(it)
                if n:
                    normalized.append(n)
            return normalized
        elif hasattr(forced_out, "_dict_"):
            n = normalize_item(forced_out)
            return [n] if n else []
        elif isinstance(forced_out, dict):
            n = normalize_item(forced_out)
            return [n] if n else []
        elif isinstance(forced_out, str):
            # maybe it's JSON string
            try:
                j = json.loads(forced_out)
                if isinstance(j, list):
                    out = []
                    for it in j:
                        out.append({
                            "title": it.get("Product_Title") or it.get("title") or "",
                            "price": parse_price(it.get("Product_Price") or it.get("price") or 0),
                            "link": it.get("Product_Link") or it.get("link") or "",
                            "image": it.get("Product_Image") or it.get("image") or "",
                            "description": it.get("Product_description") or it.get("description") or "",
                            "qty": it.get("Product_qty") or it.get("qty") or ""
                        })
                    return out
            except Exception:
                return []
    except Exception:
        return []
    return []


@app.post("/chat")
async def chat(req: Request):
    """
    Request body: { "query": "...", "force": false }.
    Response: { products: [...], reply: "...", debug: {...} }
    """
    body = await req.json()
    query = body.get("query") or body.get("message") or ""
    force = body.get("force", False)

    debug = {"stages": []}
    try:
        # 1) Run triage agent (which should delegate)
        result = await Runner.run(Triage_Agent, input=query)
        final_output = result.final_output
        debug["stages"].append({"type": str(type(final_output)), "raw": str(final_output)[:1000]})

        # Attempt normalization
        products = []
        # Case: list of dataclasses/dicts
        if isinstance(final_output, list):
            for item in final_output:
                if isinstance(item, (dict,)):
                    n = normalize_item(item)
                elif hasattr(item, "_dict_"):
                    n = normalize_item(item._dict_)
                else:
                    n = None
                if n:
                    products.append(n)

        # Case: single dataclass/dict
        elif hasattr(final_output, "_dict_") or isinstance(final_output, dict):
            item = final_output._dict_ if hasattr(final_output, "_dict_") else final_output
            n = normalize_item(item)
            if n:
                products.append(n)

        # Case: string -> try to parse markdown products
        elif isinstance(final_output, str):
            md_parsed = parse_markdown_products(final_output)
            if md_parsed:
                products = md_parsed

        # If no products and forced or nothing found, try forcing product agent
        if (not products) or force:
            debug["stages"].append({"action": "calling Product_Agent force-json"})
            forced = await ensure_structured_with_product_agent(query)
            if forced:
                products = forced

        # If still no products, return reply text if available
        reply_text = None
        if isinstance(final_output, str) and final_output.strip():
            reply_text = final_output

        # Prepare response
        return JSONResponse({
            "products": products,
            "reply": reply_text or "",
            "debug": debug
        })

    except Exception as e:
        return JSONResponse({
            "products": [],
            "reply": "",
            "debug": {"error": str(e), "stages": debug.get("stages")}
        }, status_code=500)
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






