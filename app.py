from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
from typing import List
import os

from agents import function_tool, Runner, Agent, set_tracing_disabled, FileSearchTool

# --- OpenAI API Key setup ---
part1 = 'sk-proj'
part2 = '-Mfz8dQmnuGhRCYwOE419Eno6ayP0yD2gTgoOLZz3uW6DFvIhO'
part3 = '--LuMyTVIVl1P8_OS4_4b-FtJT3BlbkFJ5yZqmw_sp8QdqC2GcD1gCKPyvdzf_MLCv1xaT8T_oY8wPdliX5c10uvhFRRJD1MJTJJMTqk4UA'
OPENAI_API_KEY = part1 + part2 + part3
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Disable tracing for production
set_tracing_disabled(True)

# --- Agents setup ---
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
    output_type=list[Product_output],
)

StoreInfoAgent = Agent(
    name="Store Info Agent",
    instructions="""
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
""",
    tools=[
        FileSearchTool(
            max_num_results=30,
            vector_store_ids=["vs_68d4ea25a3d08191babc7ee15c21a6cb"],
        ),
    ],
    handoffs=[Product_Agent, StoreInfoAgent],
    model="gpt-4o-mini",
)

# --- FastAPI App ---
app = FastAPI()

# Allow CORS (useful for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "App is running. Use /chat with GET ?message= or POST JSON."}

# --- Chat endpoint ---
@app.get("/chat")
async def chat_get(message: str):
    result = await Runner.run(Triage_Agent, input=message)
    return {"reply": result.final_output}

@app.post("/chat")
async def chat_post(request: Request):
    body = await request.json()
    message = body.get("message", "")
    result = await Runner.run(Triage_Agent, input=message)
    return JSONResponse({"reply": result.final_output})
