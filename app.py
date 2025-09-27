# app.py
import os
import sys
import platform
from fastapi import FastAPI
from agents import (
    function_tool,
    Runner,
    Agent,
    set_default_openai_api,
    set_tracing_disabled,
    set_default_openai_client,
    RunConfig,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)

# Load OpenAI API key from Vercel env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create OpenAI client (official base_url)
client = AsyncOpenAI(
    base_url="https://api.openai.com/v1/",
    api_key=OPENAI_API_KEY,
)

# Global agent config
set_default_openai_api("chat_completions")
set_default_openai_client(client=client)
set_tracing_disabled(True)

# Define model + agent
MODEL = "gpt-4o-mini"  # ✅ good for cost & speed
agent = Agent(
    name="SEO Agent",
    instructions="You are a helpful assistant that replies concisely.",
    model=MODEL,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "App running. Use /debug"}


@app.get("/debug")
async def debug():
    report = {
        "python": {
            "version": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "env": {
            "OPENAI_API_KEY": {"present": OPENAI_API_KEY is not None},
        },
        "runner_test": None,
    }

    try:
        # ✅ Use run_sync (since FastAPI will execute async loop anyway on Vercel)
        result = Runner.run_sync(
            agent,
            input="Say the name of your planet in one word.",
        )
        report["runner_test"] = {
            "ok": True,
            "final_output": str(result.final_output),
        }
    except Exception as e:
        import traceback

        report["runner_test"] = {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

    return report
