# app.py
import os
import platform
import sys
from fastapi import FastAPI
from agents import Agent, Runner, set_tracing_disabled

app = FastAPI()

# disable extra tracing logs
set_tracing_disabled()

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
            "OPENAI_API_KEY": {"present": "OPENAI_API_KEY" in os.environ},
        },
        "runner_test": None,
    }

    try:
        # define a minimal agent
        agent = Agent(
            name="TestAgent",
            instructions="Reply with 'pong' if the user says 'ping'."
        )

        runner = Runner()
        # since this is FastAPI running async, let's use the sync runner
        result = runner.run_sync(agent, "ping")

        report["runner_test"] = {"ok": True, "result": str(result)}
    except Exception as e:
        import traceback
        report["runner_test"] = {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

    return report
