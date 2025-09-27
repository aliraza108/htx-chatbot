# app.py  -- debug helper (safe to deploy)
import os
import sys
import platform
import traceback
import importlib
import inspect
import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

def safe_import(name: str) -> Dict[str, Any]:
    """Try to import module 'name' and return status + error trace if any."""
    res = {"module": name, "ok": False, "error": None}
    try:
        mod = importlib.import_module(name)
        res["ok"] = True
        res["version"] = getattr(mod, "__version__", None)
    except Exception as e:
        res["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        res["traceback"] = traceback.format_exc()
    return res

def capture_exc(func_name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and capture exception details."""
    try:
        out = fn(*args, **kwargs)
        return {"ok": True, "result": out}
    except Exception as e:
        return {
            "ok": False,
            "error": "".join(traceback.format_exception_only(type(e), e)).strip(),
            "traceback": traceback.format_exc()
        }

@app.get("/")
async def root():
    return {"message": "Debug endpoint available. Use /debug"}

@app.get("/debug")
async def debug():
    report = {}
    # Basic env & python info
    report["python"] = {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    # Important env vars (presence only - do NOT print secret)
    env_checks = {}
    for name in ["OPENAI_API_KEY", "VERCEL", "PYTHONPATH", "PATH"]:
        env_checks[name] = {"present": bool(os.environ.get(name))}
    report["env"] = env_checks

    # Try to import important modules safely
    modules_to_test = [
        "agents",    # the library that likely causes the crash
        "openai",
        "httpx",
        "uvicorn",
        "fastapi",
        "pydantic",
        "asyncio",
    ]
    imports = {m: safe_import(m) for m in modules_to_test}
    report["imports"] = imports

    # If agents imported, introspect attributes and try a small run
    agents_ok = imports.get("agents", {}).get("ok", False)
    runner_test = {"attempted": False}
    if agents_ok:
        try:
            agents_mod = importlib.import_module("agents")
            # check for attributes we need
            attrs = {}
            for attr in ("Agent", "Runner", "FileSearchTool", "function_tool", "set_tracing_disabled"):
                attrs[attr] = {"present": hasattr(agents_mod, attr)}
            report["agents_attrs"] = attrs

            # call set_tracing_disabled if present (safe)
            if hasattr(agents_mod, "set_tracing_disabled"):
                try:
                    getattr(agents_mod, "set_tracing_disabled")(True)
                    report.setdefault("agents_actions", {})["set_tracing_disabled"] = {"ok": True}
                except Exception as e:
                    report.setdefault("agents_actions", {})["set_tracing_disabled"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()}

            # Build a minimal Agent and Runner *without tools* to avoid external resources:
            if hasattr(agents_mod, "Agent") and hasattr(agents_mod, "Runner"):
                Agent = getattr(agents_mod, "Agent")
                Runner = getattr(agents_mod, "Runner")

                # Create minimal agent instance (do not rely on external tools)
                try:
                    minimal_agent = Agent(name="debug-minimal-agent", instructions="You are a debug agent. Reply 'pong' when asked 'ping'.", model="gpt-3.5-turbo")
                    report.setdefault("agents_actions", {})["create_agent"] = {"ok": True}
                except Exception as e:
                    report.setdefault("agents_actions", {})["create_agent"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
                    minimal_agent = None

                # Create Runner
                try:
                    runner = Runner(minimal_agent)
                    report.setdefault("agents_actions", {})["create_runner"] = {"ok": True}
                except Exception as e:
                    report.setdefault("agents_actions", {})["create_runner"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
                    runner = None

                # Inspect runner.run: is it a coroutine function?
                try:
                    if runner is not None:
                        is_coro_fn = inspect.iscoroutinefunction(getattr(runner, "run", None))
                        report.setdefault("agents_actions", {})["runner_run_is_coroutinefunction"] = is_coro_fn

                        # If it is coroutinefunction, try to call it with timeout to get immediate error or success
                        runner_test["attempted"] = False
                        if is_coro_fn:
                            runner_test["attempted"] = True
                            # Attempt an awaited run with timeout to avoid long blocking
                            async def do_run():
                                try:
                                    # Use a small prompt; may still call OpenAI service
                                    return await asyncio.wait_for(runner.run("ping"), timeout=10.0)
                                except asyncio.TimeoutError:
                                    return {"timeout": True, "note": "runner.run timed out (10s)"}
                                except Exception as e:
                                    return {"error": str(e), "trace": traceback.format_exc()}
                            # execute
                            run_result = await do_run()
                            runner_test["result"] = run_result
                        else:
                            runner_test["error"] = "runner.run is not a coroutine function; ensure you 'await' it where used."
                    else:
                        runner_test["error"] = "runner could not be created"
                except Exception as e:
                    runner_test["ok"] = False
                    runner_test["error"] = str(e)
                    runner_test["traceback"] = traceback.format_exc()
            else:
                report.setdefault("agents_actions", {})["create_agent"] = {"ok": False, "error": "Agent or Runner not present in agents module"}
        except Exception as e:
            report.setdefault("agents_actions", {})["import_error"] = traceback.format_exc()
    else:
        report["agents_attrs"] = {"present": False, "note": "agents failed to import"}

    report["runner_test"] = runner_test

    # Extra: show a small subset of installed packages (from pip freeze) if available
    try:
        import pkg_resources
        pkgs = {d.project_name: d.version for d in pkg_resources.working_set}
        # filter to few relevant keys to keep output small
        keys = ["openai", "httpx", "fastapi", "uvicorn", "pydantic", "openai-agents"]
        report["installed_versions"] = {k: pkgs.get(k) for k in keys}
    except Exception:
        report["installed_versions"] = "pkg_resources not available or failed"

    # Print the full report to logs (Vercel will capture)
    print("DEBUG REPORT:\n", json.dumps(report, indent=2))
    return JSONResponse(report)
