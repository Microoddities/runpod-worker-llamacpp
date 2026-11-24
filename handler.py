"""
RunPod Serverless worker that wraps llama.cpp's llama-server.

Starts llama-server once per worker, then forwards jobs to it:
- OpenAI-compatible route: RunPod sends {"openai_route": "/v1/chat/completions",
  "openai_input": {...}} for requests to https://api.runpod.ai/v2/<id>/openai/v1/...
- Plain jobs: {"input": {"messages": [...], ...}} -> /v1/chat/completions
              {"input": {"prompt": "...", ...}}   -> /v1/completions

Config (env):
  LLAMA_ARGS        llama-server args, e.g. "-hf org/repo:QUANT -ngl 999 -c 131072 -np 8 --jinja"
  LLAMA_PORT        internal port (default 8080)
  MAX_CONCURRENCY   jobs handled at once per worker (default: -np value or 8)
  STARTUP_TIMEOUT   seconds to wait for model load (default 3600)
"""

import asyncio
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import httpx
import runpod

PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE = f"http://127.0.0.1:{PORT}"
ARGS = shlex.split(os.environ.get("LLAMA_ARGS", ""))
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "3600"))


def _np_from_args(args):
    for flag in ("-np", "--parallel"):
        if flag in args:
            try:
                return int(args[args.index(flag) + 1])
            except (ValueError, IndexError):
                pass
    return 8


MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", _np_from_args(ARGS)))


def start_server():
    cmd = ["/app/llama-server", *ARGS, "--host", "127.0.0.1", "--port", str(PORT)]
    shown = " ".join(a if prev != "--api-key" else "<redacted>" for prev, a in zip([""] + cmd, cmd))
    print(f"[worker] starting: {shown}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            r = httpx.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                print("[worker] llama-server ready", flush=True)
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("llama-server did not become ready in time")


def _cache_size_gb():
    cache = os.environ.get("LLAMA_CACHE")
    if not cache or not os.path.isdir(cache):
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(cache):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


# Load the model in the background so the worker registers with RunPod right
# away. A large first-time download can otherwise outlast RunPod's startup
# window, and the platform restarts the container before it ever takes a job.
READY = threading.Event()
START_ERROR = []


def _boot():
    try:
        start_server()
        READY.set()
    except Exception as e:  # surfaced to every job via _wait_ready
        START_ERROR.append(str(e))
        print(f"[worker] startup failed: {e}", flush=True)
        READY.set()


def _progress():
    while not READY.wait(30):
        print(f"[worker] loading model... cache {_cache_size_gb():.1f} GB", flush=True)


threading.Thread(target=_boot, daemon=True).start()
threading.Thread(target=_progress, daemon=True).start()
CLIENT = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(None, connect=10))


async def _wait_ready():
    await asyncio.to_thread(READY.wait)
    if START_ERROR:
        raise RuntimeError(f"llama-server failed to start: {START_ERROR[0]}")


# Only these llama-server routes may be reached through a job. An absolute URL
# in openai_route would otherwise make httpx ignore base_url (SSRF).
ALLOWED_ROUTES = {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"}


def _safe_path(route):
    parts = urlsplit(str(route))
    path = "/" + parts.path.strip("/")
    if parts.scheme or parts.netloc or path not in ALLOWED_ROUTES:
        raise ValueError(f"route not allowed: {route!r}")
    return path


def _route(job_input):
    """Return (method, path, body, stream) for a job."""
    if "openai_route" in job_input:
        path = _safe_path(job_input["openai_route"])
        body = job_input.get("openai_input") or {}
        method = "GET" if path.rstrip("/").endswith("/models") else "POST"
        return method, path, body, bool(body.get("stream"))
    body = dict(job_input)
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    return "POST", path, body, bool(body.get("stream"))


async def handler(job):
    await _wait_ready()
    method, path, body, stream = _route(job["input"])

    if method == "GET":
        r = await CLIENT.get(path)
        yield r.json()
        return

    if not stream:
        r = await CLIENT.post(path, json=body)
        try:
            data = r.json()
        except ValueError:
            data = {"error": {"message": r.text, "code": r.status_code}}
        yield data
        return

    # Streaming: pass SSE events through as "data: ...\n\n" strings,
    # the shape RunPod's OpenAI route re-emits to the client.
    async with CLIENT.stream("POST", path, json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                yield line + "\n\n"


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _current: MAX_CONCURRENCY,
        "return_aggregate_stream": True,
    }
)
