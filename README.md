# runpod-worker-llamacpp

RunPod Serverless worker wrapping llama.cpp `llama-server`. Serves GGUF models through
RunPod's OpenAI-compatible route: `https://api.runpod.ai/v2/<endpoint_id>/openai/v1`.

Configure with env `LLAMA_ARGS` (llama-server args, e.g. `-hf org/repo:QUANT -ngl 999 -c 131072 -np 8 --jinja -n 2048`).
Model files download to `LLAMA_CACHE` (`/runpod-volume/llama-cache` when a network volume is attached).
