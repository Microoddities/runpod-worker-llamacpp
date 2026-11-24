# llama.cpp CUDA server + RunPod serverless handler
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "runpod>=1.7" "httpx>=0.27"

ENV LLAMA_CACHE=/runpod-volume/llama-cache \
    PYTHONUNBUFFERED=1

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["/opt/venv/bin/python", "/handler.py"]
