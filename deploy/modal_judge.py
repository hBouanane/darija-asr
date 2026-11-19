"""Gemma 4 31B as the transcript judge, served by vLLM on Modal (OpenAI-compatible).

    modal deploy deploy/modal_judge.py                      # 4-bit QAT on an L40S (default)
    JUDGE_VARIANT=bf16 modal deploy deploy/modal_judge.py   # full bf16 on an H100
    modal app stop darija-judge                             # when the judging is done

Both variants answer to the model name "judge", so `darija merge` settings stay
the same whichever one runs.

- qat:  google/gemma-4-31B-it-qat-w4a16-ct, Google's quantization-aware-trained
        4-bit checkpoint for vLLM (~23 GB of weights), on an L40S 48 GB (~$1.95/h).
- bf16: google/gemma-4-31B-it (~62 GB), on an H100 80 GB (~$3.95/h).

The server is private: Modal requires a proxy token on every request
(Authorization: Bearer <token-id>.<token-secret>), which is what `darija merge`
sends from merge.api_key_env. It scales to zero after SCALEDOWN of no requests,
and MAX_CONTAINERS caps how many GPUs can run at once (and so the spend).
See docs/run_guide.md, "Judge on Modal".
"""
import json
import os
import subprocess

import modal

VARIANTS = {
    "qat": ("google/gemma-4-31B-it-qat-w4a16-ct", "L40S"),
    "bf16": ("google/gemma-4-31B-it", "H100"),
}
VARIANT = os.environ.get("JUDGE_VARIANT", "qat")   # read when `modal deploy` runs
MODEL, GPU = VARIANTS[VARIANT]
MAX_MODEL_LEN = 8192                 # system prompt + 3-4 candidates + verdict fit in ~3k
MAX_CONTAINERS = 1
SCALEDOWN = 5 * 60                   # seconds without requests before the GPU is released
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "JUDGE_MODEL": MODEL})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("darija-judge")


@app.server(
    image=image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN, for the gated weights
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=8000,
    startup_timeout=30 * MINUTES,     # first start downloads the weights
    scaledown_window=SCALEDOWN,
    max_containers=MAX_CONTAINERS,
    target_concurrency=128,           # requests per container; vLLM batches them
)
class Judge:
    @modal.enter()
    def start(self):
        model = os.environ["JUDGE_MODEL"]
        cmd = [
            "vllm", "serve", model,
            "--served-model-name", "judge", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.92",
            # text only: no image/audio inputs, so no memory reserved for them
            "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0}),
            "--reasoning-parser", "gemma4",
        ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
