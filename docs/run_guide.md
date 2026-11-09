# Run guide: Voxtral on Kaggle, Cohere + MoulSot on Colab, judge anywhere

Three runs, each a YAML in `configs/`:

| run | clips | sources the judge sees |
|---|---|---|
| `gold` | 300 from `100-gt-2.5` **test** (MoulSot never trained on it) | reference, cohere, voxtral, moulsot |
| `gt100` | 5,000 from `100-gt-2.5` train (what MoulSot was trained on) | reference, cohere, voxtral |
| `extra` | 10,000 from `extra-gt-2.5` | cohere, voxtral, moulsot |

`select.n` is how many clips; `prepare --add` grows a run later. MoulSot stays out of
`gt100`: on its own training clips it mostly repeats the Gemini reference.

## How several machines share a run

Every machine reads and writes one private HF dataset repo (`DARIJA_HUB_REPO`).
That is safe while models run at the same time on different machines because:

- each model writes only to `<run>/asr/<model>/`, and each save is a new file named
  `part-<time>-<random>.parquet`; nothing shared is ever rewritten or deleted;
- `transcribe` first pulls the clip list and that model's earlier parts, so a fresh
  VM resumes where the last one stopped and never redoes finished clips;
- a failed upload is retried, and if the hub is down the part is uploaded with the next one;
- `merge` judges a clip only once every source of the run has transcribed it.

One rule: run `prepare` (and `prepare --add`) from **one machine at a time**.

## 0. Once

1. Create an HF token with **write** access, and accept the Cohere model's terms on its Hub page.
2. Pick the repo name, e.g. `your-username/darija-asr-work` (created private on first upload).
3. A Gemini API key for the judge (or a Gemma server, see step 4).

## 1. Colab: pick the clips (T4 or even CPU)

```bash
cd /content && pip install -q uv && git clone https://github.com/hBouanane/darija-asr.git && cd darija-asr
export HF_TOKEN=hf_xxx DARIJA_HUB_REPO=your-username/darija-asr-work
uv run darija prepare -c configs/gold.yaml
uv run darija prepare -c configs/gt100.yaml
uv run darija prepare -c configs/extra.yaml
```

Change sizes with `--set select.n=...`. Later, add clips to a run (only new ones are picked):

```bash
uv run darija prepare -c configs/extra.yaml --add --set select.n=10000
```

## 2. Kaggle: Voxtral on 2 x T4

Notebook settings: **Accelerator GPU T4 x2**, **Internet on**, and a secret named `HF_TOKEN`
(Add-ons > Secrets). For long runs use **Save Version > Save & Run All**: it runs headless
for up to 12 h and resumes from the hub next time.

Cell 1 (Python): credentials for every later cell.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["DARIJA_HUB_REPO"] = "your-username/darija-asr-work"
```

Cell 2: code + a separate environment for the vLLM server (in `/tmp`: `/kaggle/working` is limited to 20 GB).

```bash
%%bash
cd /tmp && pip install -q uv
git clone https://github.com/hBouanane/darija-asr.git
cd darija-asr && uv sync
uv venv /tmp/vllm-env --python 3.12
uv pip install --python /tmp/vllm-env/bin/python -U vllm --torch-backend=auto "mistral-common[audio]>=1.9.0" "transformers>=5"
```

Cell 3 (Python): one vLLM server using both GPUs (`--data-parallel-size 2`), then wait until it is up.

```python
import subprocess, time, pathlib
log = open("/tmp/vllm.log", "w")
server = subprocess.Popen(
    ["/tmp/vllm-env/bin/vllm", "serve", "mistralai/Voxtral-Mini-4B-Realtime-Arabic",
     "--tokenizer-mode", "mistral", "--dtype", "half", "--max-model-len", "1024",
     "--gpu-memory-utilization", "0.92", "--data-parallel-size", "2",
     "--compilation_config", '{"cudagraph_mode": "PIECEWISE"}'],
    stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "VLLM_DISABLE_COMPILE_CACHE": "1"})
while "Application startup complete" not in pathlib.Path("/tmp/vllm.log").read_text():
    if server.poll() is not None:
        print(pathlib.Path("/tmp/vllm.log").read_text()[-4000:]); raise SystemExit("server died")
    time.sleep(10)
print("ready")
```

Cell 4: transcribe every run (24 clips in flight = 12 per GPU). Rerun it any time; it resumes.

```bash
%%bash
cd /tmp/darija-asr
for run in gold gt100 extra; do
  uv run darija transcribe -c configs/$run.yaml --model voxtral --set asr.models.voxtral.concurrency=24
done
```

If the server dies with `--data-parallel-size 2`, run one server per GPU instead
(`CUDA_VISIBLE_DEVICES=0` on port 8000, `CUDA_VISIBLE_DEVICES=1` with `--port 8001`) and two
transcribe processes at the same time, each on half of the clips:

```bash
%%bash
cd /tmp/darija-asr
for run in gold gt100 extra; do
  uv run darija transcribe -c configs/$run.yaml --model voxtral --shard 0 --num-shards 2 &
  uv run darija transcribe -c configs/$run.yaml --model voxtral --shard 1 --num-shards 2 \
      --set asr.models.voxtral.url=ws://localhost:8001/v1/realtime &
  wait
done
```

Speed: ~3.9x real time per T4, so 10,000 extra clips (~17 h of audio) take ~2-2.5 h on two GPUs.

## 3. Colab: Cohere and MoulSot (T4)

Same setup as step 1 (`HF_TOKEN`, `DARIJA_HUB_REPO`), plus the vLLM server environment:

```bash
uv venv /content/vllm-env --python 3.12
uv pip install --python /content/vllm-env/bin/python -U vllm --torch-backend=auto "mistral-common[audio]>=1.9.0" "transformers>=5"
```

Cohere (transformers is faster than vLLM on a T4):

```bash
for run in gold gt100 extra; do
  uv run --extra asr darija transcribe -c configs/$run.yaml --model cohere --set asr.models.cohere.backend=transformers
done
```

MoulSot (vLLM; the Hub repo has a stale index file, so serve a cleaned local copy):

```bash
/content/vllm-env/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('atlasia/moulsot.v0.3', local_dir='/content/moulsot', ignore_patterns=['*.pth', 'trainer_state.json'])"
rm -f /content/moulsot/model.safetensors.index.json
nohup /content/vllm-env/bin/vllm serve /content/moulsot --served-model-name atlasia/moulsot.v0.3 --dtype half --max-model-len 2048 --gpu-memory-utilization 0.9 > vllm.log 2>&1 &
while ! grep -q "Application startup complete" vllm.log; do pgrep -f "vllm serve" >/dev/null || { echo "server died"; tail -40 vllm.log; break; }; sleep 10; done
for run in gold extra; do
  uv run darija transcribe -c configs/$run.yaml --model moulsot
done
```

Only one model fits on a T4: run Cohere first (no server), then start the MoulSot server.

## 4. Final labels: the judge (any machine, no GPU)

Gemini through its OpenAI-compatible API (the default in `configs/base.yaml`):

```bash
export GEMINI_API_KEY=xxx
for run in gold gt100 extra; do
  uv run darija agree  -c configs/$run.yaml     # pulls every machine's transcripts
  uv run darija merge  -c configs/$run.yaml     # one clip per request, 16 in flight; resumable
  uv run darija labels -c configs/$run.yaml     # -> work/<run>/labels.parquet (+ hub)
done
uv run darija build-train -c configs/gt100.yaml -c configs/extra.yaml --out your-username/darija-train --push
```

`merge` skips clips that already have a verdict, and waits for clips that some model has not
transcribed yet ("N clips wait for missing transcripts"): rerun it when they are done.

Gemma 4 31B on a rented GPU instead: start `vllm serve google/gemma-4-31b-it --max-model-len 8192` on
the pod, then point the judge at it:

```bash
J="--set merge.base_url=http://<pod-ip>:8000/v1 --set merge.api_key_env=null --set merge.model=google/gemma-4-31b-it --set merge.workers=96"
uv run darija merge -c configs/extra.yaml $J --set "merge.extra_body={chat_template_kwargs: {enable_thinking: false}}"
```

## 5. Gold set

```bash
uv run darija gold-export -c configs/gold.yaml      # wav files + gold.csv to correct by hand
# listen, fix gold_text, put x in `checked`, then:
uv run darija eval -c configs/gold.yaml
```

## Adding more data later

```bash
uv run darija prepare -c configs/extra.yaml --add --set select.n=10000   # one machine
# then rerun the same transcribe commands on Kaggle and Colab: they take only the new clips
# then rerun agree / merge / labels: merge judges only the new clips
```
