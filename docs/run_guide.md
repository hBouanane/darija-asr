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
[ -x /tmp/vllm-env/bin/vllm ] || {
  uv venv -q --clear /tmp/vllm-env --python 3.12
  uv pip install -q --python /tmp/vllm-env/bin/python vllm --torch-backend=auto "mistral-common[audio]>=1.9.0" "transformers>=5"
}
uv pip uninstall -q --python /tmp/vllm-env/bin/python torchaudio || true
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

Same setup as step 1 (`HF_TOKEN`, `DARIJA_HUB_REPO`), plus the vLLM server environment (once per VM).
Remove torchaudio from it: `--torch-backend=auto` can pick a PyTorch built for a newer CUDA than
torchaudio's, and vLLM then fails at import ("PyTorch and TorchAudio were compiled with different CUDA
versions"). vLLM does not need it for these models.

```bash
[ -x /content/vllm-env/bin/vllm ] || {
  uv venv -q --clear /content/vllm-env --python 3.12
  uv pip install -q --python /content/vllm-env/bin/python vllm --torch-backend=auto "mistral-common[audio]>=1.9.0" "transformers>=5"
}
uv pip uninstall -q --python /content/vllm-env/bin/python torchaudio || true
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

Gemini through its OpenAI-compatible API (instead of the default Gemma judge, see below):

```bash
export GEMINI_API_KEY=xxx
G='--set merge.base_url=https://generativelanguage.googleapis.com/v1beta/openai --set merge.api_key_env=GEMINI_API_KEY
   --set merge.model=gemini-2.5-pro --set merge.extra_body={} --set merge.workers=16 --set merge.max_tokens=8192'
for run in gold gt100 extra; do
  uv run darija agree  -c configs/$run.yaml     # pulls every machine's transcripts
  uv run darija merge  -c configs/$run.yaml $G  # one clip per request, 16 in flight; resumable
  uv run darija labels -c configs/$run.yaml     # -> work/<run>/labels.parquet (+ hub)
done
uv run darija build-train -c configs/gt100.yaml -c configs/extra.yaml --out your-username/darija-train --push
```

`merge` skips clips that already have a verdict, and waits for clips that some model has not
transcribed yet ("N clips wait for missing transcripts"): rerun it when they are done.

### Judge on Modal (Gemma 4 31B)

`deploy/modal_judge.py` serves Gemma 4 31B with vLLM on Modal. By default it runs Google's 4-bit
quantization-aware-trained checkpoint (`google/gemma-4-31B-it-qat-w4a16-ct`, ~23 GB) on one L40S
(~$1.95/h); `JUDGE_VARIANT=bf16` runs the full model (~62 GB) on one H100 (~$3.95/h) instead.
Both answer to the model name `judge`. The server is private (a Modal proxy token is required),
scales to zero after 5 minutes without requests, and never runs more than one GPU.

Once:

1. Accept the Gemma license on https://huggingface.co/google/gemma-4-31B-it (and on
   https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct if it asks separately).
2. Install the CLI and log in: `pip install modal`, then `modal setup`.
3. Give Modal your HF token (for the gated weights): `modal secret create huggingface-secret HF_TOKEN=hf_xxx`
4. Create a proxy token: Modal dashboard > Settings > Proxy Auth Tokens. Keep its ID and secret.

Deploy (prints the server URL, e.g. `https://<workspace>--darija-judge-judge.us-east.modal.direct`):

```bash
modal deploy deploy/modal_judge.py                      # 4-bit QAT on an L40S
# JUDGE_VARIANT=bf16 modal deploy deploy/modal_judge.py # full bf16 on an H100
```

Judge, from any machine with the repo (no GPU needed). The first request starts the GPU and
downloads the weights (~23 GB for QAT, a few minutes); later starts are faster.

```bash
export DARIJA_JUDGE_URL="https://<workspace>--darija-judge-judge.us-east.modal.direct/v1"
export MODAL_JUDGE_TOKEN="<token-id>.<token-secret>"
export WANDB_PROJECT=darija-asr        # optional: follow the run on wandb.ai (once: `uv run wandb login`)

uv run darija agree -c configs/gold.yaml
uv run darija merge -c configs/gold.yaml --limit 30     # test: read the output, note the speed
uv run darija merge -c configs/gold.yaml                # the rest of gold
for run in gt100 extra; do
  uv run darija agree -c configs/$run.yaml
  uv run darija merge -c configs/$run.yaml
done
modal app stop darija-judge     # stop paying as soon as it is done
```

The judge defaults (`configs/base.yaml`, `merge:`) are this Gemma server: model name `judge`,
temperature 0 (greedy, so the same candidates always give the same verdict), thinking off,
96 requests in flight.

With `WANDB_PROJECT` set, each `merge` is a wandb run (grouped by run name) showing clips judged,
speed, errors, reject rate, confidence and tokens every 100 clips, plus a `judged_clips` table with
the candidates next to the verdict; `transcribe` and `labels` log their progress and totals too.

Cost: ~35k clips at ~2k tokens in and ~150 out each: roughly 1-3 GPU hours, so about $3-8 on the
L40S (QAT) or $5-15 on the H100 (bf16), including start-up. To compare quality, judge 30 gold clips
with each variant (redeploy in between) and read them side by side. Thinking stays off (`enable_thinking: false`): it would multiply
the output tokens, and so the cost, several times. `merge` resumes where it stopped, so a dropped
connection or a stopped app loses nothing.

Gemma on another GPU host (RunPod, a university server): run the same `vllm serve` command there and
point `merge.base_url` at `http://<host>:8000/v1` with `merge.api_key_env=null`.

## 5. Gold set

After `labels` for the gold run:

```bash
uv run darija gold-ui -c configs/gold.yaml   # exports wav + gold.csv (pre-filled with the judge's text) + review.html
```

Download the whole `work/gold/gold/` folder (audio included) and open `review.html` in a browser.
Each clip shows its audio, the gold text to correct, the judge's verdict and every source, with the
words that differ from your text highlighted ("use this" copies a candidate into the box).
Shortcuts: Alt+P play/pause, Alt+R replay, Alt+←/→ previous/next, Ctrl+Enter mark checked and next.
Edits autosave in that browser. When done (or to keep a backup), click **Export gold.csv**, put the
file back as `work/gold/gold/gold.csv`, then:

```bash
uv run darija push -c configs/gold.yaml --path gold/gold.csv
uv run darija eval -c configs/gold.yaml        # only checked rows count
```

Rerunning `gold-ui` later keeps your corrections: it reads the existing `gold.csv`.

## Adding more data later

```bash
uv run darija prepare -c configs/extra.yaml --add --set select.n=10000   # one machine
# then rerun the same transcribe commands on Kaggle and Colab: they take only the new clips
# then rerun agree / merge / labels: merge judges only the new clips
```
