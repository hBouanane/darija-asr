"""A one-page review tool for the gold set: gold/review.html next to gold.csv.

For each clip it shows the audio, every source's transcript and the judge's
verdict, with an editable gold text pre-filled from gold.csv. Edits autosave in
the browser (localStorage) and "Export gold.csv" downloads the file `darija eval`
reads. No server: open the HTML from the gold folder (audio paths are relative).
"""
from __future__ import annotations

import json

import pandas as pd

from ..agreement import load_sources
from ..store import Run, latest_per_id, read_parts
from .gold import GOLD_COLUMNS, export_gold


def gold_records(cfg: dict) -> list[dict]:
    run = Run(cfg)
    csv_path = run.gold_dir / "gold.csv"
    if not csv_path.exists():
        export_gold(cfg, prefill="merged" if run.labels_path.exists() else "reference")
    gold = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    srcs = cfg["agreement"]["sources"]
    sources = load_sources(run, srcs).set_index("id")
    merged = latest_per_id(read_parts(run.merge_dir))
    merged = merged[merged["error"].isna()].set_index("id") if not merged.empty else merged

    records = []
    for row in gold.to_dict("records"):
        cid = row["id"]
        cands = {s: (str(sources.at[cid, s]) if cid in sources.index and pd.notna(sources.at[cid, s]) else "")
                 for s in srcs}
        judge = None
        if not merged.empty and cid in merged.index:
            m = merged.loc[cid]
            judge = {"text": m["text"] or "", "confidence": m["confidence"], "notes": m["notes"] or "",
                     "reject": bool(m["reject"])}
        records.append({**{c: row.get(c, "") for c in GOLD_COLUMNS}, "candidates": cands, "judge": judge})
    return records


def build_page(records: list[dict], run_name: str) -> str:
    data = json.dumps({"run": run_name, "columns": GOLD_COLUMNS, "records": records}, ensure_ascii=False)
    return PAGE.replace("__DATA__", data.replace("</", "<\\/"))


def write_gold_ui(cfg: dict) -> None:
    run = Run(cfg)
    records = gold_records(cfg)
    out = run.gold_dir / "review.html"
    out.write_text(build_page(records, run.name), encoding="utf-8")
    print(f"Wrote {out} ({len(records)} clips). Open it in a browser from that folder; "
          "use 'Export gold.csv' and save it over gold/gold.csv when done.")
    run.push("gold/review.html", "gold review page")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Review</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--text:#1f1e1c;--muted:#6b6a65;--border:#e3e1da;--accent:#2f6fde;--accent-text:#fff;
--ok:#1f7a4a;--ok-bg:#e6f4ec;--diff:#fbe3d6;--diff-text:#8a3a12;--judge:#eef4ec;--judge-border:#b9d3b1;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1b1a18;--card:#252421;--text:#ecebe6;--muted:#a3a19a;
--border:#3a3935;--accent:#6b9cf0;--accent-text:#0b1220;--ok:#7fd3a4;--ok-bg:#1f3328;--diff:#5a2c17;--diff-text:#ffd2bb;
--judge:#1f2b1d;--judge-border:#4b6b43;--warn:#f0b35a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
.top{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
h1{font-size:18px;font-weight:600;margin:0}
.bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin:6px 0 14px}
.bar>div{height:100%;background:var(--ok)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.meta{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;margin-bottom:10px}
.meta .id{font-weight:600}
.muted{color:var(--muted);font-size:13px}
audio{width:100%;margin:6px 0 12px}
.ar{font-family:"Noto Naskh Arabic","Segoe UI",system-ui,sans-serif;font-size:19px;line-height:1.9}
textarea{width:100%;min-height:110px;padding:10px 12px;border:1px solid var(--border);border-radius:8px;
background:var(--bg);color:var(--text);resize:vertical}
input[type=text]{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text)}
.cands{display:grid;gap:8px;margin-top:14px}
.cand{border:1px solid var(--border);border-radius:8px;padding:8px 10px}
.cand.judge{background:var(--judge);border-color:var(--judge-border)}
.cand .head{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12px;color:var(--muted)}
.cand p{margin:4px 0 0}
.diff{background:var(--diff);color:var(--diff-text);border-radius:3px;padding:0 2px}
button,select{font:inherit;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:8px;
padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
button.small{padding:2px 8px;font-size:12px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
.checked{color:var(--ok);background:var(--ok-bg);border-radius:6px;padding:2px 8px;font-size:13px}
.warn{color:var(--warn)}
kbd{border:1px solid var(--border);border-radius:4px;padding:0 4px;font-size:12px}
label.chk{display:flex;gap:6px;align-items:center;cursor:pointer}
</style></head><body><main>
<div class="top">
  <h1 id="title">Gold review</h1>
  <div class="row" style="margin:0">
    <label class="muted"><input type="checkbox" id="onlyUnchecked"> unchecked only</label>
    <button id="export" class="primary">Export gold.csv</button>
  </div>
</div>
<div class="muted" id="progressText"></div>
<div class="bar"><div id="progress" style="width:0"></div></div>
<section class="card" id="card"></section>
<p class="muted">Shortcuts: <kbd>Alt</kbd>+<kbd>P</kbd> play/pause, <kbd>Alt</kbd>+<kbd>R</kbd> replay,
<kbd>Alt</kbd>+<kbd>←</kbd>/<kbd>→</kbd> previous/next, <kbd>Ctrl</kbd>+<kbd>Enter</kbd> mark checked and go to next.
Edits are saved in this browser automatically; export the CSV to keep them.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const KEY = "gold-ui:" + DATA.run;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
const recs = DATA.records.map(r => ({...r, ...(saved[r.id] || {})}));
let idx = Math.max(0, recs.findIndex(r => !isChecked(r)));

function isChecked(r) { return ["x","1","y","yes","true","ok"].includes(String(r.checked).trim().toLowerCase()); }
function persist(r) {
  saved[r.id] = {gold_text: r.gold_text, checked: r.checked, comment: r.comment};
  try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
}
function norm(w) {
  return w.normalize("NFKD").replace(/[̀-ًͯ-ٰٟـ]/g, "")
    .replace(/[ىی]/g, "ي").replace(/ة/g, "ه").replace(/[ک]/g, "ك")
    .replace(/[^\p{L}\p{N}]/gu, "").toLowerCase();
}
function words(t) { return new Set((t || "").split(/\s+/).map(norm).filter(Boolean)); }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function marked(text, keep) {
  return (text || "").split(/\s+/).filter(Boolean).map(w => {
    const n = norm(w); return n && !keep.has(n) ? `<span class="diff">${esc(w)}</span>` : esc(w);
  }).join(" ");
}
function visible() { return document.getElementById("onlyUnchecked").checked ? recs.filter(r => !isChecked(r)) : recs; }

function render() {
  const done = recs.filter(isChecked).length;
  document.getElementById("title").textContent = `Gold review: ${DATA.run}`;
  document.getElementById("progressText").textContent = `${done} / ${recs.length} checked`;
  document.getElementById("progress").style.width = (100 * done / Math.max(1, recs.length)) + "%";
  const list = visible();
  const card = document.getElementById("card");
  if (!list.length) { card.innerHTML = "<p>Nothing left to check.</p>"; return; }
  idx = Math.min(Math.max(idx, 0), list.length - 1);
  const r = list[idx];
  const keep = words(r.gold_text);
  const cands = Object.entries(r.candidates).filter(([, t]) => t).map(([name, t]) =>
    `<div class="cand"><div class="head"><span>${esc(name)}</span>
     <button class="small" data-copy="${esc(name)}">use this</button></div>
     <p class="ar" dir="rtl">${marked(t, keep)}</p></div>`).join("");
  const j = r.judge;
  const judge = j ? `<div class="cand judge"><div class="head"><span>LLM judge · ${esc(j.confidence || "")}
     ${j.reject ? '<span class="warn">· rejected</span>' : ""}</span>
     <button class="small" data-copy="__judge">use this</button></div>
     <p class="ar" dir="rtl">${marked(j.text, keep)}</p>
     ${j.notes ? `<p class="muted">${esc(j.notes)}</p>` : ""}</div>` : "";
  card.innerHTML = `
    <div class="meta"><span class="id">#${esc(r.n)} · ${esc(r.id)}</span>
      <span class="muted">${esc(r.channel)} · ${esc(r.duration)} s · ${idx + 1} / ${list.length}</span>
      ${isChecked(r) ? '<span class="checked">checked</span>' : ""}</div>
    <audio id="player" controls src="${esc(r.audio_file)}"></audio>
    <label class="muted" for="gold">Gold transcript (what is actually said)</label>
    <textarea id="gold" class="ar" dir="rtl">${esc(r.gold_text)}</textarea>
    <div class="row">
      <label class="chk"><input type="checkbox" id="chk" ${isChecked(r) ? "checked" : ""}> checked</label>
      <input type="text" id="comment" placeholder="comment (optional)" value="${esc(r.comment || "")}" style="flex:1;min-width:200px">
    </div>
    <div class="row">
      <button id="prev">← Previous</button>
      <button id="next">Next →</button>
      <button id="checkNext" class="primary">Mark checked and next</button>
    </div>
    <div class="cands">${judge}${cands}</div>`;
  const gold = document.getElementById("gold");
  gold.addEventListener("input", () => { r.gold_text = gold.value; persist(r); });
  document.getElementById("comment").addEventListener("input", e => { r.comment = e.target.value; persist(r); });
  document.getElementById("chk").addEventListener("change", e => { r.checked = e.target.checked ? "x" : ""; persist(r); render(); });
  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("next").onclick = () => go(1);
  document.getElementById("checkNext").onclick = checkNext;
  card.querySelectorAll("[data-copy]").forEach(b => b.onclick = () => {
    const k = b.dataset.copy;
    r.gold_text = k === "__judge" ? r.judge.text : r.candidates[k];
    persist(r); render();
  });
}
function go(step) {
  const p = document.getElementById("player"); if (p) p.pause();
  idx += step; render();
  const n = document.getElementById("player"); if (n) n.play().catch(() => {});
}
function checkNext() {
  const list = visible(); const r = list[idx]; if (!r) return;
  r.checked = "x"; persist(r);
  if (document.getElementById("onlyUnchecked").checked) { render(); const n = document.getElementById("player"); if (n) n.play().catch(() => {}); }
  else go(1);
}
document.getElementById("onlyUnchecked").addEventListener("change", () => { idx = 0; render(); });
document.addEventListener("keydown", e => {
  const p = document.getElementById("player");
  if (e.altKey && e.key.toLowerCase() === "p" && p) { e.preventDefault(); p.paused ? p.play() : p.pause(); }
  else if (e.altKey && e.key.toLowerCase() === "r" && p) { e.preventDefault(); p.currentTime = 0; p.play(); }
  else if (e.altKey && e.key === "ArrowRight") { e.preventDefault(); go(1); }
  else if (e.altKey && e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
  else if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); checkNext(); }
});
document.getElementById("export").onclick = () => {
  const cell = v => { const s = String(v ?? ""); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const lines = [DATA.columns.join(",")].concat(recs.map(r => DATA.columns.map(c => cell(r[c])).join(",")));
  const blob = new Blob(["﻿" + lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "gold.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
render();
</script>
</body></html>
"""
