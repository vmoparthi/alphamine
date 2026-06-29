"""Render a self-contained HTML dashboard for a mining run.

`render_html(config, rows, reflection)` returns a single HTML string with all data
embedded — no server, no external assets, no build step. Open the file locally or
serve it from S3; it works either way. Light/dark aware.
"""
from __future__ import annotations

import html
import json
from typing import Dict, List, Optional

_CSS = """
:root{--bg:#fff;--fg:#1a1a18;--muted:#6b6a64;--line:#e6e4dc;--card:#faf9f5;
--pos:#0f6e56;--neg:#a32d2d;--accent:#534ab7}
@media(prefers-color-scheme:dark){:root{--bg:#1c1c1a;--fg:#ece9e0;--muted:#a3a199;
--line:#34332f;--card:#262521;--pos:#5dcaa5;--neg:#f09595;--accent:#afa9ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 22px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:6px 10px;font-size:13px}.chip b{color:var(--accent)}
.bar{display:flex;gap:12px;align-items:center;margin:0 0 12px;flex-wrap:wrap}
input[type=search]{flex:1;min-width:200px;padding:8px 12px;border:1px solid var(--line);
border-radius:8px;background:var(--bg);color:var(--fg);font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{cursor:pointer;user-select:none;color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--bg)}
th:hover{color:var(--fg)}td.expr{font-family:ui-monospace,Menlo,monospace;max-width:520px}
.rat{color:var(--muted);font-size:12px}.pos{color:var(--pos)}.neg{color:var(--neg)}
.count{color:var(--muted);font-size:13px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;margin:0 0 22px}
summary{cursor:pointer;font-size:14px}pre{white-space:pre-wrap;font-size:12px;color:var(--muted);margin:8px 0 0}
.note{color:var(--muted);font-size:12px;margin-top:24px}
"""

_JS = """
const rows=DATA;let sortKey='test_rank_ic',sortDir=-1;
const tb=document.getElementById('tb'),q=document.getElementById('q'),cnt=document.getElementById('cnt');
function num(v){return v===null||v===undefined?'':(v>=0?'+':'')+v.toFixed(3)}
function cls(v){return v===null||v===undefined?'':v>=0?'pos':'neg'}
function render(){
  const f=q.value.trim().toLowerCase();
  let r=rows.filter(x=>!f||x.expr.toLowerCase().includes(f)||(x.rationale||'').toLowerCase().includes(f));
  r.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];
    if(typeof x==='string'){return sortDir*x.localeCompare(y)}
    x=x===null?-1e9:x;y=y===null?-1e9:y;return sortDir*(x-y)});
  tb.innerHTML=r.map(x=>`<tr><td class="expr">${x.expr}${x.rationale?`<div class="rat">${x.rationale}</div>`:''}</td>
    <td class="${cls(x.train_rank_ic)}">${num(x.train_rank_ic)}</td><td>${num(x.train_sharpe)}</td>
    <td>${num(x.train_turnover)}</td>
    <td class="${cls(x.test_rank_ic)}">${num(x.test_rank_ic)}</td><td>${num(x.test_sharpe)}</td></tr>`).join('');
  cnt.textContent=r.length+' / '+rows.length+' alphas';
}
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k;sortDir=(k===sortKey)?-sortDir:-1;sortKey=k;render()});
q.oninput=render;render();
"""


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def render_html(config: Dict, rows: List[Dict], reflection: Optional[str] = None) -> str:
    """Build the dashboard HTML.

    rows: list of dicts with keys expr, rationale, train_rank_ic, train_sharpe,
          train_turnover, test_rank_ic, test_sharpe (test_* may be None).
    """
    chips = "".join(
        f'<span class="chip"><b>{_esc(k)}</b> {_esc(v)}</span>'
        for k, v in config.items()
    )
    refl = ""
    if reflection:
        refl = (f'<details><summary>Reflection log</summary>'
                f'<pre>{_esc(reflection)}</pre></details>')
    head = (
        '<tr>'
        '<th data-k="expr">expression</th>'
        '<th data-k="train_rank_ic">train Rank-IC</th>'
        '<th data-k="train_sharpe">train Sharpe</th>'
        '<th data-k="train_turnover">turnover</th>'
        '<th data-k="test_rank_ic">test Rank-IC</th>'
        '<th data-k="test_sharpe">test Sharpe</th>'
        '</tr>'
    )
    data_js = json.dumps(rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaMine run {_esc(config.get('run_id', ''))}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>AlphaMine run</h1>
<p class="sub">Mined alpha library, scored on the held-out test split. Click a column to sort.</p>
<div class="chips">{chips}</div>
{refl}
<div class="bar"><input id="q" type="search" placeholder="filter by expression or rationale…">
<span class="count" id="cnt"></span></div>
<table><thead>{head}</thead><tbody id="tb"></tbody></table>
<p class="note">Test numbers are the only ones to trust. Not investment advice — every result is a
hypothesis until validated out-of-sample and forward-tested.</p>
</div><script>const DATA={data_js};{_JS}</script></body></html>"""
