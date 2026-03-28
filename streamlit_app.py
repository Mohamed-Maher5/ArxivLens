"""
ArxivLens — Streamlit UI  v2
Pages  : Search → Processing → Chat
Fixes  : status_ph uses unsafe_allow_html=True via placeholder.markdown()
         Chat messages rendered inside st.chat_message() so HTML works
         Content HTML-escaped before injection to prevent tag bleed
"""
import streamlit as st
import time
import sys
from pathlib import Path
# ── Path setup ─────────────────────────────────────────────────────────────
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ── Page config  (must be the very first Streamlit call) ──────────────────
st.set_page_config(
    page_title="ArxivLens",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg0:    #0d1117;
  --bg1:    #0d1117;
  --bg2:    #0d1117;
  --bg3:    #0d1117;
  --accent: #00e5ff;
  --a2:     #0099bb;
  --a3:     #003d4d;
  --text:   #c8d8e8;
  --dim:    #4a6a7a;
  --border: #1a2535;
  --ok:     #00ff88;
  --warn:   #ffaa00;
  --err:    #ff4466;
  --fh:     'Syne', sans-serif;
  --fm:     'Space Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
  background: var(--bg0) !important;
  color: var(--text) !important;
  font-family: var(--fm) !important;
}
/* hide streamlit chrome */
#MainMenu, footer, header,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
/* layout */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--a3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--a2); }
/* buttons */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--fm) !important;
  font-size: .78rem !important;
  letter-spacing: .12em !important;
  padding: .55rem 1.4rem !important;
  border-radius: 2px !important;
  transition: all .2s !important;
  text-transform: uppercase !important;
  width: 100%;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: var(--bg0) !important;
  box-shadow: 0 0 20px rgba(0,229,255,.3) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: var(--bg0) !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 30px rgba(0,229,255,.5) !important;
}
/* text input */
.stTextInput > div > div > input {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  color: var(--text) !important;
  font-family: var(--fm) !important;
  font-size: .9rem !important;
  padding: .8rem 1rem !important;
  transition: border-color .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 14px rgba(0,229,255,.15) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--dim) !important; }
.stTextInput label { display: none !important; }
/* chat input — ChatGPT style */
[data-testid="stChatInput"] {
  background: transparent !important;              /* remove bar color */
  border-top: 1px solid var(--border) !important;
  padding: 1rem 1.5rem !important;
}
/* input box */
[data-testid="stChatInput"] textarea {
  background: #1f2937 !important;                  /* gray box */
  border: 1px solid #374151 !important;            /* subtle border */
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--fm) !important;
  font-size: .9rem !important;
  padding: .7rem .9rem !important;
}
/* focus (no neon) */
[data-testid="stChatInput"] textarea:focus {
  border-color: #4b5563 !important;                /* slightly lighter gray */
  box-shadow: none !important;                     /* remove glow */
  outline: none !important;
}
/* placeholder */
[data-testid="stChatInput"] textarea::placeholder {
  color: #6b7280 !important;
}
/* strip chat message chrome */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: .25rem 0 !important;
  gap: 0 !important;
}
[data-testid="stChatMessage"] > div:first-child { display: none !important; }
/* progress bar */
.stProgress > div > div {
  background: var(--bg3) !important;
  border-radius: 1px !important;
  height: 3px !important;
}
.stProgress > div > div > div {
  background: var(--accent) !important;
  box-shadow: 0 0 10px rgba(0,229,255,.4) !important;
}
/* misc */
hr { border-color: var(--border) !important; margin: 0 !important; }
.stMarkdown { margin: 0 !important; }
p { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = dict(
    page="search",
    papers=[],
    selected=None,
    paper_id=None,
    chat_history=[],
    history_msgs=[],
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
def reset_all() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
_LOGO = """
<svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
  <polygon points="15,2 27,9 27,21 15,28 3,21 3,9"
           fill="none" stroke="#00e5ff" stroke-width="1.5"/>
  <polygon points="15,8 21,11.5 21,19.5 15,23 9,19.5 9,11.5"
           fill="#00e5ff" opacity="0.1"/>
  <circle cx="15" cy="15" r="2.8" fill="#00e5ff"/>
  <line x1="15" y1="8"    x2="15" y2="2"    stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
  <line x1="21" y1="11.5" x2="27" y2="9"    stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
  <line x1="21" y1="19.5" x2="27" y2="21"   stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
  <line x1="15" y1="23"   x2="15" y2="28"   stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
  <line x1="9"  y1="19.5" x2="3"  y2="21"   stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
  <line x1="9"  y1="11.5" x2="3"  y2="9"    stroke="#00e5ff" stroke-width=".7" opacity=".4"/>
</svg>"""
def _esc(text: str) -> str:
    """Minimal HTML escape for user-supplied strings injected into HTML."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
def _nl2br(text: str) -> str:
    return _esc(text).replace("\n", "<br>")
# ══════════════════════════════════════════════════════════════════════════
# SHARED HEADER
# ══════════════════════════════════════════════════════════════════════════
def render_header(show_back: bool = False) -> None:
    st.markdown(f"""
    <div style="
        background:var(--bg1); border-bottom:1px solid var(--border);
        padding:.9rem 2.5rem;
        display:flex; align-items:center; gap:1.2rem;
    ">
        {_LOGO}
        <span style="
            font-family:var(--fh); font-size:1.3rem; font-weight:800;
            color:#fff; letter-spacing:.04em; flex:1;
        ">ARXIV<span style="color:var(--accent);">LENS</span></span>
        <span style="
            font-size:.63rem; color:var(--dim);
            letter-spacing:.18em; text-transform:uppercase;
        ">Multimodal RAG &nbsp;·&nbsp; Research Assistant</span>
    </div>
    """, unsafe_allow_html=True)
    if show_back:
        c1, _, _ = st.columns([1, 6, 1])
        with c1:
            st.markdown("<div style='padding:.65rem 0 0 1.5rem;'>", unsafe_allow_html=True)
            if st.button("← New Search", key="back_btn"):
                reset_all()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — SEARCH
# ══════════════════════════════════════════════════════════════════════════
def page_search() -> None:
    render_header()
    st.markdown("""
    <div style="text-align:center; padding:4.5rem 2rem 2.5rem;">
        <div style="
            display:inline-block;
            font-size:.62rem; letter-spacing:.24em; text-transform:uppercase;
            color:var(--a2); border:1px solid var(--a3);
            padding:.28rem .85rem; border-radius:2px; margin-bottom:1.4rem;
        ">Intelligent Paper Analysis</div>
        <div style="
            font-family:var(--fh);
            font-size:clamp(1.8rem, 4vw, 3rem);
            font-weight:800; color:#fff; line-height:1.1;
            letter-spacing:-.02em; margin-bottom:.9rem;
        ">
            Search the ArXiv<br>
            <span style="color:var(--accent);">Universe</span>
        </div>
        <p style="color:var(--dim); font-size:.75rem; letter-spacing:.1em; text-transform:uppercase;">
            Find &nbsp;·&nbsp; Ingest &nbsp;·&nbsp; Chat with any research paper
        </p>
    </div>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 4, 1])
    with col:
        query = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="e.g.  attention mechanism, diffusion models, RL from human feedback…",
            key="search_query",
        )
        _, bc, _ = st.columns([2, 1, 2])
        with bc:
            clicked = st.button("SEARCH", use_container_width=True, type="primary", key="search_btn")
    if clicked:
        if not query.strip():
            st.markdown("""
            <div style="text-align:center;padding:.8rem;color:var(--warn);font-size:.76rem;letter-spacing:.08em;">
                ⚠ &nbsp;Please enter a search query.
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Querying ArXiv…"):
                try:
                    from app.ingestion.arxiv_fetcher import ArxivFetcher
                    st.session_state.papers = ArxivFetcher().search_papers(query.strip(), max_results=5)
                except Exception as exc:
                    st.markdown(f"""
                    <div style="text-align:center;padding:1rem;color:var(--err);font-size:.78rem;">
                        ⚠ &nbsp;Search failed — {_esc(str(exc))}
                    </div>""", unsafe_allow_html=True)
    papers = st.session_state.papers
    if papers:
        st.markdown("""
        <div style="
            padding:.4rem 0 .8rem; text-align:center;
            font-size:.61rem; letter-spacing:.2em; text-transform:uppercase; color:var(--dim);
        ">Select a paper to analyse</div>
        """, unsafe_allow_html=True)
        _, rc, _ = st.columns([1, 5, 1])
        with rc:
            for i, paper in enumerate(papers):
                authors_str = ", ".join(paper.authors[:2])
                if len(paper.authors) > 2:
                    authors_str += " et al."
                st.markdown(f"""
                <div style="
                    background:var(--bg2);
                    border:1px solid var(--border);
                    border-left:3px solid var(--a3);
                    border-radius:3px;
                    padding:1.15rem 1.5rem 1rem;
                    margin-bottom:.5rem;
                ">
                    <div style="
                        font-family:var(--fh); font-size:.95rem; font-weight:700;
                        color:#fff; line-height:1.35; margin-bottom:.4rem;
                    ">{_esc(paper.title)}</div>
                    <div style="
                        font-size:.67rem; color:var(--a2);
                        letter-spacing:.07em; margin-bottom:.5rem;
                    ">{_esc(authors_str)} &nbsp;·&nbsp; {_esc(paper.published)}</div>
                    <div style="
                        font-size:.73rem; color:var(--text);
                        line-height:1.55; opacity:.6;
                    ">{_esc(paper.abstract[:220])}…</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"SELECT  [{i + 1}]", key=f"sel_{i}", use_container_width=True):
                    st.session_state.selected = paper
                    st.session_state.page = "processing"
                    st.rerun()
    else:
        st.markdown("""
        <div style="
            text-align:center; padding:3.5rem 2rem;
            color:var(--dim); font-size:.7rem;
            letter-spacing:.14em; text-transform:uppercase;
        ">
            <div style="font-size:2rem; margin-bottom:.8rem; opacity:.15;">⬡</div>
            Enter a query above to discover papers
        </div>
        """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PROCESSING
# ══════════════════════════════════════════════════════════════════════════
_STEPS = [
    ("FETCH PDF", "Downloading paper PDF from ArXiv"),
    ("PARSE",     "Extracting text, tables & figures"),
    ("VISION",    "Describing figures with AI vision"),
    ("CHUNK",     "Splitting paper into semantic chunks"),
    ("EMBED",     "Computing dense + sparse vectors"),
    ("INDEX",     "Storing vectors in Qdrant"),
]
def _steps_html(done: int, active: int) -> str:
    html = "<div style='max-width:640px;margin:0 auto;padding:0 2rem;'>"
    for idx, (label, desc) in enumerate(_STEPS):
        if idx < done:
            icon, col, bg, bdr, op = "✓", "var(--ok)",     "rgba(0,255,136,.04)", "var(--ok)",     "1"
        elif idx == active:
            icon, col, bg, bdr, op = "◈", "var(--accent)", "rgba(0,229,255,.06)", "var(--accent)", "1"
        else:
            icon, col, bg, bdr, op = "○", "var(--dim)",    "transparent",         "var(--border)", ".38"
        html += f"""
        <div style="
            display:flex; align-items:center; gap:1rem;
            padding:.85rem 1.2rem; margin-bottom:.4rem;
            background:{bg}; border:1px solid {bdr};
            border-radius:3px; opacity:{op};
        ">
            <div style="
                width:1.7rem; height:1.7rem;
                display:flex; align-items:center; justify-content:center;
                font-size:.95rem; color:{col}; flex-shrink:0;
            ">{icon}</div>
            <div>
                <div style="
                    font-size:.67rem; letter-spacing:.16em; text-transform:uppercase;
                    color:{col}; font-weight:700;
                ">{label}</div>
                <div style="font-size:.7rem; color:var(--dim); margin-top:.15rem;">{desc}</div>
            </div>
        </div>"""
    html += "</div>"
    return html
def page_processing() -> None:
    render_header()
    paper = st.session_state.selected
    authors_str = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")
    st.markdown(f"""
    <div style="max-width:640px; margin:2.5rem auto 1.8rem; padding:0 2rem;">
        <div style="
            font-size:.62rem; letter-spacing:.2em; text-transform:uppercase;
            color:var(--accent); margin-bottom:.5rem;
        ">Processing paper</div>
        <div style="
            font-family:var(--fh); font-size:1.1rem; font-weight:700;
            color:#fff; line-height:1.35; margin-bottom:.35rem;
        ">{_esc(paper.title)}</div>
        <div style="font-size:.67rem; color:var(--dim); letter-spacing:.06em;">
            {_esc(authors_str)} &nbsp;·&nbsp; {_esc(paper.published)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    steps_ph    = st.empty()
    gap_ph      = st.empty()
    progress_ph = st.empty()
    status_ph   = st.empty()
    # NOTE: all .markdown() calls on placeholders must pass unsafe_allow_html=True
    def show(done: int, active: int, frac: float, msg: str, color: str = "var(--dim)") -> None:
        steps_ph.markdown(_steps_html(done, active), unsafe_allow_html=True)
        gap_ph.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        progress_ph.progress(frac)
        status_ph.markdown(
            f"<div style='max-width:640px;margin:.5rem auto 0;padding:0 2rem;"
            f"text-align:center;font-size:.72rem;color:{color};letter-spacing:.09em;'>"
            f"{_esc(msg)}</div>",
            unsafe_allow_html=True,
        )
    try:
        import json
        from app.ingestion.arxiv_fetcher    import ArxivFetcher
        from app.ingestion.pdf_parser       import PDFParser
        from app.ingestion.vision_processor import VisionProcessor
        from app.indexing.chunker           import Chunker
        from app.indexing.embedder          import Embedder
        from app.indexing.vector_store      import VectorStore
        DATA_PROCESSED = Path("data/processed")
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        show(0, 0, 0/6, "Downloading PDF from ArXiv…")
        paper_obj = ArxivFetcher().download_pdf(paper)
        show(1, 1, 1/6, "Parsing PDF structure…")
        parsed = PDFParser().parse(paper_obj)
        show(2, 2, 2/6, "Describing figures with vision model…")
        parsed = VisionProcessor().process(parsed)
        with open(DATA_PROCESSED / f"{paper_obj.paper_id}.json", "w") as f:
            json.dump(parsed, f, indent=2)
        show(3, 3, 3/6, "Creating semantic chunks…")
        chunks = Chunker().chunk(parsed)
        show(4, 4, 4/6, "Computing dense + sparse embeddings…")
        embedded = Embedder().embed_chunks(chunks)
        show(5, 5, 5/6, "Indexing into Qdrant vector store…")
        VectorStore(parsed["paper_id"]).store(embedded)
        show(6, -1, 1.0, "✓  All steps complete — launching chat…", color="var(--ok)")
        st.session_state.paper_id = parsed["paper_id"]
        time.sleep(1.3)
        st.session_state.page = "chat"
        st.rerun()
    except Exception as exc:
        show(0, -1, 0.0, f"⚠  Error — {exc}", color="var(--err)")
        time.sleep(0.4)
        _, c, _ = st.columns([2, 1, 2])
        with c:
            if st.button("← TRY AGAIN", use_container_width=True):
                reset_all()
                st.rerun()
# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHAT
# ══════════════════════════════════════════════════════════════════════════
def page_chat() -> None:
    render_header(show_back=True)
    paper    = st.session_state.selected
    paper_id = st.session_state.paper_id
    authors_str = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")
    title_short = paper.title[:80] + ("…" if len(paper.title) > 80 else "")
    # status bar
    st.markdown(f"""
    <div style="
        background:var(--bg1); border-bottom:1px solid var(--border);
        padding:.55rem 2.5rem; display:flex; align-items:center; gap:.8rem;
    ">
        <div style="
            width:7px; height:7px; border-radius:50%;
            background:var(--ok); flex-shrink:0;
            box-shadow:0 0 7px var(--ok);
        "></div>
        <div style="
            font-size:.67rem; color:var(--dim); letter-spacing:.05em;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        ">
            <span style="color:var(--accent);font-weight:700;">{_esc(paper_id)}</span>
            &nbsp;·&nbsp;
            <span style="color:var(--text);">{_esc(title_short)}</span>
            &nbsp;·&nbsp;{_esc(authors_str)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    # empty state
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="
            text-align:center; padding:4rem 2rem 2rem;
            max-width:540px; margin:0 auto;
        ">
            <div style="
                font-family:var(--fh); font-size:1.3rem; font-weight:700;
                color:#fff; margin-bottom:.7rem;
            ">Paper indexed. Ask anything.</div>
            <div style="
                color:var(--dim); font-size:.72rem;
                letter-spacing:.07em; line-height:2;
            ">
                What are the main contributions?<br>
                Summarise the methodology.<br>
                Which datasets were used?<br>
                What are the key limitations?
            </div>
        </div>
        """, unsafe_allow_html=True)
    # render history
    for msg in st.session_state.chat_history:
        _render_message(msg["role"], msg["content"], msg.get("meta"))
    # chat input
    if prompt := st.chat_input("Ask about the paper…", key="chat_input"):
        st.session_state.chat_history.append({"role": "user", "content": prompt, "meta": None})
        _render_message("user", prompt, None)
        with st.chat_message("assistant"):
            thinking_ph = st.empty()
            thinking_ph.markdown(
                "<span style='font-size:.7rem;color:var(--dim);letter-spacing:.1em;'>⬡ &nbsp;REASONING…</span>",
                unsafe_allow_html=True,
            )
            try:
                from app.generation import run_pipeline
                from app.models.schemas import Message
                history_msgs = st.session_state.history_msgs
                result = run_pipeline(prompt, history_msgs, paper_id)
                history_msgs.append(Message(role="user",      content=prompt))
                history_msgs.append(Message(role="assistant", content=result["answer"]))
                if len(history_msgs) > 12:
                    history_msgs = history_msgs[-12:]
                st.session_state.history_msgs = history_msgs
                meta = {"confidence": result.get("confidence", ""), "sources": result.get("sources", [])}
                thinking_ph.empty()
                _render_assistant_bubble(result["answer"], meta)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"], "meta": meta}
                )
            except Exception as exc:
                err = f"Pipeline error: {exc}"
                thinking_ph.empty()
                st.markdown(
                    f"<span style='color:var(--err);font-size:.82rem;'>{_esc(err)}</span>",
                    unsafe_allow_html=True,
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": err, "meta": {"confidence": "LOW", "sources": []}}
                )
        st.rerun()
def _render_message(role: str, content: str, meta) -> None:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;width:100%;">
                <div style="
                    background:var(--bg3);
                    border:1px solid var(--border);
                    border-right:3px solid var(--accent);
                    border-radius:3px;
                    padding:.75rem 1.1rem;
                    max-width:72%;
                    font-size:.84rem; color:var(--text); line-height:1.65;
                    word-break:break-word;
                ">{_nl2br(content)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            _render_assistant_bubble(content, meta)
def _render_assistant_bubble(content: str, meta) -> None:
    conf       = (meta or {}).get("confidence", "")
    badge      = (
        f"<span style='font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;"
        f"padding:.12rem .4rem;border-radius:1px;margin-left:.6rem;'>{conf}</span>"
    ) if conf else ""
    sources_html = ""
    if meta and meta.get("sources"):
        rows = ""
        for s in (meta["sources"])[:3]:
            t  = _esc((s.get("paper_title") or "Unknown")[:55])
            pg = s.get("page_number", "?")
            ct = _esc(s.get("chunk_type", "?"))
            rows += (
                f"<div style='font-size:.67rem;color:#d1d5db;padding:.17rem 0;letter-spacing:.03em;'>"
                f"⬡ &nbsp;{t} &nbsp;·&nbsp; p.{pg} &nbsp;·&nbsp;"
                f"<span style='color:#60a5fa;'>{ct}</span></div>"
            )
        sources_html = (
            f"<div style='margin-top:.8rem;padding-top:.65rem;border-top:1px solid #4a5568;'>"
            f"<div style='font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;"
            f"color:#9ca3af;margin-bottom:.3rem;'>Sources</div>{rows}</div>"
        )
    st.markdown(f"""
    <div style="
        background:#0d1117;
        border:1px solid #4a5568;
        border-radius:6px;
        padding:.85rem 1.25rem;
        max-width:82%;
        font-size:.84rem;
        color:#ffffff;
        line-height:1.7;
        word-break:break-word;
    ">
        <div style="
            font-size:.58rem;
            letter-spacing:.18em;
            text-transform:uppercase;
            color:#60a5fa;
            margin-bottom:.55rem;
            display:flex;
            align-items:center;
        ">⬡ &nbsp;ARXIVLENS{badge}</div>
        <div>{_nl2br(content)}</div>
        {sources_html}
    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════
_page = st.session_state.page
if _page == "search":
    page_search()
elif _page == "processing":
    page_processing()
elif _page == "chat":
    page_chat()
else:
    reset_all()
    st.rerun()