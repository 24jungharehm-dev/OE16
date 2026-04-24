"""
Social Media Strategy Planner — Multi-Agent Pipeline
A Streamlit app orchestrating 5 AI agents using Groq (LLM) & Tavily (search).
API keys are loaded from .env — never asked from the user.
"""

import json, os, time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agents.strategy_agent import run_strategy_planner_agent
from agents.audience_agent import run_audience_research_agent
from agents.content_agent import run_content_planner_agent
from agents.schedular_agent import run_scheduler_agent
from agents.judge_agent import run_judge_agent

load_dotenv()

# ── Auto-load API keys (never shown to user) ──
GROK_KEY = os.getenv("GROK_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")


def with_retry(fn, retries=3, base_wait=15):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = base_wait * (2 ** attempt)
                st.warning(f"⏳ Rate limited — waiting {wait}s ({attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    return fn()


# ── Page Config ──
st.set_page_config(page_title="Social Media Strategy Planner", page_icon="📱", layout="wide", initial_sidebar_state="expanded")

# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary: #7c3aed; --primary-light: #a78bfa; --primary-dark: #5b21b6;
    --accent: #06b6d4; --accent-light: #67e8f9;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --bg-dark: #0f0a1a; --bg-card: #1a1230; --bg-card-hover: #231a40;
    --text-primary: #f1f0f5; --text-secondary: #a09bb5; --text-muted: #6b6580;
    --border: rgba(124,58,237,0.15); --glow: rgba(124,58,237,0.25);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #06b6d4 100%);
    padding: 2.5rem 3rem; border-radius: 20px; margin-bottom: 2rem;
    color: white; position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(124,58,237,0.3);
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -20%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { margin:0; font-size:2.4rem; font-weight:900; letter-spacing:-0.03em; position:relative; }
.hero p { margin:0.5rem 0 0; opacity:0.85; font-size:1rem; font-weight:400; position:relative; }
.hero .badge {
    display:inline-block; background:rgba(255,255,255,0.15); backdrop-filter:blur(10px);
    padding:0.3rem 0.8rem; border-radius:50px; font-size:0.75rem; font-weight:600;
    margin-top:0.8rem; border:1px solid rgba(255,255,255,0.2); position:relative;
}

/* Agent Cards */
.agent-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, #150e28 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; margin: 0.8rem 0; color: var(--text-primary);
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.agent-card:hover {
    transform: translateY(-3px); box-shadow: 0 12px 40px rgba(124,58,237,0.2);
    border-color: var(--primary-light);
}
.agent-card h3 { color: var(--primary-light); margin:0 0 0.6rem; font-weight:700; font-size:1.05rem; }

/* Metrics */
.metric-box {
    background: linear-gradient(145deg, var(--bg-card) 0%, #150e28 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}
.metric-box:hover { transform: translateY(-2px); box-shadow: 0 8px 30px var(--glow); }
.metric-val { font-size:2.2rem; font-weight:900; background:linear-gradient(135deg,var(--primary-light),var(--accent)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.metric-lbl { font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:0.3rem; }

/* Score Badge */
.score-pill {
    display:inline-block; background:linear-gradient(135deg,var(--primary),#2563eb);
    color:#fff; padding:0.5rem 1.4rem; border-radius:50px; font-weight:800; font-size:1.4rem;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}
.score-pill.low { background:linear-gradient(135deg,var(--danger),#dc2626); box-shadow:0 4px 20px rgba(239,68,68,0.4); }

/* Verdict */
.verdict-pass {
    background:linear-gradient(135deg,var(--success),#059669); color:#fff;
    padding:0.4rem 1.2rem; border-radius:50px; font-weight:700; display:inline-block;
    box-shadow:0 4px 15px rgba(16,185,129,0.3);
}
.verdict-fail {
    background:linear-gradient(135deg,var(--warning),#d97706); color:#000;
    padding:0.4rem 1.2rem; border-radius:50px; font-weight:700; display:inline-block;
    box-shadow:0 4px 15px rgba(245,158,11,0.3);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0a1a 0%, #1a1040 50%, #0f0a1a 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--text-secondary); }

/* Pipeline Steps */
.pipe-step {
    display:flex; align-items:center; gap:0.8rem; padding:0.7rem 1rem;
    border-radius:12px; margin:0.4rem 0; font-size:0.9rem; color:var(--text-secondary);
    transition: all 0.3s ease;
}
.pipe-step .num {
    width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.8rem; border:2px solid var(--border); color:var(--text-muted);
    flex-shrink:0;
}
.pipe-step.active { background:rgba(124,58,237,0.1); border-left:3px solid var(--primary); }
.pipe-step.active .num { background:var(--primary); color:#fff; border-color:var(--primary); }
.pipe-step.done { background:rgba(16,185,129,0.08); border-left:3px solid var(--success); }
.pipe-step.done .num { background:var(--success); color:#fff; border-color:var(--success); }

/* Glow button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, #2563eb 100%) !important;
    border: none !important; font-weight: 700 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 40px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Dividers */
hr { border:0; height:1px; background:linear-gradient(to right,transparent,var(--border),transparent); margin:2rem 0; }

/* Hide streamlit chrome */
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

/* Day card inside calendar */
.day-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, #150e28 100%);
    border:1px solid var(--border); border-radius:14px; padding:1.3rem; margin:0.6rem 0;
    color:var(--text-primary); transition:all 0.3s ease;
}
.day-card:hover { border-color:var(--primary-light); transform:translateY(-2px); box-shadow:0 8px 30px rgba(124,58,237,0.15); }
.day-card h4 { color:var(--accent-light); margin:0 0 0.5rem; font-weight:700; }
.day-card .hashtags { color:var(--primary-light); font-size:0.82rem; }
.day-card .cta { color:var(--success); font-size:0.82rem; }
.day-card .hook { color:var(--text-muted); font-size:0.78rem; font-style:italic; }

/* Dimension score bar */
.dim-score { text-align:center; padding:0.6rem 0.3rem; }
.dim-val { font-size:1.6rem; font-weight:800; }
.dim-lbl { font-size:0.68rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ──
st.markdown("""
<div class="hero">
    <h1>📱 Social Media Strategy Planner</h1>
    <p>AI-powered multi-agent pipeline that researches, plans, schedules & evaluates your social media strategy</p>
    <span class="badge">✨ Powered by Groq LLM &amp; Tavily Search</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_name = st.selectbox(
        "🧠 Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it",
         "meta-llama/llama-4-scout-17b-16e-instruct"],
        index=0,
        help="llama-3.1-8b-instant has higher free-tier limits",
    )

    max_revisions = st.slider("🔄 Max revision rounds", 0, 3, 1,
        help="How many times the Judge can send strategy back for improvement")

    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")
    agents_info = [
        ("1", "🧠", "Strategy Planner"),
        ("2", "👥", "Audience Researcher"),
        ("3", "📝", "Content Planner"),
        ("4", "📅", "Scheduler"),
        ("5", "⚖️", "Judge & Evaluator"),
    ]
    for num, icon, name in agents_info:
        st.markdown(f'<div class="pipe-step"><span class="num">{num}</span>{icon} {name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # API status indicator (no input fields!)
    if GROK_KEY and TAVILY_KEY:
        st.success("✅ API keys loaded from environment")
    else:
        missing = []
        if not GROK_KEY: missing.append("GROK_API_KEY")
        if not TAVILY_KEY: missing.append("TAVILY_API_KEY")
        st.error(f"❌ Missing in .env: {', '.join(missing)}")

# ── Main Input ──
col1, col2 = st.columns([2, 1])
with col1:
    product = st.text_input("🏷️ Brand / Product", value="summer sunglasses",
        placeholder="e.g., organic skincare, fitness app, streetwear brand...")
with col2:
    platform = st.selectbox("📲 Target Platform",
        ["Instagram", "Twitter / X", "LinkedIn", "TikTok", "Facebook", "YouTube"], index=0)

st.markdown("")
run_btn = st.button("🚀 Generate Strategy", type="primary", use_container_width=True)

# ── Pipeline Execution ──
if run_btn:
    if not GROK_KEY or not TAVILY_KEY:
        st.error("⚠️ API keys missing. Add GROK_API_KEY and TAVILY_API_KEY to your .env file.")
        st.stop()

    client = OpenAI(api_key=GROK_KEY, base_url="https://api.groq.com/openai/v1")

    strategy_data = audience_data = content_data = schedule_data = judge_data = None
    revision_round = 0
    feedback = None

    while revision_round <= max_revisions:
        rl = f"  — Revision {revision_round}" if revision_round > 0 else ""

        # Step 1
        with st.status(f"🧠 Strategy Planner{rl}...", expanded=(revision_round == 0)) as s:
            st.write("Researching trends & crafting strategy...")
            t0 = time.time()
            try:
                strategy_data = with_retry(lambda: run_strategy_planner_agent(
                    client=client, product=product, platform=platform,
                    tavily_api_key=TAVILY_KEY, feedback=feedback, model=model_name))
                s.update(label=f"✅ Strategy Planner ({time.time()-t0:.1f}s){rl}", state="complete")
                st.json(strategy_data)
            except Exception as e:
                s.update(label="❌ Strategy Planner Failed", state="error"); st.error(str(e)); st.stop()

        # Step 2
        with st.status("👥 Audience Research...", expanded=False) as s:
            st.write("Profiling target audiences...")
            t0 = time.time()
            try:
                audience_data = with_retry(lambda: run_audience_research_agent(
                    client=client, strategy_summary=json.dumps(strategy_data, indent=2),
                    platform=platform, tavily_api_key=TAVILY_KEY, model=model_name))
                s.update(label=f"✅ Audience Research ({time.time()-t0:.1f}s)", state="complete")
                st.json(audience_data)
            except Exception as e:
                s.update(label="❌ Audience Research Failed", state="error"); st.error(str(e)); st.stop()

        # Step 3
        with st.status("📝 Content Planner...", expanded=False) as s:
            st.write("Building 7-day content calendar...")
            t0 = time.time()
            try:
                content_data = with_retry(lambda: run_content_planner_agent(
                    client=client, strategy_summary=json.dumps(strategy_data, indent=2),
                    audience_summary=json.dumps(audience_data, indent=2),
                    platform=platform, product=product, model=model_name))
                s.update(label=f"✅ Content Planner ({time.time()-t0:.1f}s)", state="complete")
                st.json(content_data)
            except Exception as e:
                s.update(label="❌ Content Planner Failed", state="error"); st.error(str(e)); st.stop()

        # Step 4
        with st.status("📅 Scheduler...", expanded=False) as s:
            st.write("Optimizing posting schedule...")
            t0 = time.time()
            try:
                schedule_data = with_retry(lambda: run_scheduler_agent(
                    client=client, content_plan=json.dumps(content_data, indent=2),
                    audience_summary=json.dumps(audience_data, indent=2),
                    platform=platform, model=model_name))
                s.update(label=f"✅ Scheduler ({time.time()-t0:.1f}s)", state="complete")
                st.json(schedule_data)
            except Exception as e:
                s.update(label="❌ Scheduler Failed", state="error"); st.error(str(e)); st.stop()

        # Step 5
        with st.status("⚖️ Judge & Evaluator...", expanded=True) as s:
            st.write("Evaluating strategy quality...")
            t0 = time.time()
            try:
                judge_data = with_retry(lambda: run_judge_agent(
                    client=client, strategy=json.dumps(strategy_data, indent=2),
                    audience=json.dumps(audience_data, indent=2),
                    content=json.dumps(content_data, indent=2),
                    schedule=json.dumps(schedule_data, indent=2), model=model_name))
                score = judge_data.get("overall_score", 0)
                verdict = judge_data.get("verdict", "UNKNOWN")
                if verdict == "APPROVED" or revision_round >= max_revisions:
                    s.update(label=f"✅ Judge — Score: {score}/10 ({time.time()-t0:.1f}s)", state="complete")
                else:
                    s.update(label=f"🔄 Judge — Score: {score}/10 — Revision needed", state="running")
                st.json(judge_data)
            except Exception as e:
                s.update(label="❌ Judge Failed", state="error"); st.error(str(e)); st.stop()

        verdict = judge_data.get("verdict", "APPROVED")
        if verdict == "APPROVED" or revision_round >= max_revisions:
            break
        else:
            improvements = judge_data.get("critical_improvements", [])
            suggestions = judge_data.get("suggestions", [])
            feedback = (f"Score: {judge_data.get('overall_score','?')}/10\n"
                        f"Critical improvements needed:\n" + "\n".join(f"- {i}" for i in improvements)
                        + "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions))
            revision_round += 1
            st.warning(f"🔄 Strategy scored {judge_data.get('overall_score','?')}/10 — revision round {revision_round}...")

    # ── Results Dashboard ──
    st.markdown("---")
    st.markdown("## 📊 Strategy Results Dashboard")

    score = judge_data.get("overall_score", 0)
    verdict = judge_data.get("verdict", "UNKNOWN")
    calendar = content_data.get("weekly_calendar", [])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{score}/10</div><div class="metric-lbl">Overall Score</div></div>', unsafe_allow_html=True)
    with c2:
        vc = "verdict-pass" if verdict == "APPROVED" else "verdict-fail"
        st.markdown(f'<div class="metric-box"><div class="{vc}">{verdict}</div><div class="metric-lbl" style="margin-top:0.5rem">Verdict</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{revision_round}</div><div class="metric-lbl">Revisions</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(calendar)}</div><div class="metric-lbl">Content Pieces</div></div>', unsafe_allow_html=True)

    # Dimension scores
    dim_scores = judge_data.get("dimension_scores", {})
    if dim_scores:
        st.markdown("")
        st.markdown("### 📈 Evaluation Dimensions")
        dims = list(dim_scores.keys())
        cols = st.columns(len(dims))
        for i, dim in enumerate(dims):
            with cols[i]:
                val = dim_scores[dim]
                color = "#10b981" if val >= 7 else "#ef4444"
                label = dim.replace("_", " ").title()
                st.markdown(f'<div class="dim-score"><div class="dim-val" style="color:{color}">{val}</div><div class="dim-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Expandable sections
    with st.expander("🧠 Strategy Details", expanded=False):
        if strategy_data:
            st.markdown(f"**Brand Summary:** {strategy_data.get('brand_summary', 'N/A')}")
            st.markdown("**Goals:**")
            for g in strategy_data.get("goals", []): st.markdown(f"- {g}")
            st.markdown(f"**Tone & Voice:** {strategy_data.get('tone_and_voice', 'N/A')}")
            st.markdown("**Key Themes:**")
            for t in strategy_data.get("key_themes", []): st.markdown(f"- 🎯 {t}")
            st.markdown(f"**Platform Strategy:** {strategy_data.get('platform_strategy', 'N/A')}")
            st.markdown("---"); st.json(strategy_data)

    with st.expander("👥 Audience Research", expanded=False):
        if audience_data:
            p = audience_data.get("primary_audience", {})
            if p:
                st.markdown(f"**Age Range:** {p.get('age_range','N/A')}")
                st.markdown(f"**Gender Split:** {p.get('gender_split','N/A')}")
                st.markdown(f"**Income Level:** {p.get('income_level','N/A')}")
                interests = p.get("interests", [])
                if interests: st.markdown("**Interests:** " + ", ".join(interests))
            psy = audience_data.get("psychographics", {})
            if psy:
                st.markdown("**Pain Points:**")
                for pp in psy.get("pain_points", []): st.markdown(f"- 😤 {pp}")
            st.markdown("---"); st.json(audience_data)

    with st.expander("📝 Content Calendar (7 days)", expanded=True):
        if content_data:
            for day_plan in calendar:
                day = day_plan.get("day", "?")
                post_type = day_plan.get("post_type", "Post")
                caption = day_plan.get("caption", "")
                hashtags = day_plan.get("hashtags", [])
                cta = day_plan.get("cta", "")
                hook = day_plan.get("engagement_hook", "")
                st.markdown(
                    f'<div class="day-card">'
                    f'<h4>📅 {day} — {post_type}</h4>'
                    f'<p style="font-size:0.92rem;color:#e0ddf0;">{caption}</p>'
                    f'<p class="hashtags">{" ".join(hashtags)}</p>'
                    f'<p class="cta">🎯 CTA: {cta}</p>'
                    f'<p class="hook">💡 {hook}</p>'
                    f'</div>', unsafe_allow_html=True)

    with st.expander("📅 Posting Schedule", expanded=False):
        if schedule_data:
            st.markdown(f"**Posting Frequency:** {schedule_data.get('posting_frequency', 'N/A')}")
            opt = schedule_data.get("optimal_times", {})
            if opt:
                st.markdown("**Optimal Times:**")
                for slot in opt.get("primary_slots", []): st.markdown(f"- 🕐 {slot}")
                st.markdown(f"**Rationale:** {opt.get('rationale', 'N/A')}")
            tips = schedule_data.get("scheduling_tips", [])
            if tips:
                st.markdown("**Pro Tips:**")
                for tip in tips: st.markdown(f"- 💡 {tip}")
            st.markdown("---"); st.json(schedule_data)

    with st.expander("⚖️ Judge Evaluation", expanded=False):
        if judge_data:
            for s in judge_data.get("strengths", []): st.markdown(f"- ✅ {s}")
            for w in judge_data.get("weaknesses", []): st.markdown(f"- ⚠️ {w}")
            st.markdown(f"**Summary:** {judge_data.get('summary', 'N/A')}")
            st.markdown("---"); st.json(judge_data)

    # Download
    st.markdown("---")
    full_report = {"product": product, "platform": platform, "revision_rounds": revision_round,
                   "strategy": strategy_data, "audience_research": audience_data,
                   "content_plan": content_data, "posting_schedule": schedule_data, "evaluation": judge_data}
    report_json = json.dumps(full_report, indent=2)
    md_report = f"""# 📱 Social Media Strategy Report\n## {product} — {platform}\n\n**Score: {score}/10 — {verdict}** | Revisions: {revision_round}\n\n---\n\n## 🧠 Strategy\n```json\n{json.dumps(strategy_data, indent=2)}\n```\n\n## 👥 Audience\n```json\n{json.dumps(audience_data, indent=2)}\n```\n\n## 📝 Content\n```json\n{json.dumps(content_data, indent=2)}\n```\n\n## 📅 Schedule\n```json\n{json.dumps(schedule_data, indent=2)}\n```\n\n## ⚖️ Evaluation\n```json\n{json.dumps(judge_data, indent=2)}\n```"""

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Report (Markdown)", md_report,
            f"strategy_report_{product.replace(' ', '_')}.md", "text/markdown", use_container_width=True)
    with d2:
        st.download_button("📥 Download Report (JSON)", report_json,
            f"strategy_report_{product.replace(' ', '_')}.json", "application/json", use_container_width=True)
