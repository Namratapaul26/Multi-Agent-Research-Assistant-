import os
import re
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pipeline import run_fast_pipeline
from agents import get_mistral_llm, build_writer_chain, build_critic_chain, extract_top_url, DEFAULT_MODEL
from tools import web_search, scrape_url

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sleek Theme-Adaptive Custom CSS
st.markdown("""
<style>
    /* Global Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Card */
    .hero-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    /* Metric & Tag Pills (Theme Adaptive) */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 1rem 0 1.5rem 0;
    }
    .meta-tag {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background: rgba(125, 125, 125, 0.12);
        border: 1px solid rgba(125, 125, 125, 0.2);
    }

    /* Score Callout Card */
    .score-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        margin-bottom: 1.5rem;
    }
    .score-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #3B82F6;
        margin-bottom: 0.3rem;
    }
    
    /* Step Status Box */
    .step-box {
        padding: 0.85rem 1.2rem;
        border-radius: 10px;
        background: rgba(125, 125, 125, 0.08);
        border-left: 4px solid #3B82F6;
        margin: 0.8rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

# Check API Keys quietly in background
tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
keys_ready = bool(tavily_key and mistral_key)

# Sidebar (Minimal, Clean & Settings only)
with st.sidebar:
    st.markdown("### ⚙️ Research Controls")
    
    model_choice = st.selectbox(
        "AI Model",
        options=[
            "open-mistral-nemo",
            "open-mistral-7b",
            "mistral-small-latest",
            "codestral-latest",
        ],
        index=0,
        help="open-mistral-nemo is optimized for speed and high accuracy."
    )

    speed_mode = st.radio(
        "Research Depth",
        options=["⚡ Fast Executive (15-20s)", "🔬 Deep Comprehensive"],
        index=0,
        help="Fast mode delivers crisp, punchy reports in seconds."
    )
    is_concise = "Fast" in speed_mode

    st.markdown("---")
    
    # Session History
    st.markdown("### 📜 Past Searches")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            title_text = item["topic"][:28] + ("..." if len(item["topic"]) > 28 else "")
            if st.button(f"🔍 {title_text}", key=f"hist_btn_{i}", use_container_width=True):
                st.session_state.active_result = item
                st.session_state.selected_query = item["topic"]
                st.rerun()
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_result = None
            st.rerun()
    else:
        st.caption("No queries run yet in this session.")

    st.markdown("---")
    st.caption("Multi-Agent Assistant • Powered by LangChain & Mistral AI")

# Main Hero
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 Multi-Agent Research Assistant</div>
    <div class="hero-subtitle">Fast, autonomous intelligence gathering with real-time web search, reading, and peer review.</div>
</div>
""", unsafe_allow_html=True)

# Missing API Key Banner (only displayed if keys aren't in .env)
if not keys_ready:
    missing = []
    if not tavily_key:
        missing.append("`TAVILY_API_KEY`")
    if not mistral_key:
        missing.append("`MISTRAL_API_KEY`")
    st.error(f"⚠️ Missing environment variable: {' and '.join(missing)} in `.env`. Please add them to proceed.")
    st.stop()

# Topic Starters (Quick Pills)
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
if col_q1.button("🤖 What is Affinsys AI?", use_container_width=True):
    st.session_state.selected_query = "What is Affinsys AI and what does the company do?"
if col_q2.button("⚡ Solid-State Batteries", use_container_width=True):
    st.session_state.selected_query = "Latest breakthroughs in solid-state battery technology"
if col_q3.button("🛡️ Post-Quantum Cryptography", use_container_width=True):
    st.session_state.selected_query = "NIST Post-Quantum Cryptography standards and roadmap"
if col_q4.button("🧬 CRISPR Gene Therapy", use_container_width=True):
    st.session_state.selected_query = "Recent clinical trial advancements in CRISPR therapeutics"

# Search Input Form
with st.form(key="search_form", clear_on_submit=False):
    query_input = st.text_input(
        "Research Topic / Question",
        value=st.session_state.selected_query,
        placeholder="e.g. Current market status and key competitors of Affinsys AI",
        label_visibility="collapsed"
    )
    col_submit, col_space = st.columns([1, 4])
    with col_submit:
        submitted = st.form_submit_button("🚀 Start Research", type="primary", use_container_width=True)

# Execution Flow
if submitted:
    topic = query_input.strip()
    if not topic:
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.selected_query = topic
        progress_area = st.container()

        with progress_area:
            status_text = st.empty()
            progress_bar = st.progress(10)
            
            try:
                t_start = time.time()
                
                # Step 1: Search
                status_text.markdown('<div class="step-box">🔍 <b>Step 1/4:</b> Querying real-time web intelligence via Tavily...</div>', unsafe_allow_html=True)
                progress_bar.progress(25)
                search_data = web_search.invoke(topic)

                # Step 2: Reader
                top_url = extract_top_url(search_data)
                url_display = (top_url[:50] + "...") if len(top_url) > 50 else top_url
                status_text.markdown(f'<div class="step-box">📖 <b>Step 2/4:</b> Extracting primary source content (<i>{url_display}</i>)...</div>', unsafe_allow_html=True)
                progress_bar.progress(50)

                scraped_data = ""
                if top_url:
                    scraped_data = scrape_url.invoke(top_url)

                # Step 3: Writer
                status_text.markdown('<div class="step-box">✍️ <b>Step 3/4:</b> Synthesizing executive research report...</div>', unsafe_allow_html=True)
                progress_bar.progress(75)

                combined_research = (
                    f"WEB SEARCH FINDINGS:\n{search_data}\n\n"
                    f"PRIMARY SOURCE CONTEXT ({top_url}):\n{scraped_data[:3000]}"
                )

                llm = get_mistral_llm(model=model_choice, temperature=0.1)
                writer = build_writer_chain(llm=llm, concise=is_concise)
                report_content = writer.invoke({
                    "topic": topic,
                    "research": combined_research
                })

                # Step 4: Critic
                status_text.markdown('<div class="step-box">⚖️ <b>Step 4/4:</b> Reviewing & peer-grading report quality...</div>', unsafe_allow_html=True)
                progress_bar.progress(90)

                critic = build_critic_chain(llm=llm)
                critique_content = critic.invoke({"report": report_content})

                t_total = round(time.time() - t_start, 1)
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                # Package Result
                result = {
                    "topic": topic,
                    "report": report_content,
                    "critique": critique_content,
                    "search_results": search_data,
                    "scraped_url": top_url,
                    "scraped_content": scraped_data,
                    "model": model_choice,
                    "elapsed": t_total,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state.active_result = result
                st.session_state.history.append(result)
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Execution Error: {str(e)}")

# Display Active Result
current = st.session_state.active_result

if current:
    st.markdown("---")
    
    # Metadata bar
    st.markdown(f"""
    <div class="meta-bar">
        <span class="meta-tag">📌 <b>{current['topic']}</b></span>
        <span class="meta-tag">⚡ <b>{current.get('elapsed', 'N/A')}s</b> runtime</span>
        <span class="meta-tag">🤖 {current.get('model', 'open-mistral-nemo')}</span>
        <span class="meta-tag">📅 {current.get('timestamp', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    # Extract score if present
    score_match = re.search(r'Overall Score:\s*([0-9\.]+\s*/\s*10)', current.get("critique", ""), re.IGNORECASE)
    if score_match:
        score_val = score_match.group(1)
        st.markdown(f"""
        <div class="score-card">
            <div class="score-header">⭐ Peer Review Score: {score_val}</div>
            <div style="opacity: 0.85; font-size: 0.95rem;">Graded automatically by the Critic Agent based on analytical depth, factual grounding, and structure.</div>
        </div>
        """, unsafe_allow_html=True)

    # Organized Tabs
    tab_report, tab_critique, tab_intel = st.tabs([
        "📄 Research Report",
        "⚖️ Peer Review & Critique",
        "🌐 Sources & Intel"
    ])

    with tab_report:
        st.markdown(current.get("report", ""))
        st.divider()
        st.download_button(
            label="📥 Download Report (.md)",
            data=current.get("report", ""),
            file_name=f"report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=False
        )

    with tab_critique:
        st.markdown(current.get("critique", ""))

    with tab_intel:
        st.markdown("#### Primary Source Extracted")
        if current.get("scraped_url"):
            st.markdown(f"**Source URL:** [{current['scraped_url']}]({current['scraped_url']})")
        else:
            st.caption("No specific source URL scraped.")

        if current.get("scraped_content"):
            with st.expander("View Scraped Article Text", expanded=False):
                st.text(current["scraped_content"])

        st.markdown("#### Search Snippets Gathered")
        with st.expander("View Raw Tavily Search Findings", expanded=True):
            st.text(current.get("search_results", "No search results."))
