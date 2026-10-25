# 🔬 Multi-Agent Research Assistant

An autonomous multi-agent research pipeline that conducts comprehensive web investigations, scrapes in-depth technical sources, synthesizes structured reports, and performs automated peer reviews.

Built with **LangChain**, **Mistral AI**, **Tavily Search**, **BeautifulSoup**, and **Streamlit**.

---

## 🏗️ Architecture & Multi-Agent Workflow

```
       [User Input: Research Topic]
                    │
                    ▼
       ┌────────────────────────┐
       │     Search Agent       │  ◄── Tavily Web Search API
       │ (Gathers URLs & context)│
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │     Reader Agent       │  ◄── BeautifulSoup Web Scraper
       │(Deep scrape best source)│
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │     Writer Agent       │  ◄── Mistral LLM
       │ (Drafts markdown report)│
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │     Critic Agent       │  ◄── Mistral LLM
       │(Peer reviews & grades)  │
       └────────────┬───────────┘
                    │
                    ▼
        [Interactive Streamlit UI]
```

1. **🔍 Search Agent**: Queries Tavily for real-time web results, snippets, and authoritative URLs.
2. **📖 Reader Agent**: Identifies the single most relevant URL and deeply scrapes its article text.
3. **✍️ Writer Agent**: Synthesizes the search results and in-depth scraped text into an extensive, professional research report.
4. **⚖️ Critic Agent**: Rigorously evaluates the drafted report for depth, factual backing, strengths, and areas to improve, assigning an overall score out of 10.

---

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have Python 3.10+ and your virtual environment configured:

```bash
# If using virtualenv or uv
uv pip install -r requirements.txt
```

### 2. Configure API Keys

Create or update `.env` in the root directory:

```env
TAVILY_API_KEY=your_tavily_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

*(Note: You can also enter or override these API keys directly within the Streamlit sidebar UI).*

### 3. Run the Streamlit Application

Launch the interactive web application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🌟 Key Features

- **Intuitive Web UI**: Modern Streamlit interface with quick starter chips and model selector.
- **Real-Time Step Tracking**: Visual progress indicators tracking every stage of the multi-agent pipeline.
- **Tabbed Results Explorer**:
  - 📄 **Research Report**: Rendered markdown report with one-click `.md` download.
  - ⚖️ **Peer Review & Critique**: Complete score and constructive reviewer breakdown.
  - 📖 **Deep Scraped Context**: Full view of content extracted from the primary source.
  - 🔍 **Search Intelligence**: Live snippets and URLs gathered by the Search Agent.
- **Session History**: Easily navigate between previously researched queries in your active session.
- **Safe Encoding**: Handles cross-platform unicode symbols on Windows/Linux/macOS without errors.

