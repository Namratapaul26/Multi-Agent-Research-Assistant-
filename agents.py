import os
import re
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()

DEFAULT_MODEL = "open-mistral-nemo"

def get_mistral_llm(model: str = DEFAULT_MODEL, temperature: float = 0.1):
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("Mistral API key is missing. Please set MISTRAL_API_KEY in your .env file.")
    return ChatMistralAI(
        model=model,
        api_key=key,
        temperature=temperature
    )

def extract_top_url(search_results: str) -> str:
    """Extracts the first valid HTTP/HTTPS URL from search results."""
    urls = re.findall(r'https?://[^\s\n"\'<>]+', search_results)
    for u in urls:
        if not any(blocked in u for blocked in ["google.com", "bing.com"]):
            return u
    return urls[0] if urls else ""

def build_writer_chain(llm=None, concise: bool = True):
    if llm is None:
        llm = get_mistral_llm()

    length_guidance = (
        "Write a concise, high-impact executive report (approx. 500-700 words)."
        if concise else
        "Write a comprehensive, deep-dive research report (approx. 900-1200 words)."
    )

    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior research analyst. {length_guidance}
Synthesize the gathered research into a structured markdown report with the following sections:
- # [Topic Title]
- ## Executive Summary (concise overview)
- ## Key Findings (3-4 bullet-pointed insights with technical facts)
- ## Strategic Implications & Outlook
- ## Sources (clickable markdown links using URLs from research)

Be factual, objective, analytical, and avoid repetitive filler."""),
        ("human", "Topic: {topic}\n\nGathered Research:\n{research}")
    ])
    return writer_prompt | llm | StrOutputParser()

def build_critic_chain(llm=None):
    if llm is None:
        llm = get_mistral_llm()
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior peer reviewer. In under 150 words, evaluate the report strictly in this format:

### Overall Score: X/10

**Strengths:**
- [Bullet 1]
- [Bullet 2]

**Areas to Improve:**
- [Bullet 1]
- [Bullet 2]

**Verdict:** [One crisp sentence]"""),
        ("human", "Report to review:\n{report}")
    ])
    return critic_prompt | llm | StrOutputParser()
