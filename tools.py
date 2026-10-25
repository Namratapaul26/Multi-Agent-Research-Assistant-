from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

load_dotenv()

def get_tavily_client(api_key: str = None):
    from tavily import TavilyClient
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("Tavily API key is not set. Please provide it in .env or the Streamlit sidebar.")
    return TavilyClient(api_key=key)

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        client = get_tavily_client()
        results = client.search(query=query, max_results=5)
        out = []
        for r in results.get('results', []):
            out.append(
                f"Title: {r.get('title', 'No Title')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', '')[:300]}\n"
            )
        return "\n----\n".join(out) if out else "No relevant search results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url.strip(), timeout=10, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:4000] if text else "No readable text content found at this URL."
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

if __name__ == "__main__":
    print("Testing tools directly...")
    print(scrape_url.invoke("https://example.com"))
