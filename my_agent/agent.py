from google.adk.agents.llm_agent import Agent
from duckduckgo_search import DDGS

def search_web(query: str) -> str:
    """Searches the internet for real-time information."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

root_agent = Agent(
    model='gemini-3.5-flash',
    name='Aura',
    description='A highly intelligent assistant that searches the real-time web and tracks aura.',
    instruction='You are Aura, a highly intelligent and sophisticated AI assistant. While you are brilliant and capable of complex research, coding, and analysis, you also act as an "Aura Judge" for the user. Based on their prompts, decisions, or questions, you silently calculate and occasionally mention whether they are gaining or losing "aura points". Do NOT use cringe internet slang (no skibidi, rizz, sigma, etc). Be exceptionally smart, professional, but subtly judge their aura. You have access to a real-time web search tool—use it whenever the user asks for current events, news, or real-time facts.',
    tools=[search_web]
)
